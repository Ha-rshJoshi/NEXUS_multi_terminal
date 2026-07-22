"""Single source of truth for the four model architectures (LSTM/Prophet/MNN/
XGBoost), training, scaling, the data pipeline, purged time-series splitting,
and the model registry cache -- imported by both models.py and evaluation.py
so their results are comparable."""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sqlalchemy import create_engine, text
import xgboost as xgb
import joblib
from sklearn.preprocessing import MinMaxScaler

from chroma_store import get_news_collection
from repro import set_seed
from target_utils import TARGET_COLUMN, compute_next_day_target

# ----------------------------------------------------------------------
# GLOBAL CONFIG  (single source of truth -- every entrypoint reads this)
# ----------------------------------------------------------------------
SEED = 42
Days = 15  # sequence lookback window, also used as the purge width
WEIGHTS_DIR = "saved_weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

DIRECTION_PENALTY = 2.0      # wrong-sign errors cost 2x a same-sign magnitude error

CONFIG = {
    "Sequence_Days": Days,
    "Learning_Rate": 0.005,
    "LSTM_Hidden_Units": 64,
    "LSTM_Epochs": 100,
    # Both off: ~1 trading-year of history can't reliably estimate yearly
    # seasonality, and a stationary daily-return series has no strong prior
    # for day-of-week effects. Keeps Prophet's fit to trend + changepoints.
    "Prophet_Weekly_Seasonality": False,
    "Prophet_Yearly_Seasonality": False,
    "Prophet_Changepoint_Prior_Scale": 0.05,  # Prophet's own default -- trend flexibility
    "MNN_Hidden_Units": 32,
    "MNN_Epochs": 100,
    "MNN_Learning_Rate": 0.002,
    "XGBoost_Estimators": 60,
    "XGBoost_Max_Depth": 4,
    "XGBoost_Learning_Rate": 0.05,
    "Direction_Penalty": DIRECTION_PENALTY,
}

NAIVE_KEY = "Naive Baseline"
MODEL_NAMES = ["LSTM", "Prophet", "MNN", "XGBoost"]

# ----------------------------------------------------------------------
# DB -- plain SQLAlchemy engine for standalone scripts/pipelines that run
# outside Flask's app context. Table names match db_models.py exactly, so
# the ORM (writes) and this engine (raw-SQL reads) see the same rows.
# ----------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///nexus.db")
engine = create_engine(DATABASE_URL)

news_collect = get_news_collection()


# ----------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------
class QuantLSTM(nn.Module):
    def __init__(self, input_size, hidden_layers=64, output_size=1):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(input_size, hidden_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layers, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


class MultiplicativeLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_size, output_size) * 0.01)

    def forward(self, x):
        prod_matrix = torch.matmul(x, self.weights)
        return torch.prod(prod_matrix, dim=-1, keepdim=True)


# ----------------------------------------------------------------------
# ASYMMETRIC DIRECTIONAL LOSS
# ----------------------------------------------------------------------
class AsymmetricDirectionalLoss(nn.Module):
    """Squared error multiplied by `direction_penalty` whenever the
    prediction's sign disagrees with the target's sign, so the model is
    trained to prioritize getting the direction right, not just minimizing
    average magnitude error."""

    def __init__(self, direction_penalty: float = DIRECTION_PENALTY):
        super().__init__()
        self.direction_penalty = direction_penalty

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        squared_error = (preds - targets) ** 2
        wrong_direction = (torch.sign(preds) != torch.sign(targets)).float()
        weight = 1.0 + wrong_direction * (self.direction_penalty - 1.0)
        return torch.mean(squared_error * weight)


def asymmetric_xgb_objective(y_true: np.ndarray, y_pred: np.ndarray):
    """Same asymmetric penalty as AsymmetricDirectionalLoss, expressed as
    (gradient, hessian) of weighted squared error so XGBoost's tree
    boosting optimizes it directly.

    Signature matters: build_xgb() passes this to
    xgb.XGBRegressor(objective=...), whose sklearn wrapper calls a custom
    objective as plain (y_true, y_pred) numpy arrays -- not the native
    Learning API's (y_pred, dtrain) convention, where dtrain is a DMatrix
    requiring .get_label(). Calling .get_label() here would fail since
    y_true is already a plain array.
    """
    residual = y_pred - y_true
    wrong_direction = (np.sign(y_pred) != np.sign(y_true)).astype(np.float64)
    weight = 1.0 + wrong_direction * (DIRECTION_PENALTY - 1.0)
    grad = 2.0 * residual * weight
    hess = 2.0 * weight
    return grad, hess


# ----------------------------------------------------------------------
# MODEL FACTORIES  (each factory re-seeds first -> deterministic init)
# ----------------------------------------------------------------------
def build_lstm(feature_count):
    set_seed(SEED)
    return QuantLSTM(feature_count, CONFIG["LSTM_Hidden_Units"])


def build_prophet():
    """Factory for a fresh Prophet instance. Unlike the other three
    factories, no seeding or feature_count is needed -- Prophet fits
    directly on the (date, target_return_pct) series and never sees the
    engineered feature matrix the other three models train on."""
    from prophet import Prophet
    return Prophet(
        changepoint_prior_scale=CONFIG["Prophet_Changepoint_Prior_Scale"],
        weekly_seasonality=CONFIG["Prophet_Weekly_Seasonality"],
        yearly_seasonality=CONFIG["Prophet_Yearly_Seasonality"],
        daily_seasonality=False,
    )


def prophet_frame(dates, y) -> pd.DataFrame:
    """Builds the 'ds'/'y' dataframe Prophet's fit()/predict() expect, from
    this project's own parallel (dates, target_return_pct) arrays."""
    return pd.DataFrame({"ds": pd.to_datetime(np.asarray(dates)), "y": np.asarray(y, dtype=float)})


def train_prophet(model, dates, y):
    """Fits `model` in place on (dates, y) and returns it, mirroring
    train_torch's calling convention so callers can treat all four
    architectures uniformly."""
    model.fit(prophet_frame(dates, y))
    return model


def prophet_predict(model, dates) -> np.ndarray:
    """Returns predicted target_return_pct for each date in `dates` (same
    order), via Prophet's point forecast (yhat). Works whether `dates`
    falls inside the training range or beyond it, so this covers both the
    backtest windows and the live next-day forecast."""
    future = pd.DataFrame({"ds": pd.to_datetime(np.asarray(dates))})
    forecast = model.predict(future)
    return forecast["yhat"].values


def save_prophet(model, path: str) -> None:
    """Prophet models aren't reliably picklable across versions;
    prophet.serialize's JSON round-trip is the supported mechanism, and
    lets Prophet share the same weight_path-on-disk registry pattern as
    the torch/XGBoost models."""
    from prophet.serialize import model_to_json
    with open(path, "w") as f:
        f.write(model_to_json(model))


def load_prophet(path: str):
    from prophet.serialize import model_from_json
    with open(path, "r") as f:
        return model_from_json(f.read())


def build_mnn(feature_count):
    set_seed(SEED)
    return nn.Sequential(
        nn.Linear(feature_count, CONFIG["MNN_Hidden_Units"]),
        nn.ReLU(),
        MultiplicativeLayer(CONFIG["MNN_Hidden_Units"], 1),
    )


def build_xgb():
    return xgb.XGBRegressor(
        objective=asymmetric_xgb_objective,
        n_estimators=CONFIG["XGBoost_Estimators"],
        max_depth=CONFIG["XGBoost_Max_Depth"],
        learning_rate=CONFIG["XGBoost_Learning_Rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        base_score=0.0,
    )


def train_torch(model, X, y, lr, epochs):
    """Full-batch training loop shared by LSTM / MNN (Prophet and XGBoost
    each have their own dedicated train/fit path), using the
    asymmetric directional loss instead of plain MSE."""
    criterion = AsymmetricDirectionalLoss(DIRECTION_PENALTY)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        opt.step()
    model.eval()
    return model


# ----------------------------------------------------------------------
# PURGED TIME-SERIES SPLIT
# ----------------------------------------------------------------------
class PurgedTimeSeriesSplit:
    """Rolling-window, walk-forward CV splitter for sequence data. Standard
    sklearn TimeSeriesSplit would let the train fold's last rows overlap
    the test fold in two ways: the lookback sequence windows span `Days`
    consecutive rows, and the forward-shifted target means row i's label
    depends on row i+1. This purges `purge_window` rows from the end of
    every train fold so neither leakage path can occur."""

    def __init__(self, n_splits: int = 5, purge_window: int = Days, min_train_size: int = None):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.min_train_size = min_train_size

    def split(self, X):
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        if fold_size < 1:
            raise ValueError("Not enough samples for the requested number of purged splits.")

        min_train = self.min_train_size or fold_size

        for fold in range(1, self.n_splits + 1):
            train_end = fold * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)
            if test_end <= test_start:
                continue

            purged_train_end = max(min_train, train_end - self.purge_window)
            train_idx = np.arange(0, purged_train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) < min_train or len(test_idx) == 0:
                continue

            yield train_idx, test_idx


def purged_train_val_test_split(n_samples: int, purge_window: int = Days,
                                 val_frac: float = 0.15, test_frac: float = 0.15):
    """A single, contiguous three-way split -- [train] -- purge gap --
    [validation] -- purge gap -- [test] -- carved by proportion rather
    than rolling folds. This gives a genuinely held-out validation block
    that isn't swallowed by the final training set the way successive
    PurgedTimeSeriesSplit folds would be. Both gaps are `purge_window` rows
    wide, same reasoning as PurgedTimeSeriesSplit, so a Days-length
    lookback sequence starting inside validation or test can never reach
    into the preceding block.

    Returns (train_idx, val_idx, test_idx), each a 1-D array of raw row
    indices into the caller's X/dates arrays.
    """
    test_size = max(1, int(n_samples * test_frac))
    val_size = max(1, int(n_samples * val_frac))

    test_start = n_samples - test_size
    val_end = max(0, test_start - purge_window)
    val_start = max(0, val_end - val_size)
    train_end = max(1, val_start - purge_window)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(val_start, val_end)
    test_idx = np.arange(test_start, n_samples)
    return train_idx, val_idx, test_idx


def raw_range_to_seq_slice(idx_array: np.ndarray, seq_length: int, n_seq: int):
    """Converts a contiguous raw-row index range (from
    purged_train_val_test_split) into the matching [start, end) slice
    bounds in sequence-space, where seq-row i corresponds to raw row
    (i + seq_length - 1). X_static (built via X_scaled[Days-1:]) uses the
    same alignment, so these bounds apply to both."""
    if len(idx_array) == 0:
        return 0, 0
    start = max(0, int(idx_array[0]) - seq_length + 1)
    end = min(n_seq, int(idx_array[-1]) - seq_length + 2)
    return start, max(start, end)


# ----------------------------------------------------------------------
# MODEL REGISTRY -- data-driven freshness cache. A saved model is "fresh"
# only if it was trained on data reaching at least `required_through_date`,
# not just "younger than 24h". REGISTRY_SAFETY_CAP_HOURS is a generous
# backstop (a full week) in case ingestion silently stalls.
# ----------------------------------------------------------------------
REGISTRY_SAFETY_CAP_HOURS = 24 * 7


def _meta_path(weight_path: str) -> str:
    return f"{weight_path}.meta.json"


def registry_is_fresh(weight_path: str, required_through_date: str,
                       max_age_hours: int = REGISTRY_SAFETY_CAP_HOURS) -> bool:
    """True if `weight_path` exists, was already trained on data reaching
    at least `required_through_date` ('YYYY-MM-DD'), and isn't older than
    the `max_age_hours` safety cap.

    `required_through_date` should be the latest date the caller's own
    pipeline() actually pulled -- comparing against the caller's own
    requirement (rather than one global "latest date") lets the
    live-forecast path and the backtest path share weight files sensibly,
    even though one trains on the full dataset and the other holds out a
    tail.
    """
    meta_path = _meta_path(weight_path)
    if not (os.path.exists(weight_path) and os.path.exists(meta_path)):
        return False
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        trained_through = meta.get("trained_through_date")
        saved_at = datetime.fromisoformat(meta["saved_at"])
        if trained_through is None:
            return False  # older meta written before this field existed -- force a retrain once
        if trained_through < required_through_date:
            return False  # newer data has arrived since this model was trained
        return (datetime.utcnow() - saved_at) < timedelta(hours=max_age_hours)
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def registry_stamp(weight_path: str, trained_through_date: str, extra: dict = None) -> None:
    """Writes/refreshes the sidecar timestamp+data-extent file next to a
    saved model. `trained_through_date` is what registry_is_fresh compares
    future requests against."""
    meta = {"saved_at": datetime.utcnow().isoformat(), "trained_through_date": trained_through_date}
    if extra:
        meta.update(extra)
    with open(_meta_path(weight_path), "w") as f:
        json.dump(meta, f, indent=2)


# ----------------------------------------------------------------------
# DATA PIPELINE
# ----------------------------------------------------------------------
FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "sma_10", "sma_50", "rsi_14",
    "bb_lower", "bb_mid", "bb_upper", "atr_14",
    "pct_return", "log_return",
    "rsi_14_z", "atr_14_z", "bb_width_z",
    "sentiment_score",
]


def pipeline(ticker_symbol: str):
    """Pulls the stationary feature matrix for `ticker_symbol` from the
    structured DB (stock_price + sentiment_metric), computes the unified
    target via target_utils, and returns (X, y, feature_count, dates).

    `dates` is a 1:1 aligned array of 'YYYY-MM-DD' strings for each row,
    used by evaluation.py to attach real calendar dates to per-row
    backtest predictions for the Chart.js price canvas.
    """
    print(f" [DATA ENGINE] Pulling matrix from database for: {ticker_symbol}.....")

    # Wrapped in text() for cross-dialect portability: a plain ":ticker"
    # placeholder string works on SQLite but psycopg2/Postgres raises a
    # syntax error on the literal colon.
    price_query = text(
        "SELECT * FROM stock_price WHERE ticker = :ticker AND source = 'historical' "
        "ORDER BY date ASC"
    )
    df = pd.read_sql(price_query, engine, params={"ticker": ticker_symbol})
    if df.empty:
        raise ValueError(f" No historical data found for {ticker_symbol} in database.")

    sentiment_query = text("SELECT date, sentiment_score FROM sentiment_metric WHERE ticker = :ticker")
    sentiment_df = pd.read_sql(sentiment_query, engine, params={"ticker": ticker_symbol})

    if not sentiment_df.empty:
        df = pd.merge(df, sentiment_df, on="date", how="left")
    else:
        df["sentiment_score"] = np.nan

    df["sentiment_score"] = df["sentiment_score"].ffill().fillna(0.5)
    df = df.sort_values("date").reset_index(drop=True)

    # Unified target (see target_utils.py) -- drops the final undecidable row.
    df = compute_next_day_target(df, price_col="close")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"pipeline({ticker_symbol}): missing expected feature columns {missing}")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    dates = df["date"].astype(str).values
    return X, y, len(FEATURE_COLUMNS), dates


def build_sequences(X, y, seq_length: int = Days):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return (
        torch.tensor(np.array(X_seq), dtype=torch.float32),
        torch.tensor(np.array(y_seq), dtype=torch.float32).view(-1, 1),
    )


def return_pct_to_price(last_close: float, predicted_return_pct: float) -> float:
    """Converts a model's stationary % return prediction back into an
    absolute price, purely for display (e.g. the UI's 'Predicted Value').
    Models are trained on and validated against the % return -- this
    conversion never feeds back into training."""
    return float(last_close) * (1.0 + float(predicted_return_pct) / 100.0)


def get_scaler(ticker, X_raw, fit_upto=None):
    """
    Loads/saves a per-ticker MinMaxScaler so reloaded model weights always
    see data scaled the same way it was trained on. `fit_upto` restricts
    fitting to the training rows only (purged-split leakage guard).
    """
    scaler_path = f"{WEIGHTS_DIR}/scaler_{ticker}.joblib"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        return scaler.transform(X_raw), scaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    fit_rows = X_raw if fit_upto is None else X_raw[:fit_upto]
    scaler.fit(fit_rows)
    joblib.dump(scaler, scaler_path)
    return scaler.transform(X_raw), scaler
