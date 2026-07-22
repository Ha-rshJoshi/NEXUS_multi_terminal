"""Backtest/research entrypoint -- runs the same models as models.py through
a purged train/validation/test split and reports per-model metrics (MAE,
RMSE, win rate, Sharpe). This is what app.py's /api/predict calls."""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from google import genai
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text

import quant_core as core
from quant_core import (
    CONFIG, Days, NAIVE_KEY, SEED, WEIGHTS_DIR,
    build_lstm, build_mnn, build_prophet, build_xgb,
    build_sequences, engine, get_scaler, load_prophet, pipeline,
    prophet_predict, registry_is_fresh, registry_stamp, return_pct_to_price,
    save_prophet, set_seed, train_prophet, train_torch,
)


def log_experiment(ticker, best_model, all_metrics, validation_metrics=None, run_tag=None):
    """Appends one experiment record to model_registry.json. `validation_metrics`
    is the held-out validation-block RMSE/MAE/Win Rate per model -- logged
    here for tuning/diagnostics, separate from `all_metrics` (the
    test-block numbers that drive the winner pick and the dashboard).
    `run_tag` marks sweep-only experiments so they're easy to filter out
    of the log when reviewing production history."""
    registry_file = "model_registry.json"
    registry = []
    if os.path.exists(registry_file):
        with open(registry_file, "r") as f:
            registry = json.load(f)
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": ticker,
        "Winning_Model": best_model,
        "Hyperparameters": CONFIG,
        "Performance_Metrics": all_metrics,
    }
    if validation_metrics is not None:
        entry["Validation_Metrics"] = validation_metrics
    if run_tag is not None:
        entry["Run_Tag"] = run_tag
    registry.append(entry)
    with open(registry_file, "w") as f:
        json.dump(registry, f, indent=2)
    print(f" Experiment logged to {registry_file}.")


_model_prediction_table_ready = False


def _persist_model_prediction(ticker, model_name, past_day_value, predicted_price, rmse_price, mae_price,
                               win_rate_price, sharpe_ratio=None):
    """Persists one model's efficiency snapshot to the model_prediction
    table. Values are stored in currency units: predicted_value <-
    predicted_price, error_margin <- rmse (currency), mae <- mae
    (currency), win_rate <- win_rate (price-direction based) -- this keeps
    everything on the same price scale as the Chart.js "Model Comparison"
    bar chart, which plots error_margin alongside past_day_value/
    predicted_value. sharpe_ratio is the annualized Sharpe of the simple
    long/short strategy derived from this model's predicted direction (see
    strategy_sharpe_ratio) -- can be None when that strategy's return
    series has zero variance.

    Table creation is delegated to db_models' SQLAlchemy ORM metadata,
    which emits correct per-dialect DDL (Postgres SERIAL/IDENTITY vs
    SQLite AUTOINCREMENT) instead of hand-written, single-dialect SQL.
    """
    global _model_prediction_table_ready
    if not _model_prediction_table_ready:
        from db_models import create_all_tables
        create_all_tables(engine)
        _model_prediction_table_ready = True

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO model_prediction
            (ticker, model_name, past_day_value, predicted_value, error_margin, mae, win_rate, sharpe_ratio, created_at)
            VALUES (:ticker, :model_name, :past_day_value, :predicted_value, :error_margin, :mae, :win_rate, :sharpe_ratio, :created_at)
        """), {
            "ticker": ticker, "model_name": model_name,
            "past_day_value": past_day_value, "predicted_value": predicted_price,
            "error_margin": rmse_price, "mae": mae_price, "win_rate": win_rate_price,
            "sharpe_ratio": sharpe_ratio,
            "created_at": datetime.utcnow(),
        })


def directional_win_rate(preds, actuals):
    preds = np.asarray(preds).flatten()
    actuals = np.asarray(actuals).flatten()
    return float(np.mean(np.sign(preds) == np.sign(actuals)) * 100)


TRADING_DAYS_PER_YEAR = 252  # standard equity-market annualization convention


def strategy_sharpe_ratio(predicted_returns_pct, actual_returns_pct, risk_free_rate: float = 0.0):
    """Risk-adjusted metric, additive to RMSE/MAE/win_rate -- never used
    for the winner pick.

    Simulates the simplest directional strategy from a model's predicted
    return sign (long on a predicted up day, short on a predicted down
    day) applied to the actual realized return, then returns the
    annualized Sharpe ratio of that strategy's daily returns against
    `risk_free_rate` (an annualized rate). Catches something RMSE/MAE/
    win_rate can't: a model can have a decent win rate but a poor Sharpe
    if it's right on quiet days and badly wrong on volatile ones.

    Returns None, not 0.0, if the strategy-return series has zero
    variance (e.g. a model that never changes predicted sign) -- the
    ratio is genuinely undefined there, not "no edge".
    """
    predicted_returns_pct = np.asarray(predicted_returns_pct, dtype=float).flatten()
    actual_returns_pct = np.asarray(actual_returns_pct, dtype=float).flatten()
    if len(predicted_returns_pct) < 2:
        return None

    position = np.sign(predicted_returns_pct)
    strategy_returns_pct = position * actual_returns_pct

    # Sharpe is scale-invariant, so working in percentage-point units
    # throughout (rather than converting to decimal returns) is fine.
    daily_rf_pct = (risk_free_rate / TRADING_DAYS_PER_YEAR) * 100.0
    excess_returns = strategy_returns_pct - daily_rf_pct

    std = float(np.std(excess_returns, ddof=1))
    if std == 0.0 or np.isnan(std):
        return None
    mean = float(np.mean(excess_returns))
    return float((mean / std) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _load_or_train_for_backtest(model_key, builder, weight_path, X_tr, y_tr, lr, epochs, feature_count,
                                 required_through_date, is_xgb=False):
    """Registry check: load if already trained through
    `required_through_date`, otherwise train fresh and stamp the registry
    (mirrors models.py's load_or_train_torch/load_or_train_xgb)."""
    if is_xgb:
        model = builder()
        if registry_is_fresh(weight_path, required_through_date):
            model.load_model(weight_path)
            print(f"   [{model_key}] loaded from registry (through {required_through_date}).")
        else:
            model.fit(X_tr, y_tr)
            model.save_model(weight_path)
            registry_stamp(weight_path, required_through_date)
            print(f"   [{model_key}] trained fresh and saved to registry.")
        return model

    model = builder(feature_count)
    if registry_is_fresh(weight_path, required_through_date):
        model.load_state_dict(torch.load(weight_path))
        model.eval()
        print(f"   [{model_key}] loaded from registry (through {required_through_date}).")
    else:
        model = train_torch(model, X_tr, y_tr, lr, epochs)
        torch.save(model.state_dict(), weight_path)
        registry_stamp(weight_path, required_through_date)
        print(f"   [{model_key}] trained fresh and saved to registry.")
    return model


def _load_or_train_prophet_for_backtest(weight_path, dates_tr, y_tr, required_through_date):
    """Same registry check as _load_or_train_for_backtest, but Prophet
    fits on raw (date, target_return_pct) rows rather than scaled
    tensors, and saves/loads via quant_core's JSON serialize helpers."""
    if registry_is_fresh(weight_path, required_through_date):
        model = load_prophet(weight_path)
        print(f"   [Prophet] loaded from registry (through {required_through_date}).")
    else:
        model = train_prophet(build_prophet(), dates_tr, y_tr)
        save_prophet(model, weight_path)
        registry_stamp(weight_path, required_through_date)
        print(f"   [Prophet] trained fresh and saved to registry.")
    return model


def performance_backtest(stocks, run_tag=None, persist=True):
    """Runs the purged three-way split (see
    quant_core.purged_train_val_test_split); the validation block is
    logged for diagnostics only, the test block drives the returned
    report/dashboard.

    `run_tag` isolates a sweep run's weight files under a subfolder of
    WEIGHTS_DIR so they never collide with the production registry cache.
    `persist=False` skips writing to model_prediction so sweep runs don't
    pollute the dashboard's prediction history.
    """
    print("----- LOSS MINIMIZATION & DIRECTIONAL ACCURACY STUDY (PURGED 3-WAY SPLIT) -----")
    report = {}
    weights_dir = f"{WEIGHTS_DIR}/sweep_{run_tag}" if run_tag else WEIGHTS_DIR
    if run_tag:
        os.makedirs(weights_dir, exist_ok=True)

    for ticker in stocks:
        print(f"Processing {ticker}.....")
        try:
            X_raw, y_raw, feature_count, dates_raw = pipeline(ticker)
            n_samples = len(X_raw)

            train_idx, val_idx, test_idx = core.purged_train_val_test_split(n_samples, purge_window=Days)
            if len(train_idx) < Days + 1 or len(val_idx) == 0 or len(test_idx) == 0:
                raise ValueError(f"Not enough rows for a three-way purged split on {ticker}.")
            split_raw = int(train_idx[-1]) + 1
            required_through_date = str(dates_raw[train_idx[-1]])

            X_scaled, _ = get_scaler(ticker, X_raw, fit_upto=split_raw)  # fit on TRAIN block only -- no leakage
            X_seq, y_seq = build_sequences(X_scaled, y_raw, Days)
            n_seq = len(X_seq)

            train_start, train_end = core.raw_range_to_seq_slice(train_idx, Days, n_seq)
            val_start, val_end = core.raw_range_to_seq_slice(val_idx, Days, n_seq)
            test_start, test_end = core.raw_range_to_seq_slice(test_idx, Days, n_seq)
            if train_end <= train_start or val_end <= val_start or test_end <= test_start:
                raise ValueError(f"Purged split collapsed to an empty block in sequence-space for {ticker}.")

            X_tr_seq, y_tr_seq = X_seq[train_start:train_end], y_seq[train_start:train_end]
            X_val_seq, y_val_seq = X_seq[val_start:val_end], y_seq[val_start:val_end]
            X_te_seq = X_seq[test_start:test_end]

            X_static = X_scaled[Days - 1:]
            y_static = y_raw[Days - 1:]
            X_tr_st, y_tr_st = X_static[train_start:train_end], y_static[train_start:train_end]
            X_val_st = X_static[val_start:val_end]
            X_te_st = X_static[test_start:test_end]

            X_tr_st_t = torch.tensor(X_tr_st, dtype=torch.float32)
            X_val_st_t = torch.tensor(X_val_st, dtype=torch.float32)
            X_te_st_t = torch.tensor(X_te_st, dtype=torch.float32)
            y_tr_st_t = torch.tensor(y_tr_st, dtype=torch.float32).view(-1, 1)

            # Raw-row date/price alignment, computed before model training
            # since Prophet needs actual calendar dates rather than X_seq/X_static.
            close_prices_full = X_raw[:, 3]  # 'close' is index 3 in FEATURE_COLUMNS

            val_start_raw_idx = val_start + Days - 1
            n_val = val_end - val_start
            dates_val = dates_raw[val_start_raw_idx: val_start_raw_idx + n_val]

            test_start_raw_idx = test_start + Days - 1
            n_test = test_end - test_start
            dates_test = dates_raw[test_start_raw_idx: test_start_raw_idx + n_test]
            current_prices_test = close_prices_full[test_start_raw_idx: test_start_raw_idx + n_test]

            lstm_path = f"{weights_dir}/lstm_{ticker}.pth"
            prophet_path = f"{weights_dir}/prophet_{ticker}.json"
            mnn_path = f"{weights_dir}/mnn_{ticker}.pth"
            xgb_path = f"{weights_dir}/xgb_{ticker}.json"

            lstm = _load_or_train_for_backtest("LSTM", build_lstm, lstm_path, X_tr_seq, y_tr_seq,
                                                CONFIG["Learning_Rate"], CONFIG["LSTM_Epochs"], feature_count,
                                                required_through_date)
            with torch.no_grad():
                lstm_val_preds = lstm(X_val_seq).cpu().numpy()
                lstm_preds = lstm(X_te_seq).cpu().numpy()

            # Prophet trains on every raw training row -- no lookback windowing needed.
            dates_tr_raw = dates_raw[train_idx]
            y_tr_raw = y_raw[train_idx]
            prophet_model = _load_or_train_prophet_for_backtest(
                prophet_path, dates_tr_raw, y_tr_raw, required_through_date
            )
            prophet_val_preds = prophet_predict(prophet_model, dates_val)
            prophet_preds = prophet_predict(prophet_model, dates_test)

            mnn = _load_or_train_for_backtest("MNN", build_mnn, mnn_path, X_tr_st_t, y_tr_st_t,
                                               CONFIG["MNN_Learning_Rate"], CONFIG["MNN_Epochs"], feature_count,
                                               required_through_date)
            with torch.no_grad():
                mnn_val_preds = mnn(X_val_st_t).numpy().flatten()
                mnn_preds = mnn(X_te_st_t).numpy().flatten()

            xgb_model = _load_or_train_for_backtest("XGBoost", build_xgb, xgb_path, X_tr_st, y_tr_st,
                                                      None, None, feature_count, required_through_date, is_xgb=True)
            xgb_val_preds = xgb_model.predict(X_val_st).flatten()
            xgb_preds = xgb_model.predict(X_te_st).flatten()

            actuals = y_seq[test_start:test_end].cpu().numpy().flatten()
            val_actuals = y_val_seq.cpu().numpy().flatten()
            naive_preds = np.zeros_like(actuals)
            naive_val_preds = np.zeros_like(val_actuals)

            preds_map = {
                "LSTM": lstm_preds.flatten(),
                "Prophet": np.asarray(prophet_preds).flatten(),
                "MNN": mnn_preds,
                "XGBoost": xgb_preds,
            }
            val_preds_map = {
                "LSTM": lstm_val_preds.flatten(),
                "Prophet": np.asarray(prophet_val_preds).flatten(),
                "MNN": mnn_val_preds,
                "XGBoost": xgb_val_preds,
            }

            actual_prices_test = current_prices_test * (1.0 + actuals / 100.0)

            # Validation-block metrics: diagnostics/tuning only, logged to
            # model_registry.json, never shown on the dashboard or used to pick the winner.
            validation_metrics = {}
            naive_val_rmse = float(np.sqrt(mean_squared_error(val_actuals, naive_val_preds)))
            naive_val_mae = float(mean_absolute_error(val_actuals, naive_val_preds))
            validation_metrics[NAIVE_KEY] = {"RMSE": naive_val_rmse, "MAE": naive_val_mae, "Win Rate": None}
            for name, vpreds in val_preds_map.items():
                v_rmse = float(np.sqrt(mean_squared_error(val_actuals, vpreds)))
                v_mae = float(mean_absolute_error(val_actuals, vpreds))
                v_win = directional_win_rate(vpreds, val_actuals)
                validation_metrics[name] = {"RMSE": v_rmse, "MAE": v_mae, "Win Rate": v_win}

            results = {}
            print(f" {'Model':<12} | {'RMSE':<10} | {'MAE':<10} | {'Win %':<10} | Status")
            print("-" * 60)
            naive_rmse = float(np.sqrt(mean_squared_error(actuals, naive_preds)))
            naive_mae = float(mean_absolute_error(actuals, naive_preds))
            results[NAIVE_KEY] = {"RMSE": naive_rmse, "MAE": naive_mae, "Win Rate": None}
            print(f" {NAIVE_KEY:<12} | {naive_rmse:<10.3f} | {naive_mae:<10.3f} | {'--':<10} | BENCHMARK")

            # `results` stays in return-% space and picks the winner against
            # the naive benchmark; model_breakdown (below, currency units) is
            # what the dashboard actually consumes.
            best_rmse, best_name = float("inf"), ""
            for name, preds in preds_map.items():
                rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
                mae = float(mean_absolute_error(actuals, preds))
                win = directional_win_rate(preds, actuals)
                status = "BEAT" if rmse < naive_rmse else "FAILED"
                print(f" {name:<12} | {rmse:<10.3f} | {mae:<10.3f} | {win:<10.2f} | {status}")
                results[name] = {"RMSE": rmse, "MAE": mae, "Win Rate": win}
                if rmse < best_rmse:
                    best_rmse, best_name = rmse, name
            print(f" Winner [{ticker}]: {best_name} (RMSE {best_rmse:.3f}, return-%% space)")

            # Per-model efficiency breakdown (currency units) -- computed
            # independently per model, never averaged together.
            last_close = float(X_raw[-1][3])
            with torch.no_grad():
                latest_seq = X_seq[-1].unsqueeze(0)
                latest_static = X_static[-1].reshape(1, -1)
                latest_static_t = torch.tensor(latest_static, dtype=torch.float32)

                next_lstm_ret = lstm(latest_seq).item()
                next_mnn_ret = mnn(latest_static_t).item()
            next_xgb_ret = float(xgb_model.predict(latest_static)[0])
            # Next business day after the dataset's latest date -- an
            # approximation, since it doesn't know India's market holidays.
            next_date = pd.bdate_range(start=pd.to_datetime(dates_raw[-1]), periods=2)[-1]
            next_prophet_ret = float(prophet_predict(prophet_model, [next_date])[0])

            next_step_returns = {
                "LSTM": next_lstm_ret,
                "Prophet": next_prophet_ret,
                "MNN": next_mnn_ret,
                "XGBoost": next_xgb_ret,
            }

            model_breakdown = {}
            for name in ("LSTM", "Prophet", "MNN", "XGBoost"):
                return_preds_test = preds_map[name]

                predicted_price = return_pct_to_price(last_close, next_step_returns[name])

                # Each row priced off its own current price, not the single
                # latest close, so mae/rmse reflect the whole backtest window.
                predicted_prices_test = current_prices_test * (1.0 + return_preds_test / 100.0)

                mae_price = float(mean_absolute_error(actual_prices_test, predicted_prices_test))
                rmse_price = float(np.sqrt(mean_squared_error(actual_prices_test, predicted_prices_test)))

                predicted_direction = np.sign(predicted_prices_test - current_prices_test)
                actual_direction = np.sign(actual_prices_test - current_prices_test)
                win_rate_price = float(np.mean(predicted_direction == actual_direction) * 100.0)

                sharpe = strategy_sharpe_ratio(return_preds_test, actuals)

                predicted_series = [
                    {"date": str(d), "predicted_price": float(p), "actual_price": float(a)}
                    for d, p, a in zip(dates_test, predicted_prices_test, actual_prices_test)
                ]

                model_breakdown[name] = {
                    "past_day_value": last_close,
                    "predicted_price": predicted_price,
                    "mae": mae_price,
                    "rmse": rmse_price,
                    "win_rate": win_rate_price,
                    "sharpe_ratio": sharpe,
                    "predicted_series": predicted_series,
                }

                if persist:
                    _persist_model_prediction(
                        ticker, name, last_close, predicted_price, rmse_price, mae_price, win_rate_price,
                        sharpe_ratio=sharpe,
                    )

            report[ticker] = {
                "results": results,
                "validation_metrics": validation_metrics,
                "winner": best_name,
                "model_breakdown": model_breakdown,
            }

            log_experiment(ticker, best_name, results, validation_metrics=validation_metrics, run_tag=run_tag)
        except Exception as e:
            print(f" Failed on {ticker}: {e}")
            continue

    return report


def generate_ai_analyst_conclusion(report):
    print(" GENERATIVE AI: DRAFTING CONCLUSION...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(" Warning: GEMINI_API_KEY not found.")
        return
    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are a Lead Quantitative Researcher at a top-tier hedge fund.
    Out-of-sample, purged-CV backtest results across LSTM, Prophet, MNN and
    XGBoost (with a Naive Baseline benchmark) are below (RMSE/MAE/Win Rate):
    {json.dumps(report, indent=2, default=str)}

    Write a concise, clinical executive summary:
    - Best architecture overall and why its structure helped.
    - If winners differ per stock, why different regimes favour different models.
    - One-sentence deployment recommendation.
    """
    try:
        resp = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
        print("\n" + resp.text)
    except Exception as e:
        print(f" Failed to generate AI conclusion: {e}")


if __name__ == "__main__":
    set_seed(SEED)
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    hidden_unit_options = [64, 128, 256]
    epoch_options = [100, 200]
    total = len(hidden_unit_options) * len(epoch_options)
    run = 1
    print("----- AUTOMATED HYPERPARAMETER SWEEP -----")
    for units in hidden_unit_options:
        for epochs in epoch_options:
            print("-*-" * 12)
            print(f"EXPERIMENT [{run}/{total}]: {units} units | {epochs} epochs")
            print("-*-" * 12)
            CONFIG["LSTM_Hidden_Units"] = units
            CONFIG["LSTM_Epochs"] = epochs
            # Prophet has no hidden-units/epochs knobs, so it (like XGBoost)
            # sits outside this sweep's tuning axis.
            CONFIG["MNN_Hidden_Units"] = int(units / 2)
            CONFIG["MNN_Epochs"] = epochs
            run_tag = f"u{units}_e{epochs}"
            data = performance_backtest(tickers, run_tag=run_tag, persist=False)
            if data:
                generate_ai_analyst_conclusion(data)
            run += 1
    print(" SWEEP COMPLETE. Logged into model_registry.json")
