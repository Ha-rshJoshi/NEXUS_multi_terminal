"""Live-forecast entrypoint: loads or trains each of the four models per
ticker (via quant_core) and returns a per-model breakdown, never averaged."""

import os

import pandas as pd
import torch

import quant_core as core
from quant_core import (
    CONFIG, Days, SEED, WEIGHTS_DIR, build_lstm, build_mnn, build_prophet,
    build_sequences, build_xgb, get_scaler, load_prophet, pipeline,
    prophet_predict, registry_is_fresh, registry_stamp, return_pct_to_price,
    save_prophet, set_seed, train_prophet, train_torch,
)


def load_or_train_torch(builder, path, X, y, lr, epochs, feature_count, required_through_date):
    """Saved weight AND already trained through at least
    `required_through_date` -> load instantly. Otherwise train from scratch
    and stamp the registry with the data extent actually used."""
    model = builder(feature_count)
    if registry_is_fresh(path, required_through_date):
        print(f"   loading fresh registry weights (through {required_through_date}): {path}")
        model.load_state_dict(torch.load(path))
        model.eval()
    else:
        print(f"   registry stale or missing -> training from scratch -> {path}")
        train_torch(model, X, y, lr, epochs)
        torch.save(model.state_dict(), path)
        registry_stamp(path, required_through_date)
    return model


def load_or_train_xgb(path, X, y, required_through_date):
    xgb_model = build_xgb()
    if registry_is_fresh(path, required_through_date):
        print(f"   loading fresh registry weights (through {required_through_date}): {path}")
        xgb_model.load_model(path)
    else:
        print(f"   registry stale or missing -> training from scratch -> {path}")
        xgb_model.fit(X, y)
        xgb_model.save_model(path)
        registry_stamp(path, required_through_date)
    return xgb_model


def load_or_train_prophet(path, dates, y, required_through_date):
    """Same data-driven registry pattern as load_or_train_torch/xgb, but
    using Prophet's own fit/serialize calls since it trains on the raw
    (dates, target_return_pct) series directly, not scaled feature tensors."""
    if registry_is_fresh(path, required_through_date):
        print(f"   loading fresh registry weights (through {required_through_date}): {path}")
        model = load_prophet(path)
    else:
        print(f"   registry stale or missing -> training from scratch -> {path}")
        model = train_prophet(build_prophet(), dates, y)
        save_prophet(model, path)
        registry_stamp(path, required_through_date)
    return model


def forecast_ticker(ticker):
    X_raw, y_raw, feature_count, dates = pipeline(ticker)
    # "Fresh" means trained through the newest date pipeline() returned --
    # a new trading day's data triggers an automatic retrain below.
    latest_date = str(dates[-1])

    X_scaled, _ = get_scaler(ticker, X_raw)
    X_seq, y_seq = build_sequences(X_scaled, y_raw, Days)
    X_static = torch.tensor(X_scaled[Days - 1:], dtype=torch.float32)
    y_static = torch.tensor(y_raw[Days - 1:], dtype=torch.float32).view(-1, 1)

    print(" [1/4] LSTM")
    lstm = load_or_train_torch(build_lstm, f"{WEIGHTS_DIR}/lstm_{ticker}.pth",
                                X_seq, y_seq, CONFIG["Learning_Rate"],
                                CONFIG["LSTM_Epochs"], feature_count, latest_date)

    print(" [2/4] Prophet")
    prophet_model = load_or_train_prophet(f"{WEIGHTS_DIR}/prophet_{ticker}.json", dates, y_raw, latest_date)
    # Next business day after latest_date; approximate since it doesn't
    # know India's market holiday calendar (same precision used elsewhere).
    next_date = pd.bdate_range(start=pd.to_datetime(latest_date), periods=2)[-1]

    print(" [3/4] MNN")
    mnn = load_or_train_torch(build_mnn, f"{WEIGHTS_DIR}/mnn_{ticker}.pth",
                               X_static, y_static, CONFIG["MNN_Learning_Rate"],
                               CONFIG["MNN_Epochs"], feature_count, latest_date)

    print(" [4/4] XGBoost")
    xgb_model = load_or_train_xgb(f"{WEIGHTS_DIR}/xgb_{ticker}.json", X_scaled[Days - 1:], y_raw[Days - 1:], latest_date)

    last_seq = X_seq[-1].unsqueeze(0)
    last_static = X_scaled[-1].reshape(1, -1)
    last_static_t = torch.tensor(last_static, dtype=torch.float32)

    with torch.no_grad():
        lstm_ret = lstm(last_seq).item()
        mnn_ret = mnn(last_static_t).item()
    prophet_ret = float(prophet_predict(prophet_model, [next_date])[0])
    xgb_ret = float(xgb_model.predict(last_static)[0])

    last_close = float(X_raw[-1][3])  # 'close' column, see quant_core.FEATURE_COLUMNS

    # Explicit, separate per-model breakdown -- never averaged.
    return {
        "Last Close": last_close,
        "LSTM": {"past_day_value": last_close, "predicted_value": return_pct_to_price(last_close, lstm_ret),
                  "predicted_return_pct": lstm_ret},
        "Prophet": {"past_day_value": last_close, "predicted_value": return_pct_to_price(last_close, prophet_ret),
                     "predicted_return_pct": prophet_ret},
        "MNN": {"past_day_value": last_close, "predicted_value": return_pct_to_price(last_close, mnn_ret),
                 "predicted_return_pct": mnn_ret},
        "XGBoost": {"past_day_value": last_close, "predicted_value": return_pct_to_price(last_close, xgb_ret),
                     "predicted_return_pct": xgb_ret},
    }


if __name__ == "__main__":
    set_seed(SEED)
    stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    print("----- MULTI-MODEL FORECAST (PER-MODEL BREAKDOWN) -----")
    forecasts = {}
    for t in stocks:
        print(f"\n ----- {t} -----")
        try:
            forecasts[t] = forecast_ticker(t)
        except Exception as e:
            print(f"--- ERROR processing {t}: {e}")

    print("\n----- FINAL FORECASTS REPORT -----\n")
    for t, d in forecasts.items():
        print(f"{t} | Last Close: {d['Last Close']:.3f}")
        for model_name in ("LSTM", "Prophet", "MNN", "XGBoost"):
            m = d[model_name]
            print(f"   {model_name:<8} Predicted: {m['predicted_value']:.3f} "
                  f"(Return {m['predicted_return_pct']:.3f}%)")
