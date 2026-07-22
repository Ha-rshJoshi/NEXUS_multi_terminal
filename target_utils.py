"""Single source of truth for the "next day" prediction target -- every model
imports TARGET_COLUMN and compute_next_day_target from here instead of
hand-rolling its own shift logic."""

import numpy as np
import pandas as pd

# The single canonical name for the prediction target across the entire
# codebase. It represents the percentage return from today's close to the
# NEXT trading day's close: (close[t+1] - close[t]) / close[t] * 100
TARGET_COLUMN = "target_return_pct"

# Canonical name for the raw next-day close price (used only where an
# absolute price target is explicitly required, e.g. UI "Predicted Value"
# display). Kept separate from TARGET_COLUMN so nobody accidentally trains
# a model on non-stationary raw price again.
NEXT_CLOSE_COLUMN = "next_day_close"


def compute_next_day_target(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Adds TARGET_COLUMN (stationary % return, forward-shifted by exactly one
    row) and NEXT_CLOSE_COLUMN (raw next close, for display purposes only)
    to `df`. This is the ONLY place in NEXUS allowed to shift the target.

    df must already be sorted ascending by date before calling this.
    """
    if price_col not in df.columns:
        raise KeyError(
            f"compute_next_day_target: expected price column '{price_col}' "
            f"not found. Available columns: {list(df.columns)}"
        )

    working = df.copy()
    working[NEXT_CLOSE_COLUMN] = working[price_col].shift(-1)
    working[TARGET_COLUMN] = (
        (working[NEXT_CLOSE_COLUMN] - working[price_col]) / working[price_col]
    ) * 100.0

    # The final row has no "next day" yet -> drop it, it cannot be trained on.
    working = working.iloc[:-1].reset_index(drop=True)
    return working


def stationary_returns(series: pd.Series) -> pd.DataFrame:
    """
    Converts a raw price series into stationary features:
      - pct_return: simple percentage return
      - log_return: log return (numerically nicer for downstream scaling)
    Both are used as MODEL INPUTS (never as the target itself, see
    compute_next_day_target above for that).
    """
    pct_return = series.pct_change() * 100.0
    log_return = np.log(series / series.shift(1))
    return pd.DataFrame({"pct_return": pct_return, "log_return": log_return})
