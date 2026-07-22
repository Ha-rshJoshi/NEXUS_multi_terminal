"""Markowitz mean-variance portfolio optimization across NEXUS's tracked
tickers, used by /api/portfolio_optimize -- independent of the four
forecasting models."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from db_models import StockPrice

TRADING_DAYS_PER_YEAR = 252
HORIZON_KEYS = {"short", "medium", "long"}


def historical_returns_frame(tickers):
    """Aligned wide DataFrame of daily pct_return (one column per ticker).
    Inner-joined on date (via dropna) so the covariance matrix is only
    computed over dates where every requested ticker has a row -- otherwise
    a calendar gap in one stock would silently corrupt the estimate."""
    rows = (
        StockPrice.query
        .filter(StockPrice.ticker.in_(tickers), StockPrice.source == "historical")
        .with_entities(StockPrice.ticker, StockPrice.date, StockPrice.pct_return)
        .all()
    )
    if not rows:
        raise ValueError("No historical data available for the requested tickers.")

    df = pd.DataFrame(rows, columns=["ticker", "date", "pct_return"])
    wide = df.pivot(index="date", columns="ticker", values="pct_return").sort_index()
    wide = wide.dropna(how="any")

    missing = [t for t in tickers if t not in wide.columns]
    if missing or wide.empty:
        raise ValueError(f"Not enough overlapping historical data for: {missing or tickers}")
    return wide[tickers]


def expected_returns_and_covariance(returns_df):
    """Annualized mean-return vector + covariance matrix from a wide
    (date x ticker) daily pct_return DataFrame. pct_return is stored in
    percentage-point units, so this divides by 100 first to work in the
    decimal-return convention standard for portfolio math."""
    daily = returns_df.values / 100.0
    mu = daily.mean(axis=0) * TRADING_DAYS_PER_YEAR
    if daily.shape[1] == 1:
        cov = np.array([[float(np.var(daily[:, 0], ddof=1)) * TRADING_DAYS_PER_YEAR]])
    else:
        cov = np.cov(daily, rowvar=False) * TRADING_DAYS_PER_YEAR
    return mu, cov


def _portfolio_return(w, mu):
    return float(np.dot(w, mu))


def _portfolio_vol(w, cov):
    return float(np.sqrt(max(np.dot(w, np.dot(cov, w)), 0.0)))


def min_volatility_portfolio(mu, cov):
    """Long-only, fully-invested minimum-variance portfolio (the "short
    horizon" frontier point). Ignores mu by design -- pure risk minimization."""
    n = len(mu)
    if n == 1:
        return np.array([1.0])
    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(lambda w: np.dot(w, np.dot(cov, w)), x0, method="SLSQP",
                       bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"min_volatility_portfolio failed to converge: {result.message}")
    return result.x


def max_sharpe_portfolio(mu, cov, risk_free_rate=0.0):
    """Long-only, fully-invested tangency (maximum-Sharpe) portfolio -- the
    "long horizon" frontier point."""
    n = len(mu)
    if n == 1:
        return np.array([1.0])
    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_sharpe(w):
        vol = _portfolio_vol(w, cov)
        if vol == 0.0:
            return 0.0
        return -(_portfolio_return(w, mu) - risk_free_rate) / vol

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"max_sharpe_portfolio failed to converge: {result.message}")
    return result.x


def target_return_portfolio(mu, cov, target_return):
    """Long-only, fully-invested minimum-variance portfolio subject to an
    exact target return -- the "medium horizon" frontier point (target =
    midpoint between min-vol and max-Sharpe returns; see solve_frontier_point)."""
    n = len(mu)
    if n == 1:
        return np.array([1.0])
    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: np.dot(w, mu) - target_return},
    ]
    result = minimize(lambda w: np.dot(w, np.dot(cov, w)), x0, method="SLSQP",
                       bounds=bounds, constraints=constraints)
    if not result.success:
        # Exact target return can be infeasible for a narrow selection
        # (e.g. above the single best stock's return) -- fall back rather
        # than raising, since this is a legitimate edge case.
        return max_sharpe_portfolio(mu, cov)
    return result.x


def solve_frontier_point(mu, cov, horizon, risk_free_rate=0.0):
    """Maps a horizon bucket to a point on the Markowitz efficient frontier:
    short -> min-vol, long -> max-Sharpe, medium -> the frontier point at
    the return midway between those two (solved directly, not a weight
    blend). Returns (weights, objective_label)."""
    if horizon not in HORIZON_KEYS:
        raise ValueError(f"horizon must be one of {sorted(HORIZON_KEYS)}, got {horizon!r}")

    min_vol_w = min_volatility_portfolio(mu, cov)
    if horizon == "short":
        return min_vol_w, "min_volatility"

    max_sharpe_w = max_sharpe_portfolio(mu, cov, risk_free_rate)
    if horizon == "long":
        return max_sharpe_w, "max_sharpe"

    min_vol_ret = _portfolio_return(min_vol_w, mu)
    max_sharpe_ret = _portfolio_return(max_sharpe_w, mu)
    target = (min_vol_ret + max_sharpe_ret) / 2.0
    weights = target_return_portfolio(mu, cov, target)
    return weights, "target_return"


def correlation_matrix(cov):
    std = np.sqrt(np.diag(cov))
    outer = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / outer
    return np.nan_to_num(corr, nan=1.0)


def latest_close_prices(tickers):
    """Latest source='historical' close price per ticker -- same convention
    forecast_ticker()/performance_backtest() use for 'last_close', so the
    share math is priced consistently with the rest of the dashboard."""
    prices = {}
    for t in tickers:
        row = (
            StockPrice.query
            .filter_by(ticker=t, source="historical")
            .order_by(StockPrice.date.desc())
            .first()
        )
        if not row or row.close is None:
            raise ValueError(f"No historical close price available for {t}.")
        prices[t] = float(row.close)
    return prices


def allocate_shares(tickers, weights, latest_prices, amount_min, amount_max):
    """Converts ideal weight fractions into a whole-share allocation that
    fits inside [amount_min, amount_max] and stays close to the ideal
    ratios. NSE has no fractional shares, so with only a handful of
    tickers this greedily tops up whichever stock is furthest under its
    ideal weight, one share at a time, never exceeding amount_max.

    Returns (breakdown_list, total_allocated, leftover_cash_vs_amount_max).
    """
    n = len(tickers)
    prices = np.array([latest_prices[t] for t in tickers], dtype=float)

    # Floor against the top of the band so the allocation uses as much of
    # the budget as the whole-share constraint allows.
    ideal_amounts = weights * amount_max
    shares = np.maximum(np.floor(ideal_amounts / prices).astype(int), 0)

    def total_cost(s):
        return float(np.dot(s, prices))

    while True:
        current_cost = total_cost(shares)
        current_weights = (shares * prices) / current_cost if current_cost > 0 else np.zeros(n)
        deficits = weights - current_weights
        order = np.argsort(-deficits)  # most under-weighted first
        added = False
        for i in order:
            candidate = shares.copy()
            candidate[i] += 1
            if total_cost(candidate) <= amount_max:
                shares = candidate
                added = True
                break
        if not added:
            break

    final_cost = total_cost(shares)
    breakdown = [
        {
            "ticker": t,
            "weight_pct": float(weights[i] * 100.0),
            "price": float(prices[i]),
            "shares": int(shares[i]),
            "allocated_amount": float(shares[i] * prices[i]),
        }
        for i, t in enumerate(tickers)
    ]
    leftover_cash = float(amount_max - final_cost)
    under_minimum = final_cost < amount_min
    return breakdown, final_cost, leftover_cash, under_minimum


def optimize_portfolio(tickers, amount_min, amount_max, horizon, risk_free_rate=0.0):
    """Top-level entrypoint used by /api/portfolio_optimize. `tickers` may
    be any subset of NEXUS's tracked tickers; a single ticker is a
    degenerate case (100% weight) handled cleanly rather than raising."""
    tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order
    if not tickers:
        raise ValueError("At least one ticker must be selected.")

    returns_df = historical_returns_frame(tickers)
    mu, cov = expected_returns_and_covariance(returns_df)

    if len(tickers) == 1:
        weights = np.array([1.0])
        objective_used = "single_ticker_no_diversification"
    else:
        weights, objective_used = solve_frontier_point(mu, cov, horizon, risk_free_rate)

    prices = latest_close_prices(tickers)
    breakdown, total_allocated, leftover_cash, under_minimum = allocate_shares(
        tickers, weights, prices, amount_min, amount_max
    )

    port_return = _portfolio_return(weights, mu)
    port_vol = _portfolio_vol(weights, cov)
    port_sharpe = ((port_return - risk_free_rate) / port_vol) if port_vol > 0 else None

    corr = correlation_matrix(cov)
    correlation = {
        tickers[i]: {tickers[j]: float(corr[i, j]) for j in range(len(tickers))}
        for i in range(len(tickers))
    }

    return {
        "tickers": tickers,
        "horizon": horizon,
        "objective_used": objective_used,
        "breakdown": breakdown,
        "total_allocated": total_allocated,
        "leftover_cash": leftover_cash,
        "under_minimum": under_minimum,
        "amount_min": amount_min,
        "amount_max": amount_max,
        "expected_annual_return_pct": port_return * 100.0,
        "annual_volatility_pct": port_vol * 100.0,
        "sharpe_ratio": port_sharpe,
        "correlation": correlation,
    }
