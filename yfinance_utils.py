"""Fallback news + quote source for tickers Finnhub's free tier won't serve
(NSE/BSE). Used by data_engine.py and news_scraper.py."""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _extract_item(raw_item: dict):
    """yfinance's news payload shape has shifted across library versions:
    older releases return {'title': ..., 'providerPublishTime': <unix>},
    newer releases nest it under {'content': {'title': ..., 'pubDate':
    <ISO8601>}}. Handles both so a yfinance upgrade doesn't silently break
    ingestion."""
    title = raw_item.get("title")
    unix_ts = raw_item.get("providerPublishTime")

    if title is None or unix_ts is None:
        content = raw_item.get("content") or {}
        title = title or content.get("title")
        pub_date = content.get("pubDate") or content.get("displayTime")
        if pub_date and unix_ts is None:
            try:
                unix_ts = int(datetime.fromisoformat(pub_date.replace("Z", "+00:00")).timestamp())
            except (ValueError, TypeError):
                unix_ts = None

    return title, unix_ts


def fetch_recent_news(ticker: str, lookback_days: int = None):
    """Returns [{'headline': str, 'datetime': unix_seconds}, ...] for
    whatever recent news Yahoo Finance is currently surfacing for `ticker`.
    `lookback_days`, if given, filters out anything older than that
    (best-effort -- see module docstring on why this is not a true
    historical backfill)."""
    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception as e:
        logging.error(f"[yfinance_utils] Failed to fetch news for {ticker}: {e}")
        return []

    cutoff_ts = None
    if lookback_days is not None:
        cutoff_ts = (datetime.utcnow() - timedelta(days=lookback_days)).timestamp()

    results = []
    for raw in raw_items:
        title, unix_ts = _extract_item(raw)
        if not title or not unix_ts:
            continue
        if cutoff_ts is not None and unix_ts < cutoff_ts:
            continue
        results.append({"headline": title, "datetime": int(unix_ts)})

    return results


def fetch_live_quote(ticker: str):
    """Returns {'price': float, 'volume': int, 'open': float, 'high':
    float, 'low': float} or None if unavailable. Mirrors the shape
    data_engine.py needs from Finnhub's quote() so the two sources are
    interchangeable at the call site.

    Two-stage fallback: `fast_info` first (cheap, one request), then
    `.history(period='1d', interval='1m')` if that comes back empty.
    fast_info hits a more bot-restricted Yahoo endpoint and can silently
    return no data (no exception) when Yahoo is rate-limiting that
    endpoint, even while history() keeps working for the same ticker."""
    try:
        fast_info = yf.Ticker(ticker).fast_info
        price = fast_info.get("last_price") if hasattr(fast_info, "get") else fast_info.last_price
        volume = fast_info.get("last_volume") if hasattr(fast_info, "get") else fast_info.last_volume
        if price is not None:
            return {
                "price": float(price),
                "volume": int(volume) if volume else 0,
                "open": getattr(fast_info, "open", None),
                "high": getattr(fast_info, "day_high", None),
                "low": getattr(fast_info, "day_low", None),
            }
        logging.warning(
            f"[yfinance_utils] fast_info returned no price for {ticker} (no exception -- "
            f"Yahoo likely served an empty/rate-limited response on that endpoint). "
            f"Falling back to intraday history()."
        )
    except Exception as e:
        logging.warning(
            f"[yfinance_utils] fast_info raised for {ticker}: {e} -- falling back to intraday history()."
        )

    # Falls back further to the last daily candle if intraday minute bars
    # aren't available (e.g. a market holiday).
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        if hist.empty:
            hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        if hist.empty:
            logging.info(f"[yfinance_utils] No intraday or recent daily data available for {ticker} right now.")
            return None

        last_row = hist.iloc[-1]

        def _clean(value):
            return None if pd.isna(value) else float(value)

        price = _clean(last_row.get("Close"))
        if price is None:
            return None
        volume = last_row.get("Volume")
        return {
            "price": price,
            "volume": int(volume) if volume is not None and not pd.isna(volume) else 0,
            "open": _clean(last_row.get("Open")),
            "high": _clean(last_row.get("High")),
            "low": _clean(last_row.get("Low")),
        }
    except Exception as e:
        logging.error(
            f"[yfinance_utils] Failed to fetch live quote for {ticker} "
            f"(both fast_info and history() fallback failed): {e}"
        )
        return None
