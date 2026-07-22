"""Hybrid market-data ingestion engine: yfinance for the historical backfill,
Finnhub for live price/news with an automatic yfinance fallback for NSE/BSE tickers."""

import logging
import os
import time
from datetime import datetime, timedelta

import finnhub
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sqlalchemy import create_engine, text
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from finnhub_utils import is_permission_error, is_rate_limit_error, is_retryable
from yfinance_utils import fetch_live_quote as yf_fetch_live_quote
from yfinance_utils import fetch_recent_news as yf_fetch_recent_news

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

HISTORICAL_WINDOW_DAYS = 365
ZSCORE_WINDOW = 20
LIVE_TICK_RETENTION_DAYS = 5  # append-only intraday history is pruned past this window


def _rate_limited_retry(func):
    """Exponential backoff for retryable errors (429/5xx/transient network
    issues) only. 401/403 permission errors raise immediately instead of
    burning 6 backoff attempts on a failure that will never succeed."""

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1.5, min=1, max=30),
        retry=retry_if_exception(is_retryable),
    )
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if is_rate_limit_error(exc):
                logging.warning(f"[data_engine] Finnhub 429 rate limit hit, backing off: {exc}")
            elif is_permission_error(exc):
                # Debug-level: expected once a yfinance fallback exists; the
                # caller logs its own clear message when it switches sources.
                logging.debug(f"[data_engine] Finnhub permission error (expected, falling back): {exc}")
            raise
    return wrapper


class DataEngine:
    def __init__(self, tickers, finnhub_api_key: str = None):
        # Tickers must include .NS or .BO (e.g., "RELIANCE.NS")
        self.tickers = tickers

        self.db_link = os.getenv("DATABASE_URL", "sqlite:///nexus.db")
        self.engine = create_engine(self.db_link)

        api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        if not api_key:
            logging.warning("[data_engine] FINNHUB_API_KEY missing -- live streaming/news disabled.")
            self.finnhub_client = None
        else:
            self.finnhub_client = finnhub.Client(api_key=api_key)

        # De-dupe guard for the live poller
        self.last_data = {ticker: {"price": None, "volume": None} for ticker in tickers}

        self._init_database()

    # ------------------------------------------------------------------
    # SCHEMA (mirrors db_models.StockPrice / SentimentMetric table names,
    # so raw-engine writes here and Flask-SQLAlchemy reads in app.py agree)
    #
    # Table creation is delegated to db_models.create_all_tables(), which
    # uses SQLAlchemy's ORM metadata to emit correct per-dialect DDL
    # (Postgres SERIAL/IDENTITY vs SQLite AUTOINCREMENT) instead of
    # hand-written, single-dialect SQL.
    # ------------------------------------------------------------------
    def _init_database(self):
        from db_models import create_all_tables
        create_all_tables(self.engine)

    # ------------------------------------------------------------------
    # STATIONARY FEATURE ENGINEERING
    # ------------------------------------------------------------------
    def _stationary_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Adds percentage returns, log returns, and z-score normalized
        technical indicators. Raw price columns (open/high/low/close) are
        kept only for display purposes -- models.py/quant_core.py consume
        the *_z / *_return columns, never raw trending price levels."""
        df = history_df.copy()

        # TREND (kept for reference/display; not fed to models directly)
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=50, append=True)

        # MOMENTUM
        df.ta.rsi(length=14, append=True)

        # VOLATILITY
        bbands = df.ta.bbands(length=20, std=2.0)
        if bbands is not None:
            df = pd.concat([df, bbands], axis=1)
        df.ta.atr(length=14, append=True)

        # Normalize pandas_ta's generated column names to our schema
        rename_map = {}
        for col in df.columns:
            lc = col.lower()
            if lc.startswith("sma_10"):
                rename_map[col] = "sma_10"
            elif lc.startswith("sma_50"):
                rename_map[col] = "sma_50"
            elif lc.startswith("rsi_14"):
                rename_map[col] = "rsi_14"
            elif lc.startswith("bbl_"):
                rename_map[col] = "bb_lower"
            elif lc.startswith("bbm_"):
                rename_map[col] = "bb_mid"
            elif lc.startswith("bbu_"):
                rename_map[col] = "bb_upper"
            elif lc.startswith("atrr_") or lc == "atr_14" or lc.startswith("atr_"):
                rename_map[col] = "atr_14"
        df = df.rename(columns=rename_map)

        # Stationary returns (percentage + log)
        df["pct_return"] = df["close"].pct_change() * 100.0
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Z-score normalize the technical indicators over a rolling window
        # so they are scale-free and centered, instead of raw trending values.
        for raw_col, z_col in (("rsi_14", "rsi_14_z"), ("atr_14", "atr_14_z")):
            if raw_col in df.columns:
                roll_mean = df[raw_col].rolling(ZSCORE_WINDOW, min_periods=5).mean()
                roll_std = df[raw_col].rolling(ZSCORE_WINDOW, min_periods=5).std().replace(0, np.nan)
                df[z_col] = (df[raw_col] - roll_mean) / roll_std
            else:
                df[z_col] = np.nan

        if {"bb_upper", "bb_lower", "bb_mid"}.issubset(df.columns):
            bb_width = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
            roll_mean = bb_width.rolling(ZSCORE_WINDOW, min_periods=5).mean()
            roll_std = bb_width.rolling(ZSCORE_WINDOW, min_periods=5).std().replace(0, np.nan)
            df["bb_width_z"] = (bb_width - roll_mean) / roll_std
        else:
            df["bb_width_z"] = np.nan

        df = df.dropna(subset=["pct_return", "log_return"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # HISTORICAL DATA VIA YFINANCE (365 days)
    # ------------------------------------------------------------------
    def fetch_historical_indicators(self, progress_callback=None):
        logging.info("----- FETCHING 365-DAY HISTORICAL DATA FROM YFINANCE -----")
        total = len(self.tickers)

        for idx, ticker_symbol in enumerate(self.tickers, start=1):
            try:
                stock = yf.Ticker(ticker_symbol)
                history_df = stock.history(period=f"{HISTORICAL_WINDOW_DAYS}d")

                if history_df.empty:
                    logging.warning(f"No historical data found for {ticker_symbol}.")
                    continue

                history_df = history_df.reset_index()
                history_df.columns = [c.lower() for c in history_df.columns]

                if history_df["date"].dt.tz is not None:
                    history_df["date"] = history_df["date"].dt.tz_localize(None)
                history_df["date"] = history_df["date"].dt.strftime("%Y-%m-%d")

                processed_df = self._stationary_features(history_df)
                processed_df["ticker"] = ticker_symbol
                processed_df["source"] = "historical"
                processed_df["timestamp"] = datetime.utcnow()

                keep_cols = [
                    "ticker", "date", "timestamp", "source",
                    "open", "high", "low", "close", "volume",
                    "sma_10", "sma_50", "rsi_14",
                    "bb_lower", "bb_mid", "bb_upper", "atr_14",
                    "pct_return", "log_return",
                    "rsi_14_z", "atr_14_z", "bb_width_z",
                ]
                for c in keep_cols:
                    if c not in processed_df.columns:
                        processed_df[c] = np.nan
                processed_df = processed_df[keep_cols]

                self._upsert_stock_price(processed_df, ticker_symbol, source="historical")
                logging.info(f"Historical data for {ticker_symbol} stored ({len(processed_df)} rows).")

            except Exception as e:
                logging.error(f"Failed to fetch history for {ticker_symbol}: {e}")

            if progress_callback:
                progress_callback(int(idx / total * 100))

        if progress_callback:
            progress_callback(100)

    def _upsert_stock_price(self, df: pd.DataFrame, ticker: str, source: str):
        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM stock_price WHERE ticker = :t AND source = :s"),
                {"t": ticker, "s": source},
            )
        df.to_sql("stock_price", self.engine, if_exists="append", index=False)

    # ------------------------------------------------------------------
    # LIVE STREAMING VIA FINNHUB (with tenacity backoff)
    # ------------------------------------------------------------------
    @_rate_limited_retry
    def _finnhub_quote(self, ticker: str):
        return self.finnhub_client.quote(ticker)

    def stream_live(self, progress_callback=None):
        logging.info("----- POLLING LIVE QUOTES (FINNHUB, falling back to yfinance) -----")

        now = datetime.utcnow()
        current_date = now.strftime("%Y-%m-%d")
        current_ts = now
        total = len(self.tickers)

        with self.engine.begin() as conn:
            for idx, current_ticker in enumerate(self.tickers, start=1):
                quote_open = quote_high = quote_low = None
                price = volume = None

                # Finnhub first; falls back to yfinance below on a
                # permission error (free tier doesn't cover NSE/BSE quotes).
                if self.finnhub_client:
                    try:
                        quote = self._finnhub_quote(current_ticker)
                        price = quote.get("c")
                        volume = quote.get("v", 0) or 0
                        quote_open, quote_high, quote_low = quote.get("o"), quote.get("h"), quote.get("l")
                    except Exception as e:
                        if is_permission_error(e):
                            logging.info(
                                f"[data_engine] Finnhub has no live-quote access for {current_ticker} "
                                f"(plan/exchange limitation) -- falling back to yfinance."
                            )
                        else:
                            logging.warning(f"[data_engine] Finnhub quote failed for {current_ticker}: {e}")
                        price = None

                if not price:
                    fallback = yf_fetch_live_quote(current_ticker)
                    if fallback:
                        price = fallback["price"]
                        volume = fallback["volume"]
                        quote_open, quote_high, quote_low = fallback["open"], fallback["high"], fallback["low"]
                    else:
                        logging.info(
                            f"[data_engine] No live price available for {current_ticker} from either "
                            f"source this cycle (exchange likely closed, or yfinance has no fresh quote)."
                        )

                try:
                    if not price:
                        continue

                    # Append-only intraday log, recorded every poll regardless
                    # of price change -- display-only, never read by
                    # quant_core.pipeline() (models train on source='historical' only).
                    conn.execute(
                        text("""
                            INSERT INTO live_tick (ticker, timestamp, price, volume)
                            VALUES (:ticker, :ts, :price, :volume)
                        """),
                        {"ticker": current_ticker, "ts": current_ts, "price": price, "volume": volume},
                    )

                    if (
                        price == self.last_data[current_ticker]["price"]
                        and volume == self.last_data[current_ticker]["volume"]
                    ):
                        continue

                    # No timestamp embedded here: current_ts is naive UTC while
                    # the log line's own prefix is local time -- mixing the two
                    # makes the poll look stale when it isn't.
                    logging.info(f"{current_ticker} --> Price: {price:.2f}, Volume: {volume:,}")

                    conn.execute(
                        text("""
                            INSERT OR REPLACE INTO stock_price
                            (ticker, date, timestamp, source, open, high, low, close, volume)
                            VALUES (:ticker, :date, :ts, 'live', :o, :h, :l, :c, :v)
                        """) if self.db_link.startswith("sqlite") else text("""
                            INSERT INTO stock_price (ticker, date, timestamp, source, open, high, low, close, volume)
                            VALUES (:ticker, :date, :ts, 'live', :o, :h, :l, :c, :v)
                            ON CONFLICT (ticker, date, source) DO UPDATE SET
                                close = EXCLUDED.close, volume = EXCLUDED.volume, timestamp = EXCLUDED.timestamp
                        """),
                        {
                            "ticker": current_ticker, "date": current_date, "ts": current_ts,
                            "o": quote_open, "h": quote_high, "l": quote_low,
                            "c": price, "v": volume,
                        },
                    )

                    self.last_data[current_ticker]["price"] = price
                    self.last_data[current_ticker]["volume"] = volume

                except Exception as e:
                    logging.error(f"Error polling live data for {current_ticker}: {e}")

                if progress_callback:
                    progress_callback(int(idx / total * 100))

        self._prune_old_ticks()

        if progress_callback:
            progress_callback(100)

    def _prune_old_ticks(self):
        """Bounds live_tick's growth -- intraday granularity is only useful
        for a few recent trading days; anything older is deleted so the
        append-only table doesn't grow unbounded forever."""
        cutoff = datetime.utcnow() - timedelta(days=LIVE_TICK_RETENTION_DAYS)
        try:
            with self.engine.begin() as conn:
                conn.execute(text("DELETE FROM live_tick WHERE timestamp < :cutoff"), {"cutoff": cutoff})
        except Exception as e:
            logging.warning(f"[data_engine] Failed to prune old live_tick rows: {e}")

    # Quick recent-news pull for the sentiment accordion / chat context
    # refresh; falls back to yfinance when Finnhub denies the ticker's
    # exchange. Full historical paginated ingestion lives in news_scraper.py.
    @_rate_limited_retry
    def _finnhub_company_news(self, ticker: str, from_date: str, to_date: str):
        return self.finnhub_client.company_news(ticker, _from=from_date, to=to_date)

    def fetch_recent_news(self, ticker: str, lookback_days: int = 7):
        now = datetime.utcnow()
        window_start = now - timedelta(days=lookback_days)

        # Precise UNIX timestamps computed to restrict the scrape window
        start_unix = int(window_start.timestamp())
        end_unix = int(now.timestamp())
        logging.info(
            f"[data_engine] Recent news window for {ticker}: "
            f"{start_unix} -> {end_unix} ({lookback_days}d)"
        )

        if self.finnhub_client:
            from_date = datetime.utcfromtimestamp(start_unix).strftime("%Y-%m-%d")
            to_date = datetime.utcfromtimestamp(end_unix).strftime("%Y-%m-%d")
            try:
                return self._finnhub_company_news(ticker, from_date, to_date) or []
            except Exception as e:
                if is_permission_error(e):
                    logging.info(
                        f"[data_engine] Finnhub has no news access for {ticker} "
                        f"(plan/exchange limitation) -- falling back to yfinance."
                    )
                else:
                    logging.error(f"Failed to fetch recent news for {ticker}: {e}")

        # yfinance fallback -- recent-only, see yfinance_utils.py docstring.
        return yf_fetch_recent_news(ticker, lookback_days=lookback_days)
