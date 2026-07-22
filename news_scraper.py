"""Historical news ingestion for the RAG vector store -- Finnhub paginated
backfill with a yfinance (recent-only) fallback for NSE/BSE, scored by
FinBERT and embedded into ChromaDB."""

import json
import logging
import os
import time
from datetime import datetime, timedelta

import finnhub
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from chroma_store import embed_texts, get_news_collection
from finnhub_utils import is_permission_error, is_retryable
from yfinance_utils import fetch_recent_news as yf_fetch_recent_news

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

HISTORICAL_WINDOW_DAYS = 365
WEEK_STEP_DAYS = 7
FINNHUB_THROTTLE_SECONDS = 1.5  # primary flow control, NOT tenacity
CHECKPOINT_DIR = "scrape_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

_sentiment_analyzer = None


def get_sentiment_analyzer():
    """Lazy-loads the FinBERT pipeline on first use so importing this module
    (e.g. for the /api/chat RAG path) doesn't force-load a 400MB transformer
    that route doesn't need."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        from transformers import pipeline as hf_pipeline
        logging.info("Loading FinBERT Financial Sentiment Engine...")
        _sentiment_analyzer = hf_pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return _sentiment_analyzer


# ---------------------------------------------------------------------------
# Checkpointing (skip already-ingested history on subsequent restarts)
# ---------------------------------------------------------------------------
def _checkpoint_path(ticker: str) -> str:
    safe_ticker = ticker.replace("/", "_")
    return os.path.join(CHECKPOINT_DIR, f"{safe_ticker}.json")


def _load_checkpoint(ticker: str) -> dict:
    path = _checkpoint_path(ticker)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_checkpoint(ticker: str, data: dict) -> None:
    with open(_checkpoint_path(ticker), "w") as f:
        json.dump(data, f, indent=2)


# tenacity is a secondary fallback (transient errors only); sleep(1.5)
# between calls is what actually enforces the 60/min limit.
# retry_if_exception excludes 401/403 so permission errors fail fast.
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception(is_retryable),
)
def _safe_company_news_call(finnhub_client, ticker: str, from_date: str, to_date: str):
    return finnhub_client.company_news(ticker, _from=from_date, to=to_date)


def scrape_finnhub_historical(finnhub_client, ticker: str, progress_callback=None):
    """Paginates the last HISTORICAL_WINDOW_DAYS in WEEK_STEP_DAYS chunks,
    resuming from the ticker's checkpoint if one exists. Returns
    (articles, blocked_by_permission); blocked_by_permission=True means
    Finnhub denied access outright (401/403), so the caller should fall
    back to yfinance rather than retry the remaining weeks."""
    now = datetime.utcnow()
    window_start = now - timedelta(days=HISTORICAL_WINDOW_DAYS)

    start_unix = int(window_start.timestamp())
    end_unix = int(now.timestamp())
    logging.info(f"[news_scraper] {ticker} historical window UNIX: {start_unix} -> {end_unix}")

    checkpoint = _load_checkpoint(ticker)
    resume_str = checkpoint.get("last_completed_week_end")
    cursor = pd.to_datetime(resume_str) if resume_str else window_start
    if cursor >= pd.Timestamp(now):
        logging.info(f"[news_scraper] {ticker} already fully ingested as of last run -- skipping.")
        if progress_callback:
            progress_callback(100)
        return [], False

    total_weeks = max(1, int(HISTORICAL_WINDOW_DAYS / WEEK_STEP_DAYS))
    weeks_done = int((cursor - window_start).days / WEEK_STEP_DAYS)

    aggregated_articles = []
    while cursor < pd.Timestamp(now):
        chunk_start = cursor
        chunk_end = min(cursor + timedelta(days=WEEK_STEP_DAYS), now)
        from_str = chunk_start.strftime("%Y-%m-%d")
        to_str = chunk_end.strftime("%Y-%m-%d")

        try:
            articles = _safe_company_news_call(finnhub_client, ticker, from_str, to_str) or []
            for art in articles:
                aggregated_articles.append({
                    "headline": art.get("headline", ""),
                    "datetime": art.get("datetime"),  # unix seconds, from Finnhub
                })
            logging.info(f"[news_scraper] {ticker}: week {from_str}->{to_str} -> {len(articles)} articles.")
        except Exception as e:
            if is_permission_error(e):
                # Permanent for this key/symbol -- stop paginating and
                # fall back to yfinance instead of repeating the failure.
                logging.warning(
                    f"[news_scraper] {ticker}: Finnhub denied access to company-news "
                    f"({e}). Falling back to yfinance (recent news only -- see "
                    f"yfinance_utils.py). Not retrying the remaining Finnhub weeks."
                )
                return aggregated_articles, True
            logging.error(f"[news_scraper] Failed week {from_str}->{to_str} for {ticker} after retries: {e}")

        _save_checkpoint(ticker, {"last_completed_week_end": to_str})
        weeks_done += 1
        if progress_callback:
            progress_callback(min(100, int(weeks_done / total_weeks * 100)))

        cursor = chunk_end
        time.sleep(FINNHUB_THROTTLE_SECONDS)  # primary rate-limit control

    return aggregated_articles, False


# ---------------------------------------------------------------------------
# Sentiment scoring + real embedding + storage
# ---------------------------------------------------------------------------
def _score_headline(title: str) -> float:
    try:
        result = get_sentiment_analyzer()(title[:512])[0]
        label = result["label"].lower()
        score = result["score"]
        if label == "positive":
            return 0.5 + (score * 0.5)
        elif label == "negative":
            return 0.5 - (score * 0.5)
        return 0.5
    except Exception:
        return 0.5


def analyze_and_store_historical_news(finnhub_client, ticker: str, engine, progress_callback=None):
    logging.info(f"\n----- PROCESSING HISTORICAL/RECENT NEWS FOR: {ticker} -----")
    news_collect = get_news_collection()

    raw_articles = []
    if finnhub_client:
        finnhub_articles, blocked = scrape_finnhub_historical(finnhub_client, ticker, progress_callback=progress_callback)
        raw_articles.extend(finnhub_articles)
        if blocked:
            raw_articles.extend(yf_fetch_recent_news(ticker))
    else:
        logging.info(f"[news_scraper] {ticker}: no Finnhub client configured -- using yfinance directly.")
        raw_articles.extend(yf_fetch_recent_news(ticker))

    if progress_callback:
        progress_callback(100)

    if not raw_articles:
        logging.warning(f"[news_scraper] No new headlines for {ticker} (may already be cached, or none available right now).")
        return

    # Group headlines by calendar date
    daily_news_buckets = {}
    for article in raw_articles:
        headline = article.get("headline")
        raw_ts = article.get("datetime")
        if not headline or not raw_ts:
            continue
        try:
            date_str = datetime.utcfromtimestamp(int(raw_ts)).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            continue
        daily_news_buckets.setdefault(date_str, []).append(headline)

    logging.info(f"[news_scraper] {ticker}: {len(daily_news_buckets)} distinct days of headlines to process.")

    for date_str, headlines in daily_news_buckets.items():
        doc_id = f"{ticker}_{date_str}"

        # Skip days already successfully persisted (survives app restarts).
        existing = news_collect.get(ids=[doc_id])
        if existing and existing.get("ids"):
            continue

        top_headlines = headlines[:10]
        daily_scores = [_score_headline(h) for h in top_headlines]
        final_daily_score = float(np.mean(daily_scores)) if daily_scores else 0.5

        summary_doc = " | ".join(headlines[:5])
        embedding = embed_texts([summary_doc])[0]

        try:
            news_collect.upsert(
                embeddings=[embedding],
                documents=[summary_doc],
                metadatas=[{
                    "ticker": ticker,
                    "date": date_str,
                    "sentiment_score": final_daily_score,
                    "headline_count": len(headlines),
                }],
                ids=[doc_id],
            )
            _upsert_sentiment_metric(engine, ticker, date_str, final_daily_score, len(headlines))
        except Exception as e:
            logging.error(f"[news_scraper] Could not save {date_str} for {ticker} to ChromaDB: {e}")

    logging.info(f"[news_scraper] ChromaDB vector memory updated for {ticker} over {HISTORICAL_WINDOW_DAYS} days.")


def _upsert_sentiment_metric(engine, ticker, date_str, score, headline_count):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT OR REPLACE INTO sentiment_metric
                (ticker, date, sentiment_score, headline_count, created_at)
                VALUES (:ticker, :date, :score, :count, :created_at)
            """) if str(engine.url).startswith("sqlite") else text("""
                INSERT INTO sentiment_metric (ticker, date, sentiment_score, headline_count, created_at)
                VALUES (:ticker, :date, :score, :count, :created_at)
                ON CONFLICT (ticker, date) DO UPDATE SET
                    sentiment_score = EXCLUDED.sentiment_score,
                    headline_count = EXCLUDED.headline_count
            """),
            {
                "ticker": ticker, "date": date_str, "score": score,
                "count": headline_count, "created_at": datetime.utcnow(),
            },
        )


class ScraperEngine:
    """Orchestrates historical news ingestion for every tracked ticker.
    `progress_callback(percent:int)` is optional and is what app.py's
    background thread uses to drive the "Storing sentiment/news" toast."""

    def __init__(self, assets_config: dict = None, finnhub_api_key: str = None):
        self.assets_config = assets_config or {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "TCS Tata Consultancy",
            "HDFCBANK.NS": "HDFC Bank",
        }
        api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self.finnhub_client = finnhub.Client(api_key=api_key) if api_key else None

        db_link = os.getenv("DATABASE_URL", "sqlite:///nexus.db")
        self.engine = create_engine(db_link)

        # Ensures sentiment_metric exists even if instantiated before DataEngine.
        from db_models import create_all_tables
        create_all_tables(self.engine)

    def run_cycle(self, progress_callback=None):
        logging.info("INITIATING HISTORICAL FINBERT + NEWS PIPELINE (Finnhub, falling back to yfinance)...")
        logging.info(f"Targeted assets: {list(self.assets_config.keys())}")

        if not self.finnhub_client:
            logging.info(
                "[news_scraper] FINNHUB_API_KEY missing -- using yfinance directly for all tickers "
                "(recent news only, see yfinance_utils.py)."
            )

        total = len(self.assets_config)
        for idx, (ticker, _keyword) in enumerate(self.assets_config.items()):

            def _week_progress(pct_within_ticker, _idx=idx):
                if progress_callback:
                    overall = int(((_idx + (pct_within_ticker / 100.0)) / total) * 100)
                    progress_callback(min(overall, 99))

            analyze_and_store_historical_news(
                self.finnhub_client, ticker, self.engine, progress_callback=_week_progress
            )

        if progress_callback:
            progress_callback(100)


if __name__ == "__main__":
    target_stocks = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "TCS Tata Consultancy",
        "HDFCBANK.NS": "HDFC Bank",
    }
    engine = ScraperEngine(target_stocks)
    engine.run_cycle()
