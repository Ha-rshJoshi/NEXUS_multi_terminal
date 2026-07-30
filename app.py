"""NEXUS Flask application -- ties ingestion, the four-model forecast,
portfolio optimization, and the INVESTRA RAG chat into the dashboard."""

import io
import json
import logging
import os
import threading
import time

from flask import Flask, jsonify, render_template, request, send_file
from google import genai
from groq import Groq, RateLimitError as GroqRateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from chroma_store import embed_query, get_news_collection
from data_engine import DataEngine
from db_models import (
    LiveTick, ModelPrediction, SentimentMetric, StockPrice,
    historical_days_available, init_db, live_feed_last_tick_age_seconds,
    sentiment_days_available,
)
from extensions import db, socketio
from news_scraper import ScraperEngine
from portfolio_optimizer import HORIZON_KEYS, optimize_portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TICKERS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "TCS Tata Consultancy",
    "HDFCBANK.NS": "HDFC Bank",
}

# INVESTRA branding -- single source of truth, passed to the template as
# window.NEXUS_INVESTRA_CONFIG. Edit here to reskin without touching HTML/JS.
INVESTRA_CONFIG = {
    "name": "INVESTRA",
    "tagline": "Decoding Sentimental Investment",
    "opening_message": (
        "Hi, I'm INVESTRA. Ask me about market conditions, sentiment, or risk "
        "for any tracked ticker -- and once you run Portfolio Management below, "
        "I can also walk you through why your optimized split looks the way it does."
    ),
    "logo_filename": "investra.png",  # resolved to a real /static/img/... URL in dashboard.html
}

HISTORICAL_TARGET_DAYS = 360   # ~365 calendar days, tolerant of weekends/holidays at the edges
LIVE_TARGET_SECONDS = 90       # poller runs every 60s; a tick this recent means the feed is live
SENTIMENT_TARGET_DAYS = 1      # "loaded sentiment" -- at least one day ingested per ticker

# Appended to every INVESTRA prompt so responses render cleanly in a plain
# text chat bubble (no markdown symbols to escape or strip client-side).
PLAIN_TEXT_STYLE = (
    "Write in plain prose sentences only. Do not use markdown formatting of "
    "any kind -- no headers, no asterisks or bold/italics, no bullet or "
    "numbered lists, no tables."
)

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///nexus.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
init_db(app)
socketio.init_app(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logging.info("Gemini AI Client successfully initialized.")
else:
    gemini_client = None
    logging.warning("GEMINI_API_KEY is not set. Gemini generation will be skipped.")

# Groq fallback, used only when Gemini errors out (e.g. free-tier quota
# exhausted). GROQ_MODEL is overridable via env.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logging.info("Groq fallback client successfully initialized.")
else:
    groq_client = None
    logging.warning("GROQ_API_KEY is not set. No fallback available if Gemini's quota is exhausted.")

data_engine = DataEngine(tickers=list(TICKERS.keys()))
scraper_engine = ScraperEngine(assets_config=TICKERS)

INGESTION_STATE = {"running": False}


# ---------------------------------------------------------------------------
# Background ingestion: drives two independent progress toasts over
# Socket.IO while the frontend shows "Fetching data...".
# ---------------------------------------------------------------------------
def _run_ingestion_job():
    with app.app_context():
        try:
            def live_progress(pct):
                socketio.emit("progress", {"channel": "live_data", "percent": pct})

            def sentiment_progress(pct):
                socketio.emit("progress", {"channel": "sentiment_news", "percent": pct})

            socketio.emit("progress", {"channel": "live_data", "percent": 0})
            socketio.emit("progress", {"channel": "sentiment_news", "percent": 0})

            data_engine.fetch_historical_indicators(progress_callback=live_progress)
            data_engine.stream_live(progress_callback=live_progress)
            scraper_engine.run_cycle(progress_callback=sentiment_progress)

            socketio.emit("ingestion_complete", {"status": "ok"})
        except Exception as e:
            logging.error(f"[app] Ingestion job failed: {e}", exc_info=True)
            socketio.emit("ingestion_complete", {"status": "error", "message": str(e)})
        finally:
            INGESTION_STATE["running"] = False


@app.route("/api/ingest", methods=["POST"])
def trigger_ingestion():
    if INGESTION_STATE["running"]:
        return jsonify({"status": "already_running"}), 202
    INGESTION_STATE["running"] = True
    socketio.start_background_task(_run_ingestion_job)
    return jsonify({"status": "started"}), 202


# ---------------------------------------------------------------------------
# Data sufficiency check -- gates the "Start Prediction" button.
# ---------------------------------------------------------------------------
@app.route("/api/status", methods=["GET"])
def data_status():
    report = {}
    all_ready = True
    for ticker in TICKERS:
        hist_days = historical_days_available(ticker)
        sentiment_days = sentiment_days_available(ticker)
        live_tick_age = live_feed_last_tick_age_seconds(ticker)  # None until the first tick ever lands

        hist_ready = hist_days >= HISTORICAL_TARGET_DAYS
        sentiment_ready = sentiment_days >= SENTIMENT_TARGET_DAYS
        live_ready = live_tick_age is not None and live_tick_age <= LIVE_TARGET_SECONDS
        ticker_ready = hist_ready and sentiment_ready and live_ready
        all_ready = all_ready and ticker_ready

        report[ticker] = {
            "name": TICKERS[ticker],
            "historical_days": hist_days,
            "historical_target": HISTORICAL_TARGET_DAYS,
            "historical_ready": hist_ready,
            "sentiment_days": sentiment_days,
            "sentiment_ready": sentiment_ready,
            "live_tick_age_seconds": int(live_tick_age) if live_tick_age is not None else None,
            "live_target_seconds": LIVE_TARGET_SECONDS,
            "live_ready": live_ready,
            "ready": ticker_ready,
        }

    return jsonify({
        "tickers": report,
        "ingestion_running": INGESTION_STATE["running"],
        "all_ready": all_ready,
    })


# ---------------------------------------------------------------------------
# Manual prediction trigger -- never runs automatically.
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or {}
    selected_tickers = data.get("tickers") or list(TICKERS.keys())

    # Lazy import -- evaluation.py pulls in torch/xgboost/prophet, only
    # needed once a prediction is actually requested.
    from evaluation import performance_backtest

    logging.info(f"Manual prediction requested for: {selected_tickers}")
    report = performance_backtest(selected_tickers)

    # Display-only currency symbol per ticker (₹ for NSE/BSE, $ otherwise).
    for ticker, ticker_report in report.items():
        ticker_report["currency_symbol"] = "₹" if ticker.endswith((".NS", ".BO")) else "$"

    return jsonify({"report": report})


# ---------------------------------------------------------------------------
# Chart.js data feed
# ---------------------------------------------------------------------------
@app.route("/api/chart_data/<ticker>", methods=["GET"])
def chart_data(ticker):
    if ticker not in TICKERS:
        return jsonify({"error": "Unknown ticker"}), 404

    # Historical-only: live/intraday data lives in /api/live_ticks instead.
    price_rows = (
        StockPrice.query.filter_by(ticker=ticker, source="historical")
        .order_by(StockPrice.date.asc())
        .all()
    )
    sentiment_rows = (
        SentimentMetric.query.filter_by(ticker=ticker)
        .order_by(SentimentMetric.date.asc())
        .all()
    )
    prediction_rows = (
        ModelPrediction.query.filter_by(ticker=ticker)
        .order_by(ModelPrediction.created_at.desc())
        .limit(4)
        .all()
    )

    return jsonify({
        "ticker": ticker,
        "price_series": [r.to_dict() for r in price_rows],
        "sentiment_series": [r.to_dict() for r in sentiment_rows],
        "model_comparison": [r.to_dict() for r in prediction_rows],
    })


# ---------------------------------------------------------------------------
# Live Price feed -- full intraday tick history plus the latest price and
# timestamp, separate from the historical chart above.
# ---------------------------------------------------------------------------
@app.route("/api/live_ticks/<ticker>", methods=["GET"])
def live_ticks(ticker):
    if ticker not in TICKERS:
        return jsonify({"error": "Unknown ticker"}), 404

    tick_rows = (
        LiveTick.query.filter_by(ticker=ticker)
        .order_by(LiveTick.timestamp.asc())
        .all()
    )

    latest = tick_rows[-1] if tick_rows else None

    # Last historical close, so the readout can show +/- change even before
    # today's intraday ticks accumulate.
    last_close_row = (
        StockPrice.query.filter_by(ticker=ticker, source="historical")
        .order_by(StockPrice.date.desc())
        .first()
    )
    last_close = last_close_row.close if last_close_row else None

    return jsonify({
        "ticker": ticker,
        "ticks": [r.to_dict() for r in tick_rows],
        "latest_price": latest.price if latest else None,
        "latest_timestamp": latest.timestamp.isoformat() if latest else None,
        "last_close": last_close,
    })


# ---------------------------------------------------------------------------
# Sentiment accordion feed: score + top-5 recent headlines
# ---------------------------------------------------------------------------
def _ticker_sentiment_summary(ticker):
    """Average FinBERT sentiment score + top-3 recent headlines for a
    ticker, pulled from ChromaDB. Shared by /api/news and the portfolio
    auto-explain prompt so both ground on the same numbers."""
    news_collect = get_news_collection()
    chroma_data = news_collect.get(where={"ticker": ticker})

    items = []
    if chroma_data and chroma_data.get("metadatas"):
        for doc, meta in zip(chroma_data.get("documents", []), chroma_data["metadatas"]):
            items.append({
                "date": meta.get("date"),
                "sentiment_score": meta.get("sentiment_score", 0.5),
                "headline": doc,
            })
    items.sort(key=lambda x: x["date"] or "", reverse=True)
    avg_score = sum(i["sentiment_score"] for i in items) / len(items) if items else 0.5
    return avg_score, items


@app.route("/api/news/<ticker>", methods=["GET"])
def news_feed(ticker):
    if ticker not in TICKERS:
        return jsonify({"error": "Unknown ticker"}), 404

    avg_score, items = _ticker_sentiment_summary(ticker)
    return jsonify({"ticker": ticker, "average_sentiment": avg_score, "top_headlines": items[:5]})


# ---------------------------------------------------------------------------
# RAG chat endpoint
# ---------------------------------------------------------------------------
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
    retry=retry_if_exception_type(GroqRateLimitError),
)
def _call_groq(prompt):
    """Groq call, retried a couple of times with backoff on 429 -- Gemini
    being down (see _generate_rag_response) can suddenly send every request
    through Groq at once, which is enough to trip its own free-tier rate
    limit even though each individual call would normally be fine."""
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def _generate_rag_response(prompt):
    """Tries Gemini first; on failure (typically a 429 once the free tier's
    quota is used up, or a 5xx when Gemini itself is overloaded) falls back
    to Groq. Returns (response_text, provider_label). Raises if neither
    client is configured, or the last exception if both fail."""
    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                # Rolling alias -- stays correct across Gemini model
                # retirements instead of needing a pinned-version bump.
                model="gemini-flash-latest",
                contents=prompt,
            )
            return response.text, "gemini-flash-latest"
        except Exception as e:
            logging.warning(f"[app] Gemini generation failed ({e}); falling back to Groq if configured.")
            if not groq_client:
                raise

    if groq_client:
        return _call_groq(prompt), GROQ_MODEL

    raise RuntimeError("No RAG generation provider is configured (GEMINI_API_KEY and GROQ_API_KEY both absent).")


@app.route("/api/chat", methods=["POST"])
def chat():
    if not gemini_client and not groq_client:
        return jsonify({"error": "Neither GEMINI_API_KEY nor GROQ_API_KEY is configured on the server."}), 500

    data = request.json or {}
    user_query = (data.get("query") or "").strip()
    ticker = data.get("ticker")
    # Client-supplied: the frontend's last computed portfolio, so INVESTRA
    # can answer "why this split?" grounded in the actual numbers.
    portfolio_context = data.get("portfolio_context")

    if not user_query:
        return jsonify({"error": "query is required"}), 400

    try:
        query_embedding = embed_query(user_query)

        news_collect = get_news_collection()
        where_clause = {"ticker": ticker} if ticker in TICKERS else None
        results = news_collect.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where=where_clause,
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        context_lines = []
        for doc, meta in zip(documents, metadatas):
            context_lines.append(
                f"- [{meta.get('ticker')} | {meta.get('date')} | "
                f"sentiment {meta.get('sentiment_score', 0.5):.2f}] {doc}"
            )
        context_block = "\n".join(context_lines) if context_lines else "No relevant news context was retrieved."

        portfolio_block = ""
        if portfolio_context:
            portfolio_block = f"""

The user also has a Portfolio Management result already computed this
session -- reference it if the question is about their portfolio, weights,
allocation, or "why this split":
{json.dumps(portfolio_context, indent=2, default=str)}"""

        prompt = f"""You are INVESTRA, the NEXUS Execution Desk AI -- a grounded quantitative
trading and portfolio advisor.

Use ONLY the retrieved news context below to inform any factual claims about
recent events. If the context doesn't cover the question, say so plainly
instead of guessing.

Retrieved Context (top-5 semantically relevant news items):
{context_block}{portfolio_block}

User Question: "{user_query}"

Respond concisely and clinically, in the tone of an institutional trading desk analyst. {PLAIN_TEXT_STYLE}"""

        response_text, provider_used = _generate_rag_response(prompt)

        return jsonify({
            "response": response_text,
            "generated_by": provider_used,
            "context_used": [
                {"ticker": m.get("ticker"), "date": m.get("date"), "headline": d}
                for d, m in zip(documents, metadatas)
            ],
        })
    except Exception as e:
        logging.error(f"[app] /api/chat failed: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate AI response."}), 500


# ---------------------------------------------------------------------------
# Portfolio Management -- Markowitz optimization over a user-selected
# subset of the tracked tickers (see portfolio_optimizer.py for the math).
# ---------------------------------------------------------------------------
@app.route("/api/portfolio_optimize", methods=["POST"])
def portfolio_optimize():
    data = request.json or {}

    selected_tickers = data.get("tickers") or list(TICKERS.keys())
    selected_tickers = [t for t in selected_tickers if t in TICKERS]
    if not selected_tickers:
        return jsonify({"error": "At least one valid ticker must be selected."}), 400

    horizon = data.get("horizon")
    if horizon not in HORIZON_KEYS:
        return jsonify({"error": f"horizon must be one of {sorted(HORIZON_KEYS)}."}), 400

    amount = data.get("amount")
    amount_min = data.get("amount_min")
    amount_max = data.get("amount_max")
    try:
        if amount_min is None or amount_max is None:
            if amount is None:
                return jsonify({"error": "Provide either 'amount' or both 'amount_min' and 'amount_max'."}), 400
            amount_min = amount_max = float(amount)
        else:
            amount_min, amount_max = float(amount_min), float(amount_max)
    except (TypeError, ValueError):
        return jsonify({"error": "amount / amount_min / amount_max must be numeric."}), 400

    if amount_min <= 0 or amount_max <= 0 or amount_max < amount_min:
        return jsonify({"error": "Invalid amount range -- amount_max must be >= amount_min > 0."}), 400

    try:
        result = optimize_portfolio(selected_tickers, amount_min, amount_max, horizon)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"[app] /api/portfolio_optimize failed: {e}", exc_info=True)
        return jsonify({"error": "Failed to compute portfolio."}), 500

    # Auto-explain the result using the same generation helper /api/chat uses.
    explanation, explained_by = None, None
    try:
        sentiment_lines = []
        for t in selected_tickers:
            avg_score, _ = _ticker_sentiment_summary(t)
            sentiment_lines.append(f"- {t}: average sentiment {avg_score:.2f} (0 = bearish, 1 = bullish)")
        sentiment_block = "\n".join(sentiment_lines)

        explain_prompt = f"""You are INVESTRA, the NEXUS Execution Desk AI -- a grounded quantitative
portfolio advisor. A Markowitz mean-variance optimization was just run for
this user, with this result:

{json.dumps(result, indent=2, default=str)}

Recent sentiment context for the selected tickers:
{sentiment_block}

In 3-5 concise, clinical sentences, explain WHY the optimizer picked this
particular split -- grounding your reasoning in each stock's historical
return/volatility profile and the sentiment context above, and whichever
of expected return / volatility drove the "{result['objective_used']}"
choice for a "{horizon}"-term horizon. Do not restate all the numbers
verbatim, the user can already see them -- interpret them instead. {PLAIN_TEXT_STYLE}"""
        explanation, explained_by = _generate_rag_response(explain_prompt)
    except Exception as e:
        logging.warning(f"[app] Portfolio auto-explain failed (numbers are still returned): {e}")

    result["explanation"] = explanation
    result["explained_by"] = explained_by
    return jsonify(result)


# ---------------------------------------------------------------------------
# PDF reports -- built from data the frontend already fetched (a /api/predict
# ticker report, and/or the current /api/portfolio_optimize result and
# INVESTRA chat transcript), so generating a PDF never re-runs a backtest.
# ---------------------------------------------------------------------------
def _ticker_ai_summary(ticker, ticker_report):
    """Short Gemini/Groq-written paragraph explaining the winning model for
    one ticker's report, in the same voice/plain-text style as /api/chat
    and the portfolio auto-explain."""
    try:
        prompt = f"""You are INVESTRA, the NEXUS Execution Desk AI. A backtest for {ticker}
just produced this per-model report:

{json.dumps(ticker_report, indent=2, default=str)}

In 2-4 concise, clinical sentences, explain why "{ticker_report.get('winner')}"
outperformed the other models on this backtest, and give a one-line
deployment takeaway. Do not restate every number verbatim -- interpret them.
{PLAIN_TEXT_STYLE}"""
        summary, _ = _generate_rag_response(prompt)
        return summary
    except Exception as e:
        logging.warning(f"[app] AI analyst summary failed for {ticker} (report still generated): {e}")
        return None


def _ticker_report_section(ticker, ticker_report):
    """Gathers everything report_generator needs for one ticker's section:
    display name, currency, historical price series for the chart, sentiment
    summary, and an AI analyst summary."""
    price_rows = (
        StockPrice.query.filter_by(ticker=ticker, source="historical")
        .order_by(StockPrice.date.asc())
        .all()
    )
    avg_sentiment, sentiment_items = _ticker_sentiment_summary(ticker)

    return {
        "ticker": ticker,
        "display_name": TICKERS.get(ticker, ticker),
        "currency_symbol": "₹" if ticker.endswith((".NS", ".BO")) else "$",
        "ticker_report": ticker_report,
        "sentiment_summary": {
            "average_sentiment": avg_sentiment,
            "headlines": sentiment_items[:5],
        },
        "ai_summary": _ticker_ai_summary(ticker, ticker_report),
        "price_dates": [r.date for r in price_rows],
        "price_closes": [r.close for r in price_rows],
    }


@app.route("/api/report/ticker_pdf", methods=["POST"])
def report_ticker_pdf():
    data = request.json or {}
    ticker = data.get("ticker")
    ticker_report = data.get("report")
    if ticker not in TICKERS:
        return jsonify({"error": "Unknown ticker"}), 404
    if not ticker_report:
        return jsonify({"error": "'report' (the /api/predict result for this ticker) is required."}), 400

    import report_generator as rg

    try:
        section = _ticker_report_section(ticker, ticker_report)
        pdf_bytes = rg.build_ticker_pdf(
            section["ticker"], section["display_name"], section["currency_symbol"],
            section["ticker_report"], section["sentiment_summary"], section["ai_summary"],
            section["price_dates"], section["price_closes"],
        )
    except Exception as e:
        logging.error(f"[app] /api/report/ticker_pdf failed for {ticker}: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate report."}), 500

    safe_ticker = ticker.replace(".", "_")
    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True,
        download_name=f"NEXUS_{safe_ticker}_Report.pdf",
    )


@app.route("/api/report/full_pdf", methods=["POST"])
def report_full_pdf():
    data = request.json or {}
    tickers_reports = data.get("tickers_reports") or {}
    portfolio_result = data.get("portfolio_result")
    chat_transcript = data.get("chat_transcript") or []

    valid_reports = {t: r for t, r in tickers_reports.items() if t in TICKERS and r}
    if not valid_reports:
        return jsonify({"error": "No valid ticker reports were supplied."}), 400

    import report_generator as rg

    try:
        # Spaced out rather than back-to-back: each ticker triggers its own
        # AI analyst summary call, and firing them all at once is what
        # tripped Groq's rate limit the one time Gemini was down for every
        # single one of them (see _call_groq's retry for the remaining risk).
        items = list(valid_reports.items())
        sections = []
        for i, (t, r) in enumerate(items):
            sections.append(_ticker_report_section(t, r))
            if i < len(items) - 1:
                time.sleep(1.2)
        pdf_bytes = rg.build_full_report_pdf(sections, portfolio_result, chat_transcript)
    except Exception as e:
        logging.error(f"[app] /api/report/full_pdf failed: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate full report."}), 500

    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True,
        download_name="NEXUS_Full_Report.pdf",
    )


@app.route("/", methods=["GET"])
def dashboard():
    return render_template("dashboard.html", tickers=TICKERS, investra_config=INVESTRA_CONFIG)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
