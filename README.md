# NEXUS

A quant trading dashboard I built for tracking and forecasting three NSE stocks — Reliance (`RELIANCE.NS`), TCS (`TCS.NS`), and HDFC Bank (`HDFCBANK.NS`). It pulls in price history, news sentiment, and live quotes, runs four different forecasting models against each stock, and lets you build an optimized portfolio out of the three.

Built this as a personal project to actually understand how multi-model forecasting and portfolio theory work in practice, not just in theory. It's research/learning tooling, not something you should trade real money off of.

## What's in it

**Four models, compared honestly.** LSTM, Prophet, a custom Multiplicative Neural Network, and XGBoost each forecast the next day's return independently. I made a deliberate call not to average them into one ensemble number — you see each model's predicted price, RMSE, MAE, win rate, and Sharpe ratio side by side, and the "winner" is whichever one actually performed best on held-out data, not a blended guess.

Backtesting uses a purged train/validation/test split so the models aren't accidentally leaking information from the future into their own training.

**News + sentiment.** Headlines get pulled and scored, stored in Postgres for the charts and in ChromaDB (as embeddings) so the chat assistant can semantically search them later.

**Portfolio Management.** Give it an amount and a time horizon, and it runs a Markowitz optimization across whichever of the three stocks you pick. Short horizon leans toward minimum volatility, long horizon leans toward max Sharpe, medium horizon sits somewhere in between on the actual efficient frontier (not just a 50/50 blend of the other two). It rounds down to whole shares since you can't buy fractional shares on NSE, and shows you the leftover cash.

**INVESTRA.** A chat widget in the corner, grounded in the retrieved news for whatever you're asking about. If you've just run a portfolio optimization, it can also explain why it picked that particular split. Runs on Gemini, and falls back to Groq automatically if Gemini's free tier quota runs out (which it does, a lot).

## How it's put together

Three Docker containers: Postgres for structured data, a Flask app for the dashboard/API, and a background worker that keeps polling for live prices and fresh news on a loop. `quant_core.py` holds all the shared model/data logic so the live-forecast path (`models.py`) and the backtest path (`evaluation.py`) are guaranteed to produce comparable numbers — that was a bug I ran into early on and had to fix by centralizing everything.

Data comes from yfinance mostly. I originally wanted Finnhub as the primary source but it turns out their free tier just doesn't cover NSE/BSE for live quotes or news — every call gets rejected — so it tries Finnhub first and quietly falls back to yfinance.

## Stack

Flask + Flask-SocketIO + Postgres + ChromaDB on the backend, PyTorch/Prophet/XGBoost/scipy for the modeling, Bootstrap + Chart.js on the frontend, all wired together with Docker Compose.

## Running it

Copy `.env.example` to `.env` and fill in your own keys — you'll need a Gemini key, a Finnhub key (free tier's fine), and optionally a Groq key for the fallback. Pick your own Postgres username/password too.

```
cp .env.example .env
docker compose up -d --build
```

Then open `http://localhost:5000`. It'll show ingestion progress first, and once a year of history + sentiment + at least one live tick are all in, "Start Prediction" lights up.

## Where things live

- `quant_core.py` — models, config, the purged data split, everything shared
- `models.py` — live forecasting
- `evaluation.py` — backtesting + logging results to `model_registry.json`
- `portfolio_optimizer.py` — the Markowitz stuff
- `data_engine.py` / `news_scraper.py` — price polling + news ingestion
- `app.py` — all the Flask routes and the SocketIO progress events
- `db_models.py` — the Postgres side of things
- `chroma_store.py` — the vector store side
- `templates/` / `static/` — the actual dashboard

Model weights, the vector store, and scrape checkpoints aren't committed — they get rebuilt the first time you run it. Nothing in `.env` is committed either; `docker-compose.yml` just reads from it.
