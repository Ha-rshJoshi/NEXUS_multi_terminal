"""SQLAlchemy ORM models for NEXUS's structured tables (prices, sentiment,
model predictions, live ticks) -- news documents/embeddings live in ChromaDB instead."""

from datetime import datetime, timedelta

from extensions import db


class StockPrice(db.Model):
    """
    One row per (ticker, date) for historical candles, and one row per
    live tick for streamed quotes (source='live'). The `source` column is
    what /api/status uses to independently verify "1 year of historical
    data" vs "15 days of live data" have both been satisfied.
    """
    __tablename__ = "stock_price"

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False, index=True)
    date = db.Column(db.String(10), nullable=False, index=True)  # 'YYYY-MM-DD'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    source = db.Column(db.String(10), nullable=False, default="historical")  # 'historical' | 'live'

    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.BigInteger)

    # Technical indicators
    sma_10 = db.Column(db.Float)
    sma_50 = db.Column(db.Float)
    rsi_14 = db.Column(db.Float)
    bb_lower = db.Column(db.Float)
    bb_mid = db.Column(db.Float)
    bb_upper = db.Column(db.Float)
    atr_14 = db.Column(db.Float)

    # Stationary feature inputs (percentage / log returns + normalized indicators)
    pct_return = db.Column(db.Float)
    log_return = db.Column(db.Float)
    rsi_14_z = db.Column(db.Float)
    atr_14_z = db.Column(db.Float)
    bb_width_z = db.Column(db.Float)

    # Unified target (see target_utils.py) -- stored for convenience/debugging only.
    target_return_pct = db.Column(db.Float)

    __table_args__ = (
        db.UniqueConstraint("ticker", "date", "source", name="uq_stockprice_ticker_date_source"),
    )

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "date": self.date,
            "source": self.source,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "sma_10": self.sma_10,
            "sma_50": self.sma_50,
            "rsi_14": self.rsi_14,
            "bb_lower": self.bb_lower,
            "bb_mid": self.bb_mid,
            "bb_upper": self.bb_upper,
            "atr_14": self.atr_14,
            "pct_return": self.pct_return,
            "log_return": self.log_return,
        }


class SentimentMetric(db.Model):
    """Daily aggregated FinBERT sentiment score, mirrored from ChromaDB into
    a flat table purely so Chart.js can pull a fast time series without
    round-tripping through the vector store on every page load."""
    __tablename__ = "sentiment_metric"

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False, index=True)
    date = db.Column(db.String(10), nullable=False, index=True)
    sentiment_score = db.Column(db.Float, nullable=False, default=0.5)
    headline_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("ticker", "date", name="uq_sentiment_ticker_date"),
    )

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "date": self.date,
            "sentiment_score": self.sentiment_score,
            "headline_count": self.headline_count,
        }


class ModelPrediction(db.Model):
    """Per-model (not averaged) forecast breakdown -- MNN / Prophet / LSTM /
    XGBoost each get their own row so the frontend "Model Breakdown
    Accordion" can show Past Day Value / Predicted Value / Error Margin
    separately for every architecture."""
    __tablename__ = "model_prediction"

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False, index=True)
    model_name = db.Column(db.String(30), nullable=False)  # MNN | Prophet | LSTM | XGBoost
    past_day_value = db.Column(db.Float)
    predicted_value = db.Column(db.Float)
    error_margin = db.Column(db.Float)  # RMSE on the held-out purged split
    mae = db.Column(db.Float)
    win_rate = db.Column(db.Float)  # directional accuracy %
    sharpe_ratio = db.Column(db.Float)  # annualized Sharpe of the model's derived long/short strategy; nullable
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "model_name": self.model_name,
            "past_day_value": self.past_day_value,
            "predicted_value": self.predicted_value,
            "error_margin": self.error_margin,
            "mae": self.mae,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class LiveTick(db.Model):
    """Append-only intraday tick history -- one row per live poll, powering
    the dashboard's Live Price graph. Pruned periodically by data_engine.py;
    display-only, never used for model training (see quant_core.pipeline)."""
    __tablename__ = "live_tick"

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.BigInteger)

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "price": self.price,
            "volume": self.volume,
        }


def init_db(app):
    """Call once from app.py after configuring SQLALCHEMY_DATABASE_URI."""
    db.init_app(app)
    with app.app_context():
        db.create_all()


def create_all_tables(engine):
    """Dialect-agnostic table creation for processes running outside Flask's
    app context (main.py, DataEngine, ScraperEngine, evaluation.py). Reuses
    the same ORM metadata as db.create_all() so column types/autoincrement
    are generated correctly per-dialect."""
    db.metadata.create_all(bind=engine, checkfirst=True)


# ---------------------------------------------------------------------------
# Data-sufficiency helpers used by /api/status
# ---------------------------------------------------------------------------

def historical_days_available(ticker: str) -> int:
    """Calendar-day SPAN covered by stored historical rows (not raw row
    count) -- yfinance's 365d history() returns ~252 trading-day rows for a
    365 calendar-day window, so counting rows would never hit 365."""
    bounds = (
        db.session.query(db.func.min(StockPrice.date), db.func.max(StockPrice.date))
        .filter(StockPrice.ticker == ticker, StockPrice.source == "historical")
        .first()
    )
    if not bounds or not bounds[0] or not bounds[1]:
        return 0
    start = datetime.strptime(bounds[0], "%Y-%m-%d")
    end = datetime.strptime(bounds[1], "%Y-%m-%d")
    return (end - start).days


def sentiment_days_available(ticker: str) -> int:
    return (
        db.session.query(SentimentMetric.date)
        .filter(SentimentMetric.ticker == ticker)
        .distinct()
        .count()
    )


def live_feed_last_tick_age_seconds(ticker: str):
    """Seconds since the most recent live_tick row for `ticker`, or None if
    no live tick has ever been recorded. A seconds-based "is the feed
    currently flowing" check, since quant_core.pipeline() never trains on
    live ticks anyway (source='historical' only)."""
    latest = (
        db.session.query(db.func.max(LiveTick.timestamp))
        .filter(LiveTick.ticker == ticker)
        .scalar()
    )
    if not latest:
        return None
    return (datetime.utcnow() - latest).total_seconds()
