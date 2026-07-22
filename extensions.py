"""Shared Flask extension singletons (SQLAlchemy + SocketIO), kept out of
app.py to avoid circular imports with db_models.py/data_engine.py/news_scraper.py."""

from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO

db = SQLAlchemy()

# "threading" needs no monkey-patched networking library and is sufficient
# for a handful of dashboard clients; swap to "eventlet" for higher scale.
socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")
