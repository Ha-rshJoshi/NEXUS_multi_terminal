"""Joins the structured price table with ChromaDB sentiment metadata and
slices the result into lookback sequences. Not currently wired into the
live pipeline -- quant_core.pipeline() does the equivalent join directly."""

import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from chroma_store import get_news_collection
from target_utils import TARGET_COLUMN, compute_next_day_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataAlignment:
    def __init__(self):
        db_link = os.getenv("DATABASE_URL", "sqlite:///nexus.db")
        self.db_engine = create_engine(db_link)

        self.chroma_client = None
        self.news_collect = None
        try:
            self.news_collect = get_news_collection()
        except Exception as e:
            logging.error(f"Not connected with Vector Memory: {e}")

    def aligned_dataset(self, ticker: str) -> pd.DataFrame:
        logging.info(f"Aligning Multi-Modal matrices for: {ticker}...")

        # text() needed: a raw ":ticker" placeholder string works by accident
        # on SQLite but breaks on Postgres (see quant_core.py's pipeline()).
        query = text("SELECT * FROM stock_price WHERE ticker = :ticker AND source = 'historical' ORDER BY date ASC")
        tech_df = pd.read_sql(query, self.db_engine, params={"ticker": ticker})
        if tech_df.empty:
            raise ValueError(f"No data found in the database for {ticker}.")
        tech_df["date"] = pd.to_datetime(tech_df["date"]).dt.strftime("%Y-%m-%d")

        # from ChromaDB (news_sentiment collection metadata)
        sentiment_records = []
        if self.news_collect is not None:
            chroma_data = self.news_collect.get(where={"ticker": ticker})
            if chroma_data and "metadatas" in chroma_data and chroma_data["metadatas"]:
                for metadata in chroma_data["metadatas"]:
                    if "date" in metadata and "sentiment_score" in metadata:
                        sentiment_records.append({
                            "date": metadata["date"],
                            "finbert_sentiment": metadata["sentiment_score"],
                        })
        sentiment_df = pd.DataFrame(sentiment_records)

        # merge both modalities
        if sentiment_df.empty:
            logging.warning(f"No sentiment vectors found in ChromaDB for {ticker}. Default = 0.5")
            tech_df["finbert_sentiment"] = 0.5
            aligned_df = tech_df
        else:
            sentiment_df = sentiment_df.drop_duplicates(subset="date")
            aligned_df = pd.merge(tech_df, sentiment_df, on="date", how="left")
            aligned_df["finbert_sentiment"] = aligned_df["finbert_sentiment"].ffill()
            aligned_df["finbert_sentiment"] = aligned_df["finbert_sentiment"].fillna(0.5)

        aligned_df = aligned_df.sort_values("date").reset_index(drop=True)
        aligned_df = compute_next_day_target(aligned_df, price_col="close")

        return aligned_df

    def spike_input(self, aligned_df: pd.DataFrame, lookback_window: int = 15):
        """Slices the aligned dataframe into (X_sequences, y_targets) for
        sequence models. `TARGET_COLUMN` and non-feature identifier columns
        are excluded from X; y is read from TARGET_COLUMN directly."""
        exclude_cols = {"date", "ticker", "timestamp", "source", TARGET_COLUMN, "next_day_close"}
        feature_cols = [col for col in aligned_df.columns if col not in exclude_cols]

        X = aligned_df[feature_cols].values
        y = aligned_df[TARGET_COLUMN].values

        X_sequences, y_targets = [], []
        for i in range(len(X) - lookback_window):
            X_sequences.append(X[i:i + lookback_window])
            y_targets.append(y[i + lookback_window])

        return np.array(X_sequences), np.array(y_targets)
