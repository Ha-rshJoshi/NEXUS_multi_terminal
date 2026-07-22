"""Standalone pipeline worker: runs DataEngine + ScraperEngine's ingestion
loop outside the Flask process (the docker-compose pipeline_worker container)."""

import logging
import time

from data_engine import DataEngine
from news_scraper import ScraperEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s")


def main():
    logging.info("STARTING Quantitative Trading Data Pipeline.....")

    assets_config = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "TCS Tata Consultancy",
        "HDFCBANK.NS": "HDFC Bank",
    }
    tickers = list(assets_config.keys())

    market_engine = DataEngine(tickers=tickers)
    news_engine = ScraperEngine(assets_config=assets_config)
    logging.info("Both core engines started successfully.")

    # Baseline load
    market_engine.fetch_historical_indicators()
    news_engine.run_cycle()

    # 60s keeps Live Price near-real-time without adding to yfinance's
    # existing rate-limit risk (see yfinance_utils.fetch_live_quote).
    LIVE_POLL_INTERVAL_SECONDS = 60

    while True:
        try:
            logging.info("Starting live multi-modal data collection.....")
            market_engine.stream_live()
            logging.info(f"Cycle finished. Pipeline sleeping for {LIVE_POLL_INTERVAL_SECONDS}s.....")
            time.sleep(LIVE_POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logging.info("Pipeline safely stopped by user execution.")
            break
        except Exception as e:
            logging.error(f"Critical pipeline failure in main loop: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
