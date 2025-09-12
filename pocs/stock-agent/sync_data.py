#!/usr/bin/env python3
"""
NASDAQ Stock Database Builder

Downloads NASDAQ 100 stock data and stores it in a SQLite database.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime

import config
from database import StockDatabase
from nasdaq_fetcher import NasdaqStockFetcher
from stock_fetcher import StockFetcher

# Setup logging
config.setup_logging()
logger = logging.getLogger(__name__)

# Reduce yfinance logging noise
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def init_portfolio():
    portfolio_path = config.PORTFOLIO_FILE

    if not os.path.exists(portfolio_path):
        portfolio = {
            "cash": config.DEFAULT_CASH,
            "positions": {},
            "total_value": config.DEFAULT_CASH,
            "positions_value": 0,
        }
        with open(portfolio_path, "w") as f:
            json.dump(portfolio, f, indent=2)


def update_portfolio_values():
    """Update portfolio values based on current stock prices."""
    db_path = config.DATABASE_PATH
    portfolio_path = config.PORTFOLIO_FILE
    init_portfolio()
    assert os.path.exists(portfolio_path), f"Portfolio file not found: {portfolio_path}"

    # Load portfolio
    with open(portfolio_path, "r") as f:
        portfolio = json.load(f)

    positions = portfolio.get("positions", {})
    if not positions:
        logger.info("No positions in portfolio, skipping update")
        return

    symbols = list(positions.keys())

    updated_positions = {}
    total_value_change = 0

    logger.info("=" * 50)
    logger.info("UPDATING PORTFOLIO VALUES")
    logger.info("=" * 50)

    for symbol in symbols:
        # Get latest price from database
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT close, date FROM stock_prices 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 1
            """,
                (symbol,),
            )
            result = cursor.fetchall()

        assert result and len(result) > 0
        current_price = float(result[0][0])
        date = result[0][1]

        # Update position
        position = positions[symbol].copy()
        shares = position["shares"]
        old_value = position.get("value", position.get("cost", 0))
        new_value = shares * current_price

        position["value"] = round(new_value, 2)
        updated_positions[symbol] = position

        value_change = new_value - old_value
        total_value_change += value_change

        logger.info(
            f"{symbol}: {shares} shares @ ${current_price:.2f} = ${new_value:.2f} "
            f"({value_change:+.2f}) [{date}]"
        )

        # Update portfolio
        portfolio["positions"] = updated_positions

        # Calculate total portfolio value
        cash = portfolio.get("cash", 0)
        positions_value = sum(pos.get("value", 0) for pos in updated_positions.values())
        total_portfolio_value = cash + positions_value

        portfolio["total_value"] = round(total_portfolio_value, 2)
        portfolio["positions_value"] = round(positions_value, 2)

        # Save updated portfolio
        with open(portfolio_path, "w") as f:
            json.dump(portfolio, f, indent=2)

        logger.info("PORTFOLIO SUMMARY:")
        logger.info(f"Cash: ${cash:.2f}")
        logger.info(f"Positions value: ${positions_value:.2f}")
        logger.info(f"Total portfolio value: ${total_portfolio_value:.2f}")
        logger.info(f"Total value change: {total_value_change:+.2f}")
        logger.info(f"Portfolio updated: {portfolio_path}")
        logger.info("=" * 50)


def main():
    """Main application entry point"""
    try:
        logger.info("NASDAQ Stock Database Builder")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Initialize components
        nasdaq_fetcher = NasdaqStockFetcher()
        db = StockDatabase()
        fetcher = StockFetcher(db)

        # Fetch NASDAQ 100 stock list
        nasdaq_stocks = nasdaq_fetcher.get_nasdaq_stocks()
        symbols = [stock["symbol"] for stock in nasdaq_stocks if stock["symbol"]]

        # Download and store NASDAQ 100 data
        logger.info("Downloading NASDAQ 100 stock data...")
        fetcher.fetch_stocks(symbols)

        # Show final stats
        stats = db.get_database_stats()
        logger.info("=" * 50)
        logger.info("DATABASE STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total stocks: {stats['total_stocks']:,}")
        logger.info(f"Price records: {stats['price_records']:,}")
        logger.info(
            f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}"
        )
        logger.info(f"Database size: {stats['db_size_mb']} MB")
        logger.info("=" * 50)

        # Update portfolio values with latest prices
        logger.info("Updating portfolio with latest stock prices...")
        update_portfolio_values()

        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
