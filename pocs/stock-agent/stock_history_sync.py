"""
NASDAQ Stock Database Builder

Downloads NASDAQ 100 stock data and stores it in a SQLite database.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Dict, List

import agent
import config
import requests
from portfolio_database import PortfolioDatabase
from stock_fetcher_yahoo import StockFetcher
from stock_history_database import StockHistoryDatabase

logger = logging.getLogger(__name__)


class NasdaqStockFetcher:
    """Fetches list of NASDAQ-listed stocks"""

    def __init__(self):
        self.base_url = "https://www.nasdaq.com/market-activity/stocks/screener"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    def get_nasdaq_stocks(self) -> List[Dict[str, str]]:
        """
        Fetch list of NASDAQ stocks
        Returns list of dictionaries with symbol, name, and other basic info
        """
        # Use the official NASDAQ 100 API endpoint
        url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"

        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        stocks = []

        rows = data["data"]["data"]["rows"]
        for row in rows:
            symbol = row.get("symbol", "").strip()
            assert symbol, f"No symbol found for {row}"
            # ignore GOOGL, same as GOOG
            if symbol == "GOOGL":
                continue

            stocks.append(
                {
                    "symbol": symbol,
                    "name": row.get("companyName", row.get("name", "")).strip(),
                    "market_cap": "",  # NASDAQ 100 API may not include market cap
                    "sector": row.get("sector", ""),
                    "industry": row.get("industry", ""),
                }
            )

        logger.info(f"Fetched {len(stocks)} NASDAQ 100 stocks")
        return stocks


def update_portfolio_values(cfg: config.Config):
    """Update portfolio values based on current stock prices and save daily snapshot."""
    stock_history_db_path = config.STOCK_HISTORY_DATABASE_PATH

    # Load portfolio using agent's function
    portfolio = agent.load_portfolio()

    positions = portfolio.get("positions", {})
    if not positions:
        logger.info("No positions in portfolio, skipping update")
        return

    cash = portfolio.get("cash", cfg.default_cash)

    # Get the most recent date in the stock_prices table
    with sqlite3.connect(stock_history_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT MAX(date) FROM stock_prices
            """
        )
        result = cursor.fetchone()

    assert result and result[0], "No stock price data found in database"
    latest_price_date = result[0]

    logger.info("=" * 50)
    logger.info("UPDATING PORTFOLIO VALUES")
    logger.info("=" * 50)
    logger.info(f"Using prices as of: {latest_price_date}")

    total_positions_value = 0

    for symbol, position in positions.items():
        shares = position.get("shares", 0)

        # Get latest price from market database
        with sqlite3.connect(stock_history_db_path) as conn:
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

        assert result, f"No price data found for {symbol}"

        current_price = float(result[0][0])
        price_date = result[0][1]
        current_value = shares * current_price
        total_positions_value += current_value

        # Update position with current price and value
        position["current_price"] = current_price
        position["current_value"] = current_value

        logger.info(f"{symbol}: {shares} shares @ ${current_price:.2f} = ${current_value:.2f} [{price_date}]")

    total_portfolio_value = cash + total_positions_value

    # Update portfolio totals
    portfolio["positions_value"] = total_positions_value
    portfolio["total_value"] = total_portfolio_value
    portfolio["prices_as_of"] = latest_price_date

    # Save portfolio using agent's function (handles JSON + DB snapshot)
    agent.save_portfolio(portfolio)

    logger.info("PORTFOLIO SUMMARY:")
    logger.info(f"Cash: ${cash:.2f}")
    logger.info(f"Positions value: ${total_positions_value:.2f}")
    logger.info(f"Total portfolio value: ${total_portfolio_value:.2f}")
    logger.info(f"Portfolio updated and snapshot saved for {latest_price_date}")
    logger.info("=" * 50)


def main():
    """Main application entry point"""
    cfg = config.parse_config()

    # Setup logging
    config.setup_logging()

    # Reduce yfinance logging noise
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("NASDAQ Stock Database Builder")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize components
    nasdaq_fetcher = NasdaqStockFetcher()
    db = StockHistoryDatabase()
    fetcher = StockFetcher(db)

    # Fetch NASDAQ 100 stock list and include ^NDX (NASDAQ 100 Index)
    nasdaq_stocks = nasdaq_fetcher.get_nasdaq_stocks()
    symbols = ["^NDX"] + [stock["symbol"] for stock in nasdaq_stocks if stock["symbol"]]

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
    logger.info(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    logger.info(f"Database size: {stats['db_size_mb']} MB")
    logger.info("=" * 50)

    # Update portfolio values with latest prices
    logger.info("Updating portfolio with latest stock prices...")
    update_portfolio_values(cfg)

    # Save NASDAQ 100 index history for benchmarking
    logger.info("Saving NASDAQ 100 index history...")
    market_db_path = config.STOCK_HISTORY_DATABASE_PATH
    with sqlite3.connect(market_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT close, date FROM stock_prices
            WHERE symbol = '^NDX'
            ORDER BY date DESC
            LIMIT 1
            """
        )
        ndx_result = cursor.fetchall()

    assert ndx_result, "NASDAQ 100 (^NDX) data not found in database"
    ndx_value = float(ndx_result[0][0])
    ndx_date = ndx_result[0][1]

    portfolio_db = PortfolioDatabase()
    portfolio_db.save_nasdaq100_snapshot(date=ndx_date, value=ndx_value)
    logger.info(f"Saved NASDAQ 100 snapshot: {ndx_value:.2f} for {ndx_date}")

    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
