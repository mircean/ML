import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

import config
from stock_history_database import StockHistoryDatabase

# Setup logging
config.setup_logging()
logger = logging.getLogger(__name__)


class StockFetcher:
    """Comprehensive stock fetcher for NASDAQ 100"""

    def __init__(self, db):
        self.db = db
        self.rate_limit_delay = 0.5  # Delay between requests to avoid rate limiting

        # Date range for 3 years
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=3 * 365)

    def fetch_stock_info(self, symbol: str) -> Optional[Dict]:
        """Fetch comprehensive stock information using yfinance"""
        # Add delay to respect rate limits
        time.sleep(self.rate_limit_delay)

        try:
            ticker = yf.Ticker(symbol)

            # Get stock info
            info = ticker.info
            assert info is not None, f"No data available for {symbol}"
            assert info.get("regularMarketPrice") is not None, (
                f"No data available for {symbol}"
            )

            # Get historical data
            history = ticker.history(
                start=self.start_date.strftime("%Y-%m-%d"),
                end=self.end_date.strftime("%Y-%m-%d"),
                auto_adjust=False,
            )
            assert not history.empty, f"No historical data available for {symbol}"

            # Get actions (splits and dividends)
            actions = ticker.actions
            if actions.empty:
                # Create empty DataFrame with proper structure
                actions = pd.DataFrame(columns=["Dividends", "Stock Splits"])

            # Get financials (may not be available for all stocks)
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow

            return {
                "symbol": symbol,
                "info": info,
                "history": history,
                "actions": actions,
                "financials": financials,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
            }

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def process_stock_data(self, stock_data: Dict) -> bool:
        """Process and store stock data in database"""
        try:
            symbol = stock_data["symbol"]
            info = stock_data["info"]

            # Insert stock metadata
            self.db.insert_stock(
                symbol=symbol,
                name=info.get("longName", info.get("shortName", symbol)),
                sector=info.get("sector"),
                industry=info.get("industry"),
            )

            # Insert price data
            if not stock_data["history"].empty:
                self.db.insert_price_data(symbol, stock_data["history"])

            # Insert actions (splits and dividends)
            if not stock_data["actions"].empty:
                self.db.insert_stock_actions(symbol, stock_data["actions"])

            # Prepare fundamentals data
            fundamentals = {
                "marketCap": info.get("marketCap"),
                "enterpriseValue": info.get("enterpriseValue"),
                "trailingPE": info.get("trailingPE"),
                "pegRatio": info.get("pegRatio"),
                "priceToBook": info.get("priceToBook"),
                "priceToSalesTrailing12Months": info.get(
                    "priceToSalesTrailing12Months"
                ),
                "enterpriseToRevenue": info.get("enterpriseToRevenue"),
                "enterpriseToEbitda": info.get("enterpriseToEbitda"),
                "debtToEquity": info.get("debtToEquity"),
                "returnOnEquity": info.get("returnOnEquity"),
                "returnOnAssets": info.get("returnOnAssets"),
                "grossMargins": info.get("grossMargins"),
                "operatingMargins": info.get("operatingMargins"),
                "profitMargins": info.get("profitMargins"),
                "beta": info.get("beta"),
                "dividendYield": info.get("dividendYield"),
                "payoutRatio": info.get("payoutRatio"),
                "sharesOutstanding": info.get("sharesOutstanding"),
                "floatShares": info.get("floatShares"),
            }

            # Clean up None values and convert to proper types
            cleaned_fundamentals = {}
            for key, value in fundamentals.items():
                if value is not None and not (
                    isinstance(value, float) and np.isnan(value)
                ):
                    try:
                        cleaned_fundamentals[key] = float(value)
                    except (ValueError, TypeError):
                        pass

            if cleaned_fundamentals:
                self.db.insert_fundamentals(symbol, cleaned_fundamentals)

            # Prepare statistics data
            statistics = {
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "fiftyDayAverage": info.get("fiftyDayAverage"),
                "twoHundredDayAverage": info.get("twoHundredDayAverage"),
                "averageVolume10days": info.get("averageVolume10days"),
                "averageVolume3Month": info.get("averageVolume3Month"),
                "sharesShort": info.get("sharesShort"),
                "shortRatio": info.get("shortRatio"),
                "shortPercentOfFloat": info.get("shortPercentOfFloat"),
            }

            # Clean up statistics data
            cleaned_statistics = {}
            for key, value in statistics.items():
                if value is not None and not (
                    isinstance(value, float) and np.isnan(value)
                ):
                    try:
                        if key in [
                            "averageVolume10days",
                            "averageVolume3Month",
                            "sharesShort",
                        ]:
                            cleaned_statistics[key] = int(value)
                        else:
                            cleaned_statistics[key] = float(value)
                    except (ValueError, TypeError):
                        pass

            if cleaned_statistics:
                self.db.insert_statistics(symbol, cleaned_statistics)

            return True

        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}")
            return False

    def fetch_single_stock(self, symbol: str) -> bool:
        """Fetch and process data for a single stock"""
        logger.info(f"Fetching data for {symbol}")
        stock_data = self.fetch_stock_info(symbol)

        if stock_data:
            return self.process_stock_data(stock_data)
        return False

    def fetch_stocks(
        self,
        symbols: List[str],
    ):
        """Fetch data for NASDAQ 100 stocks sequentially"""
        logger.info(f"Processing {len(symbols)} stocks...")

        # Truncate all tables before fetching new data
        logger.info("Truncating all tables...")
        self.db.truncate_all_tables()

        # Process stocks sequentially with progress bar
        successful = 0
        failed = 0

        with tqdm(symbols, desc="Fetching stock data") as pbar:
            for symbol in pbar:
                success = self.fetch_single_stock(symbol)
                if success:
                    successful += 1
                else:
                    failed += 1
                pbar.set_postfix({"✓": successful, "✗": failed})

        logger.info(f"Completed: {successful} successful, {failed} failed")

        # Print database statistics
        stats = self.db.get_database_stats()
        logger.info(f"Database stats: {stats}")

    def update_single_stock(self, symbol: str) -> bool:
        """Update data for a single stock"""
        logger.info(f"Updating {symbol}...")
        return self.fetch_single_stock(symbol)

    def get_stock_summary(self, symbol: str) -> Optional[Dict]:
        """Get a summary of stock data"""
        try:
            data = self.db.get_stock_data(symbol)
            if data["prices"].empty:
                return None

            latest_price = data["prices"].iloc[-1]
            price_range = data["prices"]["close"]

            summary = {
                "symbol": symbol,
                "latest_date": latest_price["date"],
                "latest_close": latest_price["close"],
                "volume": latest_price["volume"],
                "52_week_high": price_range.max(),
                "52_week_low": price_range.min(),
                "price_change_1y": (
                    (latest_price["close"] - data["prices"].iloc[0]["close"])
                    / data["prices"].iloc[0]["close"]
                )
                * 100,
                "data_points": len(data["prices"]),
            }

            if not data["fundamentals"].empty:
                latest_fundamentals = data["fundamentals"].iloc[0]
                summary.update(
                    {
                        "pe_ratio": latest_fundamentals.get("pe_ratio"),
                        "market_cap": latest_fundamentals.get("market_cap"),
                        "dividend_yield": latest_fundamentals.get("dividend_yield"),
                    }
                )

            return summary

        except Exception as e:
            logger.error(f"Error getting summary for {symbol}: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    db = StockHistoryDatabase()
    fetcher = StockFetcher(db)

    # Fetch data for a few major stocks as test
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print("Testing with major stocks...")
    fetcher.fetch_stocks(symbols=test_symbols, force_update=True)

    # Print summaries
    for symbol in test_symbols:
        summary = fetcher.get_stock_summary(symbol)
        if summary:
            print(
                f"\n{symbol}: ${summary['latest_close']:.2f}, "
                f"Volume: {summary['volume']:,}, "
                f"Data points: {summary['data_points']}"
            )
