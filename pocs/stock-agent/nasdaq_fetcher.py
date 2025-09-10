import logging
import time
from typing import Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

# Setup logging
config.setup_logging()
logger = logging.getLogger(__name__)


class NasdaqStockFetcher:
    """Fetches list of NASDAQ-listed stocks"""

    def __init__(self):
        self.base_url = "https://www.nasdaq.com/market-activity/stocks/screener"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

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


if __name__ == "__main__":
    fetcher = NasdaqStockFetcher()
    stocks = fetcher.get_nasdaq_stocks()
    print(f"Fetched {len(stocks)} NASDAQ stocks")
