"""
Prompts for the Stock Trading Agent
"""

import config


def get_database_schema() -> str:
    """
    Get the database schema documentation.

    Returns:
        Database schema documentation string
    """
    return """# NASDAQ Stock Database Schema

You have access to a comprehensive NASDAQ 100 stock database with 3 years of historical data. The database contains the following tables:

## Tables Overview

### 1. `stocks` - Stock metadata
Primary table containing basic information about each stock.

**Schema:**
```sql
CREATE TABLE stocks (
    symbol TEXT PRIMARY KEY,        -- Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    name TEXT,                     -- Company name (e.g., 'Apple Inc.')
    sector TEXT,                   -- Business sector (e.g., 'Technology')
    industry TEXT,                 -- Industry classification
    date TIMESTAMP                 -- Last updated timestamp
);
```

### 2. `stock_prices` - Daily price data
Historical daily trading data for each stock.

**Schema:**
```sql
CREATE TABLE stock_prices (
    symbol TEXT,                   -- Stock ticker (foreign key to stocks)
    date DATE,                     -- Trading date (YYYY-MM-DD)
    open REAL,                     -- Opening price
    high REAL,                     -- Highest price of the day
    low REAL,                      -- Lowest price of the day
    close REAL,                    -- Closing price
    adj_close REAL,                -- Adjusted closing price (accounts for splits/dividends)
    volume INTEGER,                -- Number of shares traded
    PRIMARY KEY (symbol, date)
);
```

### 3. `stock_fundamentals` - Financial metrics and ratios
Comprehensive fundamental analysis data for each stock.

**Schema:**
```sql
CREATE TABLE stock_fundamentals (
    symbol TEXT,                   -- Stock ticker
    date DATE,                     -- Date of the data
    market_cap REAL,              -- Market capitalization
    enterprise_value REAL,        -- Enterprise value
    pe_ratio REAL,                -- Price-to-Earnings ratio
    peg_ratio REAL,               -- Price/Earnings to Growth ratio
    price_to_book REAL,           -- Price-to-Book ratio
    price_to_sales REAL,          -- Price-to-Sales ratio
    ev_to_revenue REAL,           -- Enterprise Value to Revenue
    ev_to_ebitda REAL,            -- Enterprise Value to EBITDA
    debt_to_equity REAL,          -- Debt-to-Equity ratio
    return_on_equity REAL,        -- Return on Equity (%)
    return_on_assets REAL,        -- Return on Assets (%)
    gross_margin REAL,            -- Gross profit margin (%)
    operating_margin REAL,        -- Operating margin (%)
    profit_margin REAL,           -- Net profit margin (%)
    beta REAL,                    -- Stock beta (volatility vs market)
    dividend_yield REAL,          -- Dividend yield (%)
    payout_ratio REAL,            -- Dividend payout ratio (%)
    shares_outstanding REAL,      -- Number of shares outstanding
    float_shares REAL,            -- Number of shares available for trading
    PRIMARY KEY (symbol, date)
);
```

### 4. `stock_actions` - Corporate actions
Stock splits and dividend payments.

**Schema:**
```sql
CREATE TABLE stock_actions (
    symbol TEXT,                   -- Stock ticker
    date DATE,                     -- Date of the action
    action_type TEXT,              -- 'split' or 'dividend'
    value REAL,                    -- Split ratio or dividend amount
    ratio TEXT,                    -- Text description of split ratio
    PRIMARY KEY (symbol, date, action_type)
);
```

### 5. `stock_statistics` - Trading statistics
Statistical data about stock performance and trading patterns.

**Schema:**
```sql
CREATE TABLE stock_statistics (
    symbol TEXT,                   -- Stock ticker
    date DATE,                     -- Date of the data
    fifty_two_week_high REAL,     -- 52-week high price
    fifty_two_week_low REAL,      -- 52-week low price
    fifty_day_average REAL,       -- 50-day moving average
    two_hundred_day_average REAL, -- 200-day moving average
    avg_volume_10day INTEGER,     -- Average volume over 10 days
    avg_volume_3month INTEGER,    -- Average volume over 3 months
    shares_short REAL,            -- Number of shares sold short
    short_ratio REAL,             -- Short interest ratio
    short_percent_float REAL,     -- Short interest as % of float
    PRIMARY KEY (symbol, date)
);
```

## Key Relationships

- All tables link via the `symbol` field
- `stock_prices` contains daily time series data
- `stock_fundamentals` and `stock_statistics` typically have one record per stock (latest data)
- `stock_actions` contains historical events (may have multiple records per stock)

## Data Coverage

- **Time Range:** Approximately 3 years of historical data
- **Stocks:** NASDAQ 100 constituents (~100 stocks)
- **Update Frequency:** The database is refreshed completely on each run

## Common Query Patterns

**Get latest stock prices:**
```sql
SELECT symbol, close, volume, date 
FROM stock_prices 
WHERE date = (SELECT MAX(date) FROM stock_prices)
ORDER BY volume DESC;
```

**Find stocks with highest P/E ratios:**
```sql
SELECT s.symbol, s.name, f.pe_ratio, f.market_cap
FROM stocks s
JOIN stock_fundamentals f ON s.symbol = f.symbol
WHERE f.pe_ratio IS NOT NULL
ORDER BY f.pe_ratio DESC;
```

**Get price performance over time:**
```sql
SELECT symbol, date, close,
       (close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date)) / 
       LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) * 100 as daily_return
FROM stock_prices
WHERE symbol = 'AAPL'
ORDER BY date DESC
LIMIT 30;
```
"""


def get_system_prompt(portfolio_cash: float, portfolio_positions: dict) -> str:
    """
    Get the main system prompt for the trading agent.

    Args:
        portfolio_cash: Available cash amount
        portfolio_positions: Current stock positions
        database_schema: Database schema documentation

    Returns:
        Formatted system prompt string
    """
    return f"""You are a stock trading agent with the following constraints:
- Starting capital: ${config.DEFAULT_CASH}
- Maximum positions: {config.MAX_POSITIONS} stocks  
- Strategy: Long-only, buy and hold good stocks, sell inferior stocks
- Goal: Beat NASDAQ performance

Analyze the market, find good investment opportunities, and make trading decisions.
If you find a stock that is better than any the stocks in current portfolio, sell the inferior stock and buy the better one.
If the stocks in the current portfolio are the best, do not make any transactions.
Do not neccesarily use all the cash to buy stocks, buy only stocks that are worth it.

{get_database_schema()}

You have access to web search tools for additional market research.
Think carefully after each tool call and explain your reasoning.

Please use the available tools:
1. Use `run_sql` to execute SQL queries against this database to answer user questions about stocks, financial metrics, price movements, and market analysis.
2. Use search_market_news to get recent market trends and news

Keep analyzing and researching until you have enough information to make confident trading recommendations.

After your analysis, provide specific BUY/SELL recommendations with:
- Stock symbol
- Number of shares to buy/sell
- Reasoning for the recommendation

Current portfolio status:
- Cash: ${portfolio_cash:.2f}
- Positions: {portfolio_positions}
"""


def get_thinking_prompt() -> str:
    """
    Get the prompt for the thinking node.

    Returns:
        Thinking prompt string
    """
    return """Based on the analysis results you just received, please think through:

1. What do the market conditions tell you?
2. Which stocks look most promising based on fundamentals and momentum?
3. Are there any risks or concerns you should consider?
4. How does this information help you make trading decisions?

Provide your detailed analysis and reasoning."""


def get_summary_prompt(
    portfolio_cash: float, portfolio_positions: dict, recommendations_text: str
) -> str:
    """
    Get the prompt for the summary/recommendations display.

    Args:
        portfolio_cash: Available cash amount
        portfolio_positions: Current stock positions
        recommendations_text: The AI's final recommendations

    Returns:
        Formatted summary string
    """
    return f"""
🎯 TRADING ANALYSIS COMPLETE

📊 Portfolio Status:
- Cash Available: ${portfolio_cash:.2f}
- Current Positions: {len(portfolio_positions)}/{config.MAX_POSITIONS}

📋 FINAL RECOMMENDATIONS:
{recommendations_text}

✅ Analysis session completed!
"""
