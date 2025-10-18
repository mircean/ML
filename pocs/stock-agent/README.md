# Stock Trading Agent

AI-powered stock trading agent built with LangGraph that analyzes NASDAQ 100 stocks and makes trading recommendations using comprehensive historical data and real-time market news.

## Scripts

**`stock_history_sync.py`** - Downloads NASDAQ 100 stock data and updates portfolio values
```bash
python stock_history_sync.py
```

**`agent.py`** - Runs the trading analysis agent
```bash
python agent.py
```

## Setup

1. Install dependencies: `uv sync`
2. Set environment variables in `.env`:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
   - `LANGSMITH_API_KEY`
3. Run `stock_history_sync.py` first to populate the database
4. Run `agent.py` to start trading analysis

## Features

- **Historical Analysis**: 3 years of NASDAQ 100 stock data (OHLCV, fundamentals, statistics)
- **Market Intelligence**: Real-time news integration via web search
- **Portfolio Management**: Track positions, cash, and performance ($1000 starting capital)
- **Trading Constraints**: Long-only strategy, max 10 positions
- **Data Storage**: Local SQLite database for fast analysis

## Output

The agent provides detailed market analysis and specific buy/sell recommendations with reasoning based on technical indicators, fundamental metrics, and current market conditions.