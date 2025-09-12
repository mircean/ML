# Stock Trading Agent

AI-powered stock trading agent built with LangGraph that analyzes NASDAQ 100 stocks and makes trading recommendations.

## Scripts

**`sync_data.py`** - Downloads NASDAQ 100 stock data and updates portfolio values
```bash
python sync_data.py
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
3. Run `sync_data.py` first to populate the database
4. Run `agent.py` to start trading analysis

## Features

- 3 years of NASDAQ 100 historical data
- Fundamental and technical analysis
- Market news integration
- Portfolio tracking and optimization
- SQLite database with comprehensive stock metrics