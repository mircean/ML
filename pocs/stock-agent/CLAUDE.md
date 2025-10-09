# CLAUDE.md

Development context for Claude Code when working with this LangGraph stock trading agent.

## Quick Commands
```bash
uv sync              # Install deps
python sync_data.py  # Download data (run first!)
python agent.py      # Run agent
```

## Core Architecture

### LangGraph Workflow Structure
The trading agent is implemented as a LangGraph StateGraph with the following key nodes:
- **initialize_agent**: Loads portfolio data and sets up system context
- **market_analysis**: Main LLM analysis node that uses tools to gather market data
- **tools**: Executes SQL queries and web searches
- **recommend_trades**: Generates final trading recommendations

The workflow uses conditional routing where `market_analysis` loops back to `tools` until sufficient data is gathered (up to `MAX_TOOL_CALLS` iterations), then proceeds to final recommendations.

### Database Schema
The SQLite database (`nasdaq_stocks.db`) contains 5 main tables:
- `stocks`: Stock metadata (symbol, name, sector, industry)
- `stock_prices`: Daily OHLCV price data
- `stock_fundamentals`: Financial ratios and metrics (P/E, market cap, margins, etc.)
- `stock_statistics`: Trading statistics (52-week highs/lows, moving averages, short interest)
- `stock_actions`: Corporate actions (splits, dividends)

### Memory System
The agent memory database (`agent_memory.db`) stores historical analysis:
- `agent_scores`: Daily stock scores (composite, momentum, quality, technical scores)
- Automatically updated after each agent run (daily overwrite)
- Enables trend analysis: "Has NVDA been consistently strong for 5 days?"

### Key Configuration
- Maximum tool calls per session: 10 (configurable via `MAX_TOOL_CALLS`)
- Portfolio limits: Max 10 positions, $1000 starting cash
- LLM Model: GPT-5 with temperature 0.1
- Trading strategy: Long-only, buy and hold good stocks

## Development Commands

### Setup and Dependencies
```bash
# Install dependencies
uv sync

# Set up environment variables in .env:
# OPENAI_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY
```

### Core Operations
```bash
# Download/update stock data (run this first)
python sync_data.py

# Run the trading agent
python agent.py

### Database Operations
The `sync_data.py` script completely refreshes the database on each run using:
- `NasdaqStockFetcher`: Downloads NASDAQ 100 stock list
- `StockFetcher`: Downloads comprehensive stock data via yfinance
- `StockDatabase`: Manages SQLite database operations and schema

## Important Implementation Details

### Code Organization
- **Import-time safety**: Only logger definitions and imports happen at module level
- **Main function pattern**: All setup (environment loading, logging config, LLM initialization) happens in `main()`
- **Factory functions**: Node functions that need dependencies use factory pattern (e.g., `create_market_analysis_node()`)
- **Self-contained execution**: Everything needed for the application runs within `main()`

### State Management
The `TradingState` class tracks:
- Messages (conversation history)
- Portfolio data (cash, positions)
- Tool call counter
- Analysis and trading completion flags

### Tool Integration
Three main tools are available to the LLM:
- `run_sql`: Executes SQL queries against the stock database with comprehensive error handling
- `search_market_news`: Uses Tavily API for web search
- `retrieve_last_N_days_of_analysis`: Queries historical scores for specific stocks to identify trends

### Portfolio Management
- Portfolio data is stored in `portfolio.json`
- The `update_portfolio_values()` function recalculates position values based on current stock prices
- Trading constraints are enforced through system prompts and configuration

### Prompt Engineering
The system uses structured prompts with:
- Complete database schema documentation embedded in prompts
- Step-by-step analysis instructions
- Reflection prompts after each tool call
- Clear trading constraints and objectives

## File Structure

**Core Files:**
- `agent.py`: Main LangGraph trading agent implementation
- `config.py`: Configuration constants and logging setup
- `prompts.py`: System prompts and database schema documentation

**Data Management:**
- `database.py`: StockDatabase and MemoryDatabase classes for SQLite operations
- `sync_data.py`: Data downloading and portfolio value updates
- `schema.sql`: Database table definitions and indexes
- `schema_memory.sql`: Memory database schema for historical scores
- `stock_fetcher.py`: yfinance integration for stock data
- `nasdaq_fetcher.py`: NASDAQ 100 stock list fetching

**Data Files:**
- `nasdaq_stocks.db`: SQLite database (generated)
- `agent_memory.db`: Memory database for historical scores (generated)
- `portfolio.json`: Portfolio state (generated)
- `log.txt`: Application logs (generated)

## Development Preferences

- Prefer simpler, more readable code over clever optimizations
- Favor explicit over implicit behavior
- Keep functions focused and single-purpose
- Minimize external dependencies when possible
- **No code execution at import time**: All setup code should be in `main()` functions, with only logger definitions at module level


## Development Notes

When working with this codebase:
- Always run `sync_data.py` before `agent.py` to ensure fresh data
- The database is completely rebuilt on each sync to ensure data consistency
- Portfolio positions are updated with current market prices during sync
- The agent's analysis is structured around iterative tool use with reflection
- All SQL queries should be SELECT-only; database modifications happen through the sync process