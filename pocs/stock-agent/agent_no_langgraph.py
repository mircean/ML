#!/usr/bin/env python3
"""
Trading Agent without LangGraph

An AI trading agent that replicates the exact functionality of the LangGraph version
but uses a simple loop-based approach instead of the graph framework.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import config
import prompts
from database import MemoryDatabase

# Load environment variables
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Helper functions
def print_portfolio(portfolio: dict, label: str = "Portfolio"):
    """Print portfolio in a consistent format."""
    logger.info(f"📊 {label}:")
    cash = portfolio.get("cash", 0)
    positions = portfolio.get("positions", {})
    logger.info(f"  Cash: ${cash:.2f}")

    total_positions_value = 0
    for symbol, position in positions.items():
        shares = position.get("shares", 0)
        value = position.get("value", 0)
        total_positions_value += value
        logger.info(f"  {symbol}: {shares} shares (${value:.2f})")

    total_value = cash + total_positions_value
    logger.info(f"  Total Value: ${total_value:.2f}")


def apply_trades_to_portfolio(original_portfolio: dict, trade_recommendations: list) -> dict:
    """Apply trade recommendations to create a new portfolio. Processes SELL orders first, then BUY orders."""
    import copy

    new_portfolio = copy.deepcopy(original_portfolio)
    cash = new_portfolio.get("cash", 0)
    positions = new_portfolio.get("positions", {})

    # Process SELL orders first
    for trade in trade_recommendations:
        if trade.action == "SELL" and trade.symbol and trade.shares and trade.price:
            symbol = trade.symbol
            if symbol in positions:
                shares_str = f"{trade.shares}" if trade.shares else "N/A"
                price_str = f"${trade.price:.2f}" if trade.price else "market price"
                logger.info(f"  {trade.action} {trade.symbol}: {shares_str} shares at {price_str}")

                # Add cash from sale
                cash += trade.shares * trade.price
                # Remove or reduce position
                current_shares = positions[symbol].get("shares", 0)
                if current_shares <= trade.shares:
                    del positions[symbol]  # Sell entire position
                else:
                    # Partial sell - just update shares and value
                    remaining_shares = current_shares - trade.shares
                    positions[symbol]["shares"] = remaining_shares
                    positions[symbol]["value"] = remaining_shares * trade.price

    # Process BUY orders second
    for trade in trade_recommendations:
        if trade.action == "BUY" and trade.symbol and trade.shares and trade.price:
            trade_cost = trade.shares * trade.price
            if cash >= trade_cost:
                shares_str = f"{trade.shares}" if trade.shares else "N/A"
                price_str = f"${trade.price:.2f}" if trade.price else "market price"
                logger.info(f"  {trade.action} {trade.symbol}: {shares_str} shares at {price_str}")

                cash -= trade_cost
                symbol = trade.symbol
                if symbol in positions:
                    # Add to existing position
                    positions[symbol]["shares"] += trade.shares
                    positions[symbol]["value"] = positions[symbol]["shares"] * trade.price
                else:
                    # New position
                    positions[symbol] = {"shares": trade.shares, "value": trade_cost}

    # Calculate totals and create complete portfolio
    total_positions_value = sum(pos.get("value", 0) for pos in positions.values())

    new_portfolio["cash"] = cash
    new_portfolio["positions"] = positions
    new_portfolio["total_value"] = cash + total_positions_value
    new_portfolio["positions_value"] = total_positions_value

    return new_portfolio


def save_portfolio(portfolio: dict):
    """Save portfolio to file."""
    import json

    with open(config.PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

    logger.info(f"✅ Portfolio saved to {config.PORTFOLIO_FILE}")


class TradeRecommendation(BaseModel):
    """Structured trade recommendation from the trading agent."""

    action: str  # "BUY", "SELL", or "HOLD"
    symbol: Optional[str] = None  # Stock symbol (if applicable)
    shares: Optional[float] = None  # Number of shares
    price: Optional[float] = None  # Target price
    reasoning: str  # Detailed reasoning for the recommendation
    confidence: Optional[str] = None  # "HIGH", "MEDIUM", "LOW"


class StockScore(BaseModel):
    """Individual stock scoring breakdown."""

    symbol: str
    composite_score: float
    momentum_score: float
    quality_score: float
    technical_score: float
    current_price: float


class TradingAnalysis(BaseModel):
    """Complete trading analysis with optional recommendations."""

    summary: str  # Overall market analysis summary
    trade_recommendations: List[TradeRecommendation] = []  # Empty list when scores-only mode
    market_outlook: str  # Bull/Bear/Neutral with reasoning
    risk_assessment: str  # Always provided - risk analysis is valuable regardless of mode
    current_holdings_scores: List[StockScore]  # Scores for current positions
    top_alternatives: List[StockScore]  # Top alternatives not held


@dataclass
class Position:
    symbol: str
    shares: float
    avg_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def total_return(self) -> float:
        return (self.current_price - self.avg_cost) / self.avg_cost * 100 if self.avg_cost > 0 else 0.0


class TradingAgent:
    """Trading agent implementation without LangGraph."""

    def __init__(self, cfg: config.Config = None):
        """Initialize the trading agent."""
        self.cfg = cfg or config.Config()
        self.messages: List[BaseMessage] = []
        self.portfolio_cash: float = 0.0
        self.portfolio_positions: Dict[str, Dict] = {}
        self.tool_call_count: int = 0
        self.trading_analysis: Optional[TradingAnalysis] = None

        # Configure LLMs with deterministic settings
        self.llm = ChatOpenAI(model=self.cfg.llm_model, temperature=self.cfg.llm_temperature, seed=self.cfg.llm_seed)

        # Always use TradingAnalysis (trade_recommendations will be empty in scores-only mode)
        self.structured_llm = self.llm.with_structured_output(TradingAnalysis)

        # Bind tools to LLM
        self.tools = [self.run_sql, self.search_market_news, self.retrieve_last_N_days_of_analysis]
        self.tool_map = {tool.__name__: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def run_sql(self, query: str) -> str:
        """
        Execute a SQL query against the NASDAQ stocks database.

        This function provides access to a comprehensive database of NASDAQ 100 stocks
        with 3 years of historical data including prices, fundamentals, and statistics.

        Args:
            query: SQL SELECT query to execute (INSERT/UPDATE/DELETE not allowed)

        Returns:
            JSON string with query results, including data, columns, and metadata

        Example:
            run_sql("SELECT symbol, name, sector FROM stocks LIMIT 10")
            run_sql("SELECT symbol, close FROM stock_prices WHERE date = '2024-01-15'")
        """
        import sqlite3
        from pathlib import Path

        db_path = config.DATABASE_PATH
        assert Path(db_path).exists(), f"Database file not found: {db_path}"
        assert query and query.strip(), "Empty query provided"

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            cursor.execute(query)

            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []

            # Fetch results
            rows = cursor.fetchall()

            # Convert rows to list of dictionaries
            data = []
            for row in rows:
                row_dict = {}
                for i, value in enumerate(row):
                    # Handle different data types
                    if value is None:
                        row_dict[columns[i]] = None
                    elif isinstance(value, (int, float, str)):
                        row_dict[columns[i]] = value
                    else:
                        row_dict[columns[i]] = str(value)
                data.append(row_dict)

            result = {
                "data": data,
                "columns": columns,
                "row_count": len(data),
                "query_executed": query,
            }

            return json.dumps(result, indent=2)

    def search_market_news(self, query: str) -> str:
        """Search for recent market news and analysis."""
        try:
            search = TavilySearchResults(max_results=3)
            results = search.invoke(query)
            return json.dumps({"success": True, "query": query, "results": results}, indent=2)
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                    "message": "Web search not available",
                }
            )

    def retrieve_last_N_days_of_analysis(self, symbol: str, days: int = 7) -> str:
        """Retrieve stock scores from the last N days of analysis to identify trends and patterns.

        Args:
            symbol: Stock ticker to filter by (e.g. 'AAPL', 'GOOGL')
            days: Number of days to look back (default 7)
        """
        try:
            memory_db = MemoryDatabase()
            data = memory_db.get_last_n_days(days, symbol)

            if not data:
                return json.dumps({"message": f"No analysis data found for {symbol} in the last {days} days", "symbol": symbol, "days_requested": days})

            return json.dumps({"symbol": symbol, "days_requested": days, "records_found": len(data), "analysis_data": data}, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "message": f"Failed to retrieve memory data for {symbol}"})

    def initialize_portfolio(self):
        """Initialize portfolio from file."""
        portfolio_file = config.PORTFOLIO_FILE
        assert os.path.exists(portfolio_file), f"Portfolio file not found: {portfolio_file}"

        # Load portfolio data
        with open(portfolio_file, "r") as f:
            portfolio = json.load(f)

        self.portfolio_cash = portfolio.get("cash", self.cfg.default_cash)
        self.portfolio_positions = portfolio.get("positions", {})

        # Add system message about portfolio and constraints
        system_msg = prompts.get_system_prompt(
            portfolio_cash=self.portfolio_cash,
            portfolio_positions=self.portfolio_positions,
            cfg=self.cfg,
        )

        self.messages.append(SystemMessage(content=system_msg))

    def execute_tool_calls(self, tool_calls):
        """Execute tool calls and return results."""
        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name in self.tool_map:
                try:
                    # Execute the tool
                    result = self.tool_map[tool_name](**tool_args)
                    tool_results.append(ToolMessage(content=result, tool_call_id=tool_id))
                except Exception as e:
                    error_msg = f"Error executing {tool_name}: {str(e)}"
                    tool_results.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
            else:
                error_msg = f"Unknown tool: {tool_name}"
                tool_results.append(ToolMessage(content=error_msg, tool_call_id=tool_id))

        return tool_results

    def run_analysis_loop(self):
        """Run the main analysis loop (equivalent to LangGraph's tool-calling loop)."""
        logger.info("🚀 Starting Trading Agent Analysis...")
        logger.info("=" * 80)

        while self.tool_call_count < self.cfg.max_tool_calls:
            # Get response from LLM
            response = self.llm_with_tools.invoke(self.messages)
            self.messages.append(response)

            # Check if there are tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                self.tool_call_count += 1
                logger.info("\n--- MARKET_ANALYSIS (Tool Call %d) ---", self.tool_call_count)
                logger.info(response.content if response.content else "Executing tools...")
                logger.info("-" * 40)

                # Execute tools
                tool_results = self.execute_tool_calls(response.tool_calls)
                self.messages.extend(tool_results)

                # Continue the loop for more analysis
                continue
            else:
                # No more tool calls, analysis is complete
                logger.info("\n--- MARKET_ANALYSIS ---")
                logger.info(response.content)
                logger.info("-" * 40)
                break

        if self.tool_call_count >= self.cfg.max_tool_calls:
            logger.info("Reached maximum tool calls limit (%d)", self.cfg.max_tool_calls)

    def generate_recommendations(self):
        """Generate structured recommendations."""
        # Collect analysis context from previous messages
        analysis_context = ""
        for msg in self.messages:
            if isinstance(msg, AIMessage) and msg.content:
                analysis_context += msg.content + "\n\n"

        # Get structured analysis prompt
        structured_prompt = prompts.get_trading_analysis_prompt(
            portfolio_cash=self.portfolio_cash,
            portfolio_positions=self.portfolio_positions,
            analysis_context=analysis_context[-2000:],  # Limit context length
            cfg=self.cfg,
        )

        # Get structured output from LLM
        trading_analysis = self.structured_llm.invoke([HumanMessage(content=structured_prompt)])

        # Store the structured analysis
        self.trading_analysis = trading_analysis

        # Create a human-readable summary message
        summary_text = f"""
🎯 TRADING ANALYSIS COMPLETE

📊 Market Analysis Summary:
{trading_analysis.summary}

🎯 Market Outlook: {trading_analysis.market_outlook}

⚠️ Risk Assessment: {trading_analysis.risk_assessment}

📋 Trade Recommendations ({len(trading_analysis.trade_recommendations)} recommendations):
"""

        for i, rec in enumerate(trading_analysis.trade_recommendations, 1):
            price_str = f"${rec.price:.2f}" if rec.price else "N/A"
            summary_text += f"""{i}. {rec.action} {rec.symbol or "N/A"}
- Shares: {rec.shares or "N/A"}
- Target Price: {price_str}
- Confidence: {rec.confidence or "N/A"}
- Reasoning: {rec.reasoning}

"""

        summary_text += "✅ Analysis session completed!"

        logger.info("\n--- RECOMMEND_TRADES ---")
        logger.info(summary_text)
        logger.info("-" * 40)

        self.messages.append(AIMessage(content=summary_text))

    def run(self) -> TradingAnalysis:
        """Run the complete trading analysis."""
        # Step 1: Initialize portfolio (equivalent to initialize_agent node)
        self.initialize_portfolio()

        # Step 2: Run analysis loop with tools (equivalent to market_analysis + tools nodes)
        self.run_analysis_loop()

        # Step 3: Generate structured recommendations (equivalent to recommend_trades node)
        self.generate_recommendations()

        logger.info("\n✅ Trading session completed successfully!")

        # Log final portfolio comparison
        logger.info("\n--- PORTFOLIO SUMMARY ---")
        current_portfolio = {"cash": self.portfolio_cash, "positions": self.portfolio_positions}
        print_portfolio(current_portfolio, "Current Portfolio")

        # Only log "after" portfolio if there are actual trade recommendations
        if self.trading_analysis.trade_recommendations:
            logger.info("📈 Applying Recommended Portfolio Changes:")

            # Apply trades and show resulting portfolio
            final_portfolio = apply_trades_to_portfolio(current_portfolio, self.trading_analysis.trade_recommendations)
            logger.info("")
            print_portfolio(final_portfolio, "Portfolio After Trades")

            # Execute trades if enabled
            if self.cfg.execute_trades:
                save_portfolio(final_portfolio)
            else:
                logger.info("📋 Trade execution disabled - portfolio file not updated")
        else:
            logger.info("📊 No trades recommended - maintain current portfolio")

        logger.info("-" * 40)

        # Save scores to memory
        memory_db = MemoryDatabase()
        today = datetime.now().strftime("%Y-%m-%d")
        memory_db.update_memory(today, self.trading_analysis.current_holdings_scores, self.trading_analysis.top_alternatives)

        # Return structured analysis
        assert self.trading_analysis, "Structured analysis must be present"
        return self.trading_analysis


def main(cfg: config.Config = None):
    """Main application entry point"""
    # Load environment variables
    load_dotenv()

    # Parse configuration with command line overrides (if not provided)
    if cfg is None:
        cfg = config.parse_config()

    # Setup logging
    config.setup_logging()

    # Print initial portfolio before running agent
    with open(config.PORTFOLIO_FILE, "r") as f:
        portfolio = json.load(f)
    print_portfolio(portfolio, "Initial Portfolio")

    # Create and run the agent
    agent = TradingAgent(cfg)
    trading_analysis = agent.run()

    return trading_analysis


if __name__ == "__main__":
    main()
