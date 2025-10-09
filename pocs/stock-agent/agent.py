#!/usr/bin/env python3
"""
LangGraph Trading Agent

An AI trading agent built with LangGraph that thinks after each tool call.
Uses NASDAQ database and web search to make trading decisions.
"""

import copy
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Optional

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
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
    trade_recommendations: List[TradeRecommendation] = []  # Trade recommendations from agent
    market_outlook: str  # Bull/Bear/Neutral with reasoning
    risk_assessment: str  # Risk factors identified
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


# Define the graph state
class TradingState(Dict):
    messages: Annotated[List[BaseMessage], add_messages]
    portfolio_cash: float
    portfolio_positions: Dict[str, Dict]  # Serializable position data
    trade_actions: List[Dict]  # Serializable trade actions
    analysis_complete: bool
    trading_complete: bool
    tool_call_count: int
    trading_analysis: Optional[TradingAnalysis] = None  # Structured output from LLM


@tool
def run_sql(query: str) -> str:
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


@tool
def search_market_news(query: str) -> str:
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


@tool
def retrieve_last_N_days_of_analysis(symbol: str, days: int = 7) -> str:
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


# Helper functions
def print_portfolio(portfolio: dict, label: str = "Portfolio"):
    """Build portfolio string in a consistent format."""
    cash = portfolio.get("cash", 0)
    positions = portfolio.get("positions", {})

    text = f"📊 {label}:\n"
    text += f"  Cash: ${cash:.2f}\n"

    # Sort positions by value (descending)
    sorted_positions = sorted(positions.items(), key=lambda x: x[1].get("value", 0), reverse=True)

    total_positions_value = 0
    for symbol, position in sorted_positions:
        shares = position.get("shares", 0)
        value = position.get("value", 0)
        total_positions_value += value
        text += f"  {symbol}: {shares} shares (${value:.2f})\n"

    total_value = cash + total_positions_value
    text += f"  Total Value: ${total_value:.2f}"

    return text


def apply_trades_to_portfolio(original_portfolio: dict, trade_recommendations: list) -> dict:
    """Apply trade recommendations to create a new portfolio. Processes SELL orders first, then BUY orders."""
    logger.info("📈 Applying Recommended Portfolio Changes:")

    new_portfolio = copy.deepcopy(original_portfolio)
    cash = new_portfolio.get("cash", 0)
    positions = new_portfolio.get("positions", {})

    trades_recommended = 0

    # Process SELL orders first
    for trade in trade_recommendations:
        if trade.action == "SELL" and trade.symbol and trade.shares and trade.price:
            symbol = trade.symbol
            if symbol in positions:
                trades_recommended += 1
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
                trades_recommended += 1
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

    if trades_recommended > 0:
        logger.info(f"✅ {trades_recommended} trades recommended and simulated")
        # Create portfolio with just cash and positions
        new_portfolio["cash"] = cash
        new_portfolio["positions"] = positions
        return new_portfolio

    logger.info("ℹ️ No valid trades to execute")
    return None


def save_portfolio(portfolio: dict):
    """Save portfolio to file."""
    import json

    with open(config.PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

    logger.info(f"✅ Portfolio saved to {config.PORTFOLIO_FILE}")


# Define the agent nodes
def initialize_agent_node(state: TradingState, cfg: config.Config) -> TradingState:
    """Initialize agent with portfolio and system message."""
    portfolio_file = config.PORTFOLIO_FILE
    assert os.path.exists(portfolio_file), f"Portfolio file not found: {portfolio_file}"
    # Load portfolio data
    with open(portfolio_file, "r") as f:
        portfolio = json.load(f)

    state["portfolio_cash"] = portfolio.get("cash", cfg.default_cash)
    state["portfolio_positions"] = portfolio.get("positions", {})
    state["trade_actions"] = []
    state["analysis_complete"] = False
    state["trading_complete"] = False
    state["tool_call_count"] = 0

    # Add system message about portfolio and constraints
    system_msg = prompts.get_system_prompt(
        portfolio_cash=state["portfolio_cash"],
        portfolio_positions=state["portfolio_positions"],
        cfg=cfg,
    )

    state["messages"].append(SystemMessage(content=system_msg))

    return state


def create_market_analysis_node(llm_with_tools, cfg: config.Config):
    """Create the market analysis node with the configured LLM."""

    def market_analysis_node(state: TradingState) -> TradingState:
        """Analyze market data and make trading decisions."""
        # The LLM will respond with tool calls, which will be handled by the tool node
        response = llm_with_tools.invoke(state["messages"])
        logger.info(f"Market analysis node reasoning: {response.content}")
        state["messages"].append(response)
        return state

    return market_analysis_node


def create_tools_node_wrapper(tool_node, cfg: config.Config):
    """Create the tools node wrapper with the configured tool node."""

    def tools_node_wrapper(state: TradingState) -> TradingState:
        """Execute tools and increment the tool call counter."""
        # Count how many tool calls we're about to make, or how many iterations so far
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            state["tool_call_count"] += 1  # len(last_message.tool_calls)

        # Execute the tools
        result = tool_node.invoke(state)

        # Ensure the counter is preserved in the result
        result["tool_call_count"] = state["tool_call_count"]
        return result

    return tools_node_wrapper


def print_analysis(trading_analysis):
    text = f"""
🎯 TRADING ANALYSIS COMPLETE
📊 Market Analysis Summary: {trading_analysis.summary}
🎯 Market Outlook: {trading_analysis.market_outlook}"""

    # Always add risk assessment
    text += f"\n⚠️ Risk Assessment: {trading_analysis.risk_assessment}"

    # Add trade recommendations if provided (full mode)
    if trading_analysis.trade_recommendations:
        text += "\n📋 Trade Recommendations\n"
        for rec in trading_analysis.trade_recommendations:
            text += f"{str(rec)}\n"

    # Always show scores
    text += "📋 Current Holdings Scores:\n"
    for rec in trading_analysis.current_holdings_scores:
        text += f"{rec.symbol}: {rec.composite_score}\n"
    text += "📋 Top Alternatives:\n"
    for rec in trading_analysis.top_alternatives:
        text += f"{rec.symbol}: {rec.composite_score}\n"

    text += "✅ Analysis session completed!"
    return text


def create_analysis_output_node(structured_llm, cfg: config.Config):
    """Create the analysis output node with the configured structured LLM."""

    def analysis_output_node(state: TradingState) -> TradingState:
        """Provide final structured analysis output (scores only or with recommendations)."""
        # Collect analysis context from previous messages
        analysis_context = ""
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                analysis_context += msg.content + "\n\n"

        # Get structured analysis prompt
        structured_prompt = prompts.get_trading_analysis_prompt(
            portfolio_cash=state["portfolio_cash"],
            portfolio_positions=state["portfolio_positions"],
            analysis_context=analysis_context[-2000:],  # Limit context length
            cfg=cfg,
        )

        # Get structured output from LLM
        trading_analysis = structured_llm.invoke([HumanMessage(content=structured_prompt)])

        # Store the structured analysis in state for later use
        state["trading_analysis"] = trading_analysis

        analysis_text = print_analysis(trading_analysis)
        # Create a human-readable summary message

        state["messages"].append(AIMessage(content=analysis_text))
        return state

    return analysis_output_node


# Define routing logic
def should_continue(state: TradingState, cfg: config.Config):
    last_message = state["messages"][-1]

    # If the last message has tool calls, and we haven't reached the tool call limit, go to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls and state["tool_call_count"] < cfg.max_tool_calls:
        logger.info(f"""Tool call {[tool["name"] for tool in last_message.tool_calls]} tool call count: {state["tool_call_count"]}""")
        return "tools"
    # Otherwise go to analysis_output (analysis is complete)
    return "analysis_output"


def main(cfg: config.Config = None):
    """Main application entry point"""
    # Load environment variables
    load_dotenv()

    # Parse configuration with command line overrides (if not provided)
    if cfg is None:
        cfg = config.parse_config()

    # Setup logging
    config.setup_logging()

    # Configure the LLM with deterministic settings
    llm = ChatOpenAI(model=cfg.llm_model, temperature=cfg.llm_temperature, seed=cfg.llm_seed)

    # Always use TradingAnalysis (trade_recommendations will be empty in scores-only mode)
    structured_llm = llm.with_structured_output(TradingAnalysis)

    # Tool setup
    tools = [run_sql, search_market_news, retrieve_last_N_days_of_analysis]
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    # Build the graph
    workflow = StateGraph(TradingState)

    # Create node functions with dependencies
    market_analysis_node = create_market_analysis_node(llm_with_tools, cfg)
    tools_node_wrapper = create_tools_node_wrapper(tool_node, cfg)
    analysis_output_node = create_analysis_output_node(structured_llm, cfg)

    # Add nodes
    workflow.add_node("initialize_agent", lambda state: initialize_agent_node(state, cfg))
    workflow.add_node("market_analysis", market_analysis_node)
    workflow.add_node("analysis_output", analysis_output_node)
    workflow.add_node("tools", tools_node_wrapper)

    # Add edges
    workflow.set_entry_point("initialize_agent")
    workflow.add_edge("initialize_agent", "market_analysis")
    workflow.add_conditional_edges("market_analysis", lambda state: should_continue(state, cfg))
    workflow.add_edge("tools", "market_analysis")
    workflow.add_edge("analysis_output", END)

    # Compile the graph
    app = workflow.compile()

    # Run the LangGraph trading agent
    logger.info("🚀 Starting LangGraph Trading Agent...")
    logger.info("=" * 80)

    # Initialize state
    initial_state = TradingState(
        messages=[],
        portfolio_cash=cfg.default_cash,
        portfolio_positions={},
        trade_actions=[],
        analysis_complete=False,
        trading_complete=False,
        tool_call_count=0,
        trading_analysis=None,
    )

    # Print initial portfolio
    with open(config.PORTFOLIO_FILE, "r") as f:
        portfolio = json.load(f)
    logger.info(print_portfolio(portfolio, "Initial Portfolio"))

    final_state = None

    # Run the graph
    for step in app.stream(initial_state, config={"recursion_limit": 50}):
        for node_name, state in step.items():
            final_state = state  # Keep track of final state
            if node_name != "tools":  # Don't print tool outputs directly
                logger.info(f"\n--- {node_name.upper()} ---")
                if state["messages"]:
                    last_msg = state["messages"][-1]
                    logger.debug(last_msg.content)

    logger.info("\n✅ Trading session completed successfully!")

    # Return structured output directly
    assert final_state, "Final state must exist"
    trading_analysis = final_state["trading_analysis"]
    assert trading_analysis, "Structured analysis must be present in final state"
    logger.info(print_analysis(trading_analysis))

    # Log final portfolio comparison
    logger.info(print_portfolio(portfolio, "Current Portfolio"))

    # Only log "after" portfolio if there are actual trade recommendations
    # there are always trade recommendations
    assert trading_analysis.trade_recommendations, "Trade recommendations must be present in final state"

    # Apply trades and show resulting portfolio
    final_portfolio = apply_trades_to_portfolio(portfolio, trading_analysis.trade_recommendations)
    if final_portfolio:
        logger.info(print_portfolio(final_portfolio, "Portfolio After Trades"))
        if cfg.execute_trades:
            save_portfolio(final_portfolio)
        else:
            logger.info("📋 Trade execution disabled - portfolio file not updated")

    # Save scores to memory
    memory_db = MemoryDatabase()
    today = datetime.now().strftime("%Y-%m-%d")
    memory_db.update_memory(today, trading_analysis.current_holdings_scores, trading_analysis.top_alternatives)

    return trading_analysis, final_portfolio


if __name__ == "__main__":
    main()
