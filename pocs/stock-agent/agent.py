#!/usr/bin/env python3
"""
LangGraph Trading Agent

An AI trading agent built with LangGraph that thinks after each tool call.
Uses NASDAQ database and web search to make trading decisions.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Annotated, Dict, List, Optional

import config
import prompts

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


class TradingAnalysis(BaseModel):
    """Complete trading analysis with recommendations."""

    summary: str  # Overall market analysis summary
    trade_recommendations: List[TradeRecommendation]
    market_outlook: str  # Bull/Bear/Neutral with reasoning
    risk_assessment: str  # Risk factors identified
    # Configure structured output LLM for final recommendations


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
    structured_analysis: Optional[TradingAnalysis] = None  # Structured output from LLM


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


# Define the agent nodes
def initialize_agent_node(state: TradingState) -> TradingState:
    """Initialize agent with portfolio and system message."""
    portfolio_file = config.PORTFOLIO_FILE
    assert os.path.exists(portfolio_file), f"Portfolio file not found: {portfolio_file}"
    # Load portfolio data
    with open(portfolio_file, "r") as f:
        portfolio = json.load(f)

    state["portfolio_cash"] = portfolio.get("cash", config.DEFAULT_CASH)
    state["portfolio_positions"] = portfolio.get("positions", {})
    state["trade_actions"] = []
    state["analysis_complete"] = False
    state["trading_complete"] = False
    state["tool_call_count"] = 0

    # Add system message about portfolio and constraints
    system_msg = prompts.get_system_prompt(
        portfolio_cash=state["portfolio_cash"],
        portfolio_positions=state["portfolio_positions"],
    )

    state["messages"].append(SystemMessage(content=system_msg))

    return state


def create_market_analysis_node(llm_with_tools):
    """Create the market analysis node with the configured LLM."""

    def market_analysis_node(state: TradingState) -> TradingState:
        """Analyze market data and make trading decisions."""
        # The LLM will respond with tool calls, which will be handled by the tool node
        response = llm_with_tools.invoke(state["messages"])
        state["messages"].append(response)
        return state

    return market_analysis_node


def create_tools_node_wrapper(tool_node):
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


def create_recommend_trades_node(structured_llm):
    """Create the recommend trades node with the configured structured LLM."""

    def recommend_trades_node(state: TradingState) -> TradingState:
        """Provide final structured recommendations."""
        # Collect analysis context from previous messages
        analysis_context = ""
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                analysis_context += msg.content + "\n\n"

        # Get structured analysis prompt
        structured_prompt = prompts.get_structured_analysis_prompt(
            portfolio_cash=state["portfolio_cash"],
            portfolio_positions=state["portfolio_positions"],
            analysis_context=analysis_context[-2000:],  # Limit context length
        )

        # Get structured output from LLM
        structured_analysis = structured_llm.invoke([HumanMessage(content=structured_prompt)])

        # Store the structured analysis in state for later use
        state["structured_analysis"] = structured_analysis

        # Create a human-readable summary message
        summary_text = f"""
🎯 TRADING ANALYSIS COMPLETE

📊 Market Analysis Summary:
{structured_analysis.summary}

🎯 Market Outlook: {structured_analysis.market_outlook}

⚠️ Risk Assessment: {structured_analysis.risk_assessment}

📋 Trade Recommendations ({len(structured_analysis.trade_recommendations)} recommendations):
"""

        for i, rec in enumerate(structured_analysis.trade_recommendations, 1):
            price_str = f"${rec.price:.2f}" if rec.price else "N/A"
            summary_text += f"""{i}. {rec.action} {rec.symbol or "N/A"}
- Shares: {rec.shares or "N/A"}
- Target Price: {price_str}
- Confidence: {rec.confidence or "N/A"}
- Reasoning: {rec.reasoning}

"""

        summary_text += "✅ Analysis session completed!"
        state["messages"].append(AIMessage(content=summary_text))
        return state

    return recommend_trades_node


# Define routing logic
def should_continue(state: TradingState):
    last_message = state["messages"][-1]

    # If the last message has tool calls, and we haven't reached the tool call limit, go to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls and state["tool_call_count"] < config.MAX_TOOL_CALLS:
        logger.info(f"Tool call count: {state['tool_call_count']}")
        return "tools"
    # Otherwise go to recommend_trades (analysis is complete)
    return "recommend_trades"


def main():
    """Main application entry point"""
    # Load environment variables
    load_dotenv()

    # Setup logging
    config.setup_logging()

    # Configure the LLM
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    structured_llm = llm.with_structured_output(TradingAnalysis)

    # Tool setup
    tools = [run_sql, search_market_news]
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    # Build the graph
    workflow = StateGraph(TradingState)

    # Create node functions with dependencies
    market_analysis_node = create_market_analysis_node(llm_with_tools)
    tools_node_wrapper = create_tools_node_wrapper(tool_node)
    recommend_trades_node = create_recommend_trades_node(structured_llm)

    # Add nodes
    workflow.add_node("initialize_agent", initialize_agent_node)
    workflow.add_node("market_analysis", market_analysis_node)
    workflow.add_node("recommend_trades", recommend_trades_node)
    workflow.add_node("tools", tools_node_wrapper)

    # Add edges
    workflow.set_entry_point("initialize_agent")
    workflow.add_edge("initialize_agent", "market_analysis")
    workflow.add_conditional_edges("market_analysis", should_continue)
    workflow.add_edge("tools", "market_analysis")
    workflow.add_edge("recommend_trades", END)

    # Compile the graph
    app = workflow.compile()

    # Run the LangGraph trading agent
    logger.info("🚀 Starting LangGraph Trading Agent...")
    logger.info("=" * 80)

    # Initialize state
    initial_state = TradingState(
        messages=[],
        portfolio_cash=config.DEFAULT_CASH,
        portfolio_positions={},
        trade_actions=[],
        analysis_complete=False,
        trading_complete=False,
        tool_call_count=0,
        structured_analysis=None,
    )

    final_state = None

    # Run the graph
    for step in app.stream(initial_state, config={"recursion_limit": 50}):
        for node_name, state in step.items():
            final_state = state  # Keep track of final state
            if node_name != "tools":  # Don't print tool outputs directly
                logger.info(f"\n--- {node_name.upper()} ---")
                if state["messages"]:
                    last_msg = state["messages"][-1]
                    logger.info(last_msg.content)
                logger.info("-" * 40)

    logger.info("\n✅ Trading session completed successfully!")

    # Return structured output directly
    assert final_state, "Final state must exist"
    structured_analysis = final_state["structured_analysis"]
    assert structured_analysis, "Structured analysis must be present in final state"
    return structured_analysis


if __name__ == "__main__":
    main()
