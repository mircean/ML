#!/usr/bin/env python3
"""
LangGraph Trading Agent

An AI trading agent built with LangGraph that thinks after each tool call.
Uses NASDAQ database and web search to make trading decisions.
"""

import json
import os
from dataclasses import dataclass
from typing import Annotated, Dict, List

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

load_dotenv()

import config
import prompts

# Setup logging
config.setup_logging()

# Configure the LLM
llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)


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
        return (
            (self.current_price - self.avg_cost) / self.avg_cost * 100
            if self.avg_cost > 0
            else 0.0
        )


@dataclass
class TradeAction:
    action: str  # "BUY" or "SELL"
    symbol: str
    shares: float
    price: float
    reasoning: str


# Define the graph state
class TradingState(Dict):
    messages: Annotated[List[BaseMessage], add_messages]
    portfolio_cash: float
    portfolio_positions: Dict[str, Dict]  # Serializable position data
    trade_actions: List[Dict]  # Serializable trade actions
    analysis_complete: bool
    trading_complete: bool
    tool_call_count: int


# Portfolio management functions
def load_portfolio_data():
    """Load the current portfolio from storage."""
    portfolio_file = config.PORTFOLIO_FILE

    if os.path.exists(portfolio_file):
        with open(portfolio_file, "r") as f:
            data = json.load(f)
            return {
                "success": True,
                "cash": data.get("cash", config.DEFAULT_CASH),
                "positions": data.get("positions", {}),
                "message": f"Portfolio loaded: ${data.get('cash', config.DEFAULT_CASH):.2f} cash, {len(data.get('positions', {}))} positions",
            }
    else:
        return {
            "success": True,
            "cash": config.DEFAULT_CASH,
            "positions": {},
            "message": f"New portfolio initialized with ${config.DEFAULT_CASH}",
        }


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
        columns = (
            [description[0] for description in cursor.description]
            if cursor.description
            else []
        )

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
        return json.dumps(
            {"success": True, "query": query, "results": results}, indent=2
        )
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
    # Load portfolio data
    portfolio_data = load_portfolio_data()

    state["portfolio_cash"] = portfolio_data.get("cash", 1000.0)
    state["portfolio_positions"] = portfolio_data.get("positions", {})
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


def market_analysis_node(state: TradingState) -> TradingState:
    """Analyze market data and make trading decisions."""
    # The LLM will respond with tool calls, which will be handled by the tool node
    response = llm_with_tools.invoke(state["messages"])
    state["messages"].append(response)

    return state


def thinking_node(state: TradingState) -> TradingState:
    """Let the agent think about the results."""
    thinking_prompt = prompts.get_thinking_prompt()

    state["messages"].append(HumanMessage(content=thinking_prompt))
    response = llm.invoke(state["messages"])
    state["messages"].append(response)

    return state


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


def recommend_trades_node(state: TradingState) -> TradingState:
    """Provide final summary of recommendations."""
    # Find the last AI message that contains recommendations
    recommendations_text = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            recommendations_text = msg.content
            break

    summary = prompts.get_summary_prompt(
        portfolio_cash=state["portfolio_cash"],
        portfolio_positions=state["portfolio_positions"],
        recommendations_text=recommendations_text,
    )

    state["messages"].append(AIMessage(content=summary))

    return state


# Tool setup
tools = [run_sql, search_market_news]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

thinking_enabled = False


# Define routing logic
def should_continue(state: TradingState):
    last_message = state["messages"][-1]

    # If the last message has tool calls, and we haven't reached the tool call limit, go to tools
    if (
        hasattr(last_message, "tool_calls")
        and last_message.tool_calls
        and state["tool_call_count"] < config.MAX_TOOL_CALLS
    ):
        print(f"Tool call count: {state['tool_call_count']}")
        return "tools"
    # Otherwise go to recommend_trades (analysis is complete)
    return "recommend_trades"


# Build the graph
workflow = StateGraph(TradingState)

# Add nodes
workflow.add_node("initialize_agent", initialize_agent_node)
workflow.add_node("market_analysis", market_analysis_node)
workflow.add_node("thinking", thinking_node)
workflow.add_node("recommend_trades", recommend_trades_node)
workflow.add_node("tools", tools_node_wrapper)

# Add edges
workflow.set_entry_point("initialize_agent")
workflow.add_edge("initialize_agent", "market_analysis")
workflow.add_conditional_edges("market_analysis", should_continue)
if thinking_enabled:
    workflow.add_edge("tools", "thinking")
    workflow.add_edge("thinking", "market_analysis")
else:
    workflow.add_edge("tools", "market_analysis")
workflow.add_edge("recommend_trades", END)


# Compile the graph
app = workflow.compile()


def run_trading_agent():
    """Run the LangGraph trading agent."""
    print("🚀 Starting LangGraph Trading Agent...")
    print("=" * 80)

    # Initialize state
    initial_state = TradingState(
        messages=[],
        portfolio_cash=config.DEFAULT_CASH,
        portfolio_positions={},
        trade_actions=[],
        analysis_complete=False,
        trading_complete=False,
        tool_call_count=0,
    )

    try:
        # Run the graph
        for step in app.stream(initial_state, config={"recursion_limit": 50}):
            for node_name, state in step.items():
                if node_name != "tools":  # Don't print tool outputs directly
                    print(f"\n--- {node_name.upper()} ---")
                    if state["messages"]:
                        last_msg = state["messages"][-1]
                        print(last_msg.content)
                    print("-" * 40)

        print("\n✅ Trading session completed successfully!")

    except Exception as e:
        print(f"❌ Error running trading agent: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_trading_agent()
