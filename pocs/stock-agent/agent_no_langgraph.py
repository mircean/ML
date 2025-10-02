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
from typing import Dict, List, Optional

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
    ToolMessage,
)
from langchain_openai import ChatOpenAI
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

    def __init__(self):
        """Initialize the trading agent."""
        self.messages: List[BaseMessage] = []
        self.portfolio_cash: float = 0.0
        self.portfolio_positions: Dict[str, Dict] = {}
        self.tool_call_count: int = 0
        self.structured_analysis: Optional[TradingAnalysis] = None

        # Configure LLMs
        self.llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
        self.structured_llm = self.llm.with_structured_output(TradingAnalysis)

        # Bind tools to LLM
        self.tools = [self.run_sql, self.search_market_news]
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

    def initialize_portfolio(self):
        """Initialize portfolio from file."""
        portfolio_file = config.PORTFOLIO_FILE
        assert os.path.exists(portfolio_file), f"Portfolio file not found: {portfolio_file}"

        # Load portfolio data
        with open(portfolio_file, "r") as f:
            portfolio = json.load(f)

        self.portfolio_cash = portfolio.get("cash", config.DEFAULT_CASH)
        self.portfolio_positions = portfolio.get("positions", {})

        # Add system message about portfolio and constraints
        system_msg = prompts.get_system_prompt(
            portfolio_cash=self.portfolio_cash,
            portfolio_positions=self.portfolio_positions,
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

        while self.tool_call_count < config.MAX_TOOL_CALLS:
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

        if self.tool_call_count >= config.MAX_TOOL_CALLS:
            logger.info("Reached maximum tool calls limit (%d)", config.MAX_TOOL_CALLS)

    def generate_recommendations(self):
        """Generate structured recommendations."""
        # Collect analysis context from previous messages
        analysis_context = ""
        for msg in self.messages:
            if isinstance(msg, AIMessage) and msg.content:
                analysis_context += msg.content + "\n\n"

        # Get structured analysis prompt
        structured_prompt = prompts.get_structured_analysis_prompt(
            portfolio_cash=self.portfolio_cash,
            portfolio_positions=self.portfolio_positions,
            analysis_context=analysis_context[-2000:],  # Limit context length
        )

        # Get structured output from LLM
        structured_analysis = self.structured_llm.invoke([HumanMessage(content=structured_prompt)])

        # Store the structured analysis
        self.structured_analysis = structured_analysis

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

        # Return structured analysis
        assert self.structured_analysis, "Structured analysis must be present"
        return self.structured_analysis


def main():
    """Main application entry point"""
    # Load environment variables
    load_dotenv()

    # Setup logging
    config.setup_logging()

    # Create and run the agent
    agent = TradingAgent()
    structured_analysis = agent.run()

    return structured_analysis


if __name__ == "__main__":
    main()