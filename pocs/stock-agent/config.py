"""
Configuration constants for the Stock Trading Agent
"""

import logging
from typing import Final

# Database
DATABASE_PATH: Final[str] = "nasdaq_stocks.db"
PORTFOLIO_FILE: Final[str] = "portfolio.json"

# Trading limits
MAX_TOOL_CALLS: Final[int] = 10
MAX_POSITIONS: Final[int] = 10
DEFAULT_CASH: Final[float] = 1000.0

# Model settings
LLM_MODEL: Final[str] = "gpt-5"
# LLM_TEMPERATURE: Final[float] = 0.1 # Original value
LLM_TEMPERATURE: Final[float] = 0.0  # Maximum determinism
LLM_SEED: Final[int] = 12345  # Fixed seed for reproducible results

# Logging
LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE: Final[str] = "log.txt"

# Email scopes
SCOPES = ["Mail.Read", "Mail.Send"]


def setup_logging():
    """Configure logging to both console and file."""
    # Clear any existing handlers
    logging.getLogger().handlers.clear()

    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger
