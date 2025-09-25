"""
Configuration settings for Outlook email client.
"""

import logging

# Email scopes
SCOPES = ["Mail.Read", "Mail.Send"]

# Message retrieval settings
MESSAGE_LIMIT = 10

# Logging configuration
DEFAULT_LOG_LEVEL = "INFO"


def setup_logging(log_level: str = DEFAULT_LOG_LEVEL):
    """Setup logging configuration."""
    logging.basicConfig(level=getattr(logging, log_level.upper()), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
