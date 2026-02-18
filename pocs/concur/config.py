"""
Configuration for the Concur Expense Receipt Tracker.
"""

import argparse
import logging
from dataclasses import dataclass, fields

# Email scopes
SCOPES = ["Mail.Read", "Mail.Send"]

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Config:
    """Runtime configuration overridable via command line."""

    dry_run: bool = False
    days_to_search: int = 90


def parse_config() -> Config:
    """Parse command line arguments and return Config."""
    parser = argparse.ArgumentParser(description="Concur Expense Receipt Tracker")
    default_config = Config()

    for field in fields(Config):
        cli_arg = "--" + field.name.replace("_", "-")
        default_value = getattr(default_config, field.name)

        if field.type is bool:
            if default_value:
                parser.add_argument(
                    f"--no-{field.name.replace('_', '-')}",
                    dest=field.name,
                    action="store_false",
                    help=f"Disable {field.name} (default: {default_value})",
                )
            else:
                parser.add_argument(
                    cli_arg,
                    dest=field.name,
                    action="store_true",
                    help=f"Enable {field.name} (default: {default_value})",
                )
        else:
            parser.add_argument(
                cli_arg,
                dest=field.name,
                type=field.type,
                help=f"{field.name} (default: {default_value})",
            )

    args = parser.parse_args()

    cfg = Config()
    for field in fields(Config):
        arg_value = getattr(args, field.name, None)
        if arg_value is not None:
            setattr(cfg, field.name, arg_value)

    return cfg


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress verbose Azure SDK logging
    logging.getLogger("azure.ai.formrecognizer").setLevel(logging.WARNING)
    logging.getLogger("azure.core").setLevel(logging.WARNING)
