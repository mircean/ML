#!/usr/bin/env python3
"""
Refactored email reader using separated auth and client components.
"""

import logging
import os
import sys

from dotenv import load_dotenv
from outlook_auth import OutlookAuthenticator
from outlook_client import OutlookClient

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_config():
    """Load configuration from environment variables."""
    load_dotenv()

    required_vars = ["GRAPH_TENANT_ID", "GRAPH_CLIENT_ID"]

    config = {}
    missing_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        config[var.lower()] = value

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    config["scopes"] = ["Mail.Read"]
    config["message_limit"] = int(os.getenv("MESSAGE_LIMIT", "10"))
    config["log_level"] = os.getenv("LOG_LEVEL", "INFO")
    config["search_term"] = os.getenv("SEARCH_TERM", "")  # Optional search term

    return config


def format_message_info(message):
    """Format message information for display."""
    sender = message.get("from", {}).get("emailAddress", {})
    sender_name = sender.get("name", "Unknown")
    sender_email = sender.get("address", "unknown@example.com")

    flag = "Unread" if not message.get("isRead") else "Read"
    preview = (message.get("bodyPreview") or "").replace("\r", " ").replace("\n", " ")[:100]

    return {
        "flag": flag,
        "date": message.get("receivedDateTime"),
        "sender_name": sender_name,
        "sender_email": sender_email,
        "subject": message.get("subject"),
        "preview": preview,
        "has_attachments": message.get("hasAttachments", False)
    }


def main():
    """Main application entry point."""
    config = load_config()
    setup_logging(config["log_level"])

    logger.info("Starting email reader")

    try:
        authenticator = OutlookAuthenticator(
            tenant_id=config["graph_tenant_id"],
            client_id=config["graph_client_id"],
            scopes=config["scopes"]
        )

        client = OutlookClient(authenticator)

        select_fields = [
            "subject", "from", "receivedDateTime", "isRead",
            "hasAttachments", "bodyPreview"
        ]

        # Use search if search_term is provided, otherwise get inbox messages
        if config["search_term"]:
            logger.info(f"Searching for: '{config['search_term']}'")
            messages = client.search_messages(
                search_term=config["search_term"],
                limit=config["message_limit"],
                select_fields=select_fields
            )
        else:
            messages = client.get_messages(
                limit=config["message_limit"],
                select_fields=select_fields
            )

        if not messages:
            logger.info("No messages found")
            return

        logger.info(f"Processing {len(messages)} messages:")
        logger.info("=" * 80)

        for message in messages:
            info = format_message_info(message)

            logger.info(f"[{info['flag']:6}] {info['date']} | {info['sender_name']} <{info['sender_email']}>")
            logger.info(f"Subject: {info['subject']}")
            logger.info(f"Preview: {info['preview']!r}")

            if info["has_attachments"]:
                attachments = client.get_message_attachments(
                    message["id"],
                    select_fields=["name", "contentType", "size"]
                )

                if attachments:
                    logger.info("Attachments:")
                    for attachment in attachments:
                        name = attachment.get("name")
                        content_type = attachment.get("contentType")
                        size = attachment.get("size", 0)
                        logger.info(f"  - {name} ({content_type}, {size} bytes)")

            logger.info("-" * 80)

    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


