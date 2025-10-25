"""
Test script for Outlook client functionality.
"""

import logging
import os

import config
from dotenv import load_dotenv
from outlook_auth import OutlookAuthenticator
from outlook_client import OutlookClient

logger = logging.getLogger(__name__)


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
        "has_attachments": message.get("hasAttachments", False),
    }


def test_read(client):
    """Test reading messages from inbox."""
    select_fields = ["subject", "from", "receivedDateTime", "isRead", "hasAttachments", "bodyPreview"]

    message_limit = config.MESSAGE_LIMIT
    messages = client.get_messages(limit=message_limit, select_fields=select_fields)

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
            attachments = client.get_message_attachments(message["id"], select_fields=["name", "contentType", "size"])

            if attachments:
                logger.info("Attachments:")
                for attachment in attachments:
                    name = attachment.get("name")
                    content_type = attachment.get("contentType")
                    size = attachment.get("size", 0)
                    logger.info(f"  - {name} ({content_type}, {size} bytes)")

        logger.info("-" * 80)


def test_send(client):
    """Test sending an email."""
    client.send_email(
        to="mircean@outlook.com",
        subject="Test Email",
        body="Hello from Python!",
        body_type="Text",
    )


def main():
    """Main application entry point."""
    load_dotenv()
    config.setup_logging()

    # Check required environment variables
    tenant_id = os.getenv("GRAPH_TENANT_ID")
    client_id = os.getenv("GRAPH_CLIENT_ID")
    assert tenant_id and client_id, "Missing required environment variables: GRAPH_TENANT_ID, GRAPH_CLIENT_ID"

    logger.info("Starting Outlook test")

    authenticator = OutlookAuthenticator(tenant_id=tenant_id, client_id=client_id, scopes=config.SCOPES)
    client = OutlookClient(authenticator)

    test_read(client)
    # test_send(client)


if __name__ == "__main__":
    main()
