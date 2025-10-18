#!/usr/bin/env python3
"""
Refactored email reader using separated auth and client components.
"""

import datetime
import logging
import os
import sys

import config
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
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
    select_fields = ["subject", "from", "receivedDateTime", "isRead", "hasAttachments", "bodyPreview"]

    message_limit = int(os.getenv("MESSAGE_LIMIT", str(config.MESSAGE_LIMIT)))
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


def parse_pdf_with_azure(pdf_content: bytes, filename: str) -> dict:
    """Parse PDF using Azure Form Recognizer."""
    try:
        endpoint = os.getenv("AZURE_ENDPOINT")
        key = os.getenv("AZURE_API_KEY")

        assert endpoint and key, "Missing Azure Form Recognizer credentials: AZURE_ENDPOINT, AZURE_API_KEY"

        client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

        logger.info(f"Analyzing {filename} with Azure Form Recognizer...")

        # Use prebuilt invoice model for invoices
        poller = client.begin_analyze_document("prebuilt-invoice", pdf_content)
        result = poller.result()

        assert len(result.documents) == 1, f"Expected 1 document, got {len(result.documents)}"
        document = result.documents[0]
        return {"total": document.fields.get("InvoiceTotal").value.amount, "date": document.fields.get("InvoiceDate").value}
        # TODO: can retrieve ticker number and passenger name?

    except Exception as e:
        logger.error(f"Failed to parse PDF {filename} with Azure: {e}")
        raise e


def test_send(client):
    client.send_email(
        to="mircean@outlook.com",
        subject="Test Email",
        body="Hello from Python!",
        body_type="Text",  # or "HTML"
    )


def test_search(client, total, date):
    """Search emails containing 'invoice' with attachments from last 6 months."""
    logger.info("Starting invoice search")

    # Search for emails with attachments from Amex travel
    search_results = client.search_messages(
        query="hasattachments:true received>=2024-03-01 from:donotreply@mytrips.amexgbt.com",
        limit=50,
        select_fields=["subject", "from", "receivedDateTime", "hasAttachments", "bodyPreview"],
    )

    if not search_results:
        logger.info("No invoice emails with attachments found")
        return

    logger.info(f"Found {len(search_results)} invoice emails with attachments:")
    logger.info("=" * 80)

    found = False
    for search_result in search_results:
        message_id = search_result["hitId"]
        message = search_result["resource"]
        info = format_message_info(message)
        logger.info(f"[{info['flag']:6}] {info['date']} | {info['sender_name']} <{info['sender_email']}>")
        logger.info(f"Subject: {info['subject']}")
        logger.info(f"Preview: {info['preview']!r}")

        # Get attachments
        attachments = client.get_message_attachments(message_id, select_fields=["name", "contentType", "size"])
        assert attachments, f"No attachments found for message {message_id}"
        for attachment in attachments:
            name = attachment.get("name")
            attachment_id = attachment.get("id")
            if name.endswith(".pdf"):
                content_type = attachment.get("contentType")
                assert content_type == "application/pdf", f"Expected application/pdf, got {content_type}"
                size = attachment.get("size", 0)
                logger.info(f"  - {name} ({content_type}, {size} bytes)")

                pdf_content = client.get_attachment_content(message_id, attachment_id)
                invoice = parse_pdf_with_azure(pdf_content, name)
                logger.info(f"Invoice: {invoice}")
                if total == invoice["total"] and date == invoice["date"]:
                    logger.info(f"Invoice matches: {name}")
                    found = True
                    break

        logger.info("-" * 80)
        if found:
            break

    if found:
        return pdf_content
    logger.info("No invoice matches found")
    return None


def main():
    """Main application entry point."""
    load_dotenv()

    # Check required environment variables
    tenant_id = os.getenv("GRAPH_TENANT_ID")
    client_id = os.getenv("GRAPH_CLIENT_ID")
    assert tenant_id and client_id, "Missing required environment variables: GRAPH_TENANT_ID, GRAPH_CLIENT_ID"

    config.setup_logging()

    logger.info("Starting email reader")

    authenticator = OutlookAuthenticator(tenant_id=tenant_id, client_id=client_id, scopes=config.SCOPES)

    client = OutlookClient(authenticator)

    test_read(client)
    # test_send(client)
    date = "2025-09-13"
    test_search(client, total=947.38, date=datetime.datetime.strptime(date, "%Y-%m-%d").date())


if __name__ == "__main__":
    main()
