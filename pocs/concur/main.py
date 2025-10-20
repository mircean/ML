#!/usr/bin/env python3
"""
Main application for finding Concur expense reports with missing receipts.
"""

import logging
import os
import sys

from dotenv import load_dotenv

import config
from concur_auth import ConcurAuthenticator
from concur_client import ConcurClient

logger = logging.getLogger(__name__)


def filter_expenses_requiring_receipts(expenses):
    """
    Filter expenses that require receipts.

    Args:
        expenses: List of expense dictionaries

    Returns:
        List of expenses requiring receipts
    """
    receipt_required_expenses = []

    for expense in expenses:
        requires_receipt = (
            expense.get("isImageRequired", False)
            or expense.get("isPaperReceiptRequired", False)
            or expense.get("receiptRequired", False)
        )

        if requires_receipt:
            receipt_required_expenses.append(expense)

    return receipt_required_expenses


def format_expense_info(expense):
    """Format expense information for logging."""
    expense_type = expense.get("expenseTypeName", "Unknown")
    amount = expense.get("transactionAmount", {}).get("value", 0)
    currency = expense.get("transactionAmount", {}).get("currencyCode", "USD")
    date = expense.get("transactionDate", "Unknown date")
    description = expense.get("description", "No description")

    return f"  - {expense_type}: {currency} {amount} on {date} - {description}"


def get_reports_with_missing_receipts(client):
    """
    Get all active expense reports and identify expenses requiring receipts.

    Args:
        client: ConcurClient instance

    Returns:
        List of dictionaries containing report info and expenses requiring receipts
    """

    active_reports = client.get_active_expense_reports()
    reports_with_missing_receipts = []

    for report in active_reports:
        report_id = report["ID"]
        report_name = report.get("Name", "Unknown Report")

        try:
            expenses = client.get_expenses(report_id)
            expenses_needing_receipts = filter_expenses_requiring_receipts(expenses)

            if expenses_needing_receipts:
                reports_with_missing_receipts.append(
                    {
                        "report_id": report_id,
                        "report_name": report_name,
                        "report_total": report.get("Total", 0),
                        "report_currency": report.get("CurrencyCode", "USD"),
                        "creation_date": report.get("CreateDate"),
                        "expenses_needing_receipts": expenses_needing_receipts,
                    }
                )

        except Exception as e:
            logger.error(f"Failed to get expenses for report {report_id}: {e}")
            continue

    logger.info(
        f"Found {len(reports_with_missing_receipts)} reports with expenses requiring receipts"
    )
    return reports_with_missing_receipts


def main():
    """Main application entry point."""
    load_dotenv()

    # Check required environment variables
    client_id = os.getenv("CONCUR_CLIENT_ID")
    client_secret = os.getenv("CONCUR_CLIENT_SECRET")
    base_url = os.getenv("CONCUR_BASE_URL", "https://us2.api.concursolutions.com")
    scope = os.getenv(
                "CONCUR_SCOPE",
                "openid profile user.read identity.user.ids.read expense.report.read")

    if not client_id or not client_secret:
        logger.error("CONCUR_CLIENT_ID and CONCUR_CLIENT_SECRET must be set in .env file")
        sys.exit(1)

    config.setup_logging()

    logger.info("Starting Concur expense report receipt checker")

    try:
        # Initialize authenticator with OAuth 2.0 PKCE flow
        authenticator = ConcurAuthenticator(
            client_id=client_id,
            client_secret=client_secret,
            base_url=base_url,
            redirect_uri=os.getenv("REDIRECT_URI", "http://localhost:53682/callback"),
            scope=scope
        )

        client = ConcurClient(authenticator)

        logger.info("Fetching expense reports with missing receipts...")
        reports_with_missing_receipts = get_reports_with_missing_receipts(client)

        if not reports_with_missing_receipts:
            logger.info("✅ No active expense reports found with missing receipts!")
            return

        logger.info(
            f"📋 Found {len(reports_with_missing_receipts)} expense reports with missing receipts:"
        )
        logger.info("-" * 80)

        for report_data in reports_with_missing_receipts:
            logger.info(f"📊 Report: {report_data['report_name']}")
            logger.info(f"   ID: {report_data['report_id']}")
            logger.info(
                f"   Total: {report_data['report_currency']} {report_data['report_total']}"
            )
            logger.info(f"   Created: {report_data['creation_date']}")
            logger.info(
                f"   Expenses needing receipts: {len(report_data['expenses_needing_receipts'])}"
            )

            for expense in report_data["expenses_needing_receipts"]:
                logger.info(format_expense_info(expense))

            logger.info("-" * 80)

        total_expenses_needing_receipts = sum(
            len(report["expenses_needing_receipts"])
            for report in reports_with_missing_receipts
        )

        logger.info(
            f"🔍 Summary: {total_expenses_needing_receipts} total expenses requiring receipts across {len(reports_with_missing_receipts)} reports"
        )

    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
