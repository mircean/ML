"""
Main application for finding Concur expense reports with missing receipts.
"""

import datetime
import logging
import os
import sys

import config
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from concur_auth import ConcurAuthenticator
from concur_client import ConcurClient
from dotenv import load_dotenv
from outlook_auth import OutlookAuthenticator
from outlook_client import OutlookClient
from outlook_main import search

logger = logging.getLogger(__name__)


def format_expense_info(expense):
    """Format expense information for logging."""
    return f"""  - {expense["ExpenseTypeName"]}: {expense["TransactionAmount"]} on {expense["TransactionDate"]}"""


def find_and_attach_receipts(client):
    """
    Find airline expenses without receipts and search Outlook for matching receipts.
    Prints statistics at the end.

    Args:
        client: ConcurClient instance
    """

    logger.info("Searching for airline expenses without receipts...")

    active_reports = client.get_active_expense_reports()

    total_airline_expenses_no_receipt = 0
    matched_count = 0
    not_matched_count = 0

    for report in active_reports:
        report_id = report["ID"]
        report_name = report.get("Name", "Unknown Report")
        logger.info(f"Processing report: {report_name}")

        expenses = client.get_expenses(report_id)
        # Iterate through all expenses to check for airline expenses without receipts
        for expense in expenses:
            expense_type = expense["ExpenseTypeName"]
            receipt_required = expense["IsImageRequired"]
            has_receipt = expense.get("HasImage", False)
            if expense_type == "Airfare" and receipt_required and not has_receipt:
                total_airline_expenses_no_receipt += 1
                logger.info(f"Found airline expense without receipt: {format_expense_info(expense)}")

                amount = expense["TransactionAmount"]
                transaction_date_str = expense["TransactionDate"]
                transaction_date = datetime.datetime.strptime(transaction_date_str[:10], "%Y-%m-%d").date()

                logger.info(f"Searching Outlook for receipt: amount={amount}, date={transaction_date}")

                receipt_pdf = search(total=amount, date=transaction_date)

                if receipt_pdf:
                    logger.info("✅ Found matching receipt in Outlook!")
                    matched_count += 1
                    # TODO: Upload receipt to Concur
                else:
                    logger.warning("❌ No matching receipt found in Outlook")
                    not_matched_count += 1

    # Print statistics
    logger.info("=" * 80)
    logger.info("📊 AIRLINE RECEIPT SEARCH STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total airline expenses without receipts: {total_airline_expenses_no_receipt}")
    logger.info(f"Matched receipts found in Outlook: {matched_count}")
    logger.info(f"Not matched: {not_matched_count}")
    logger.info("=" * 80)


def main():
    """Main application entry point."""
    load_dotenv()
    config.setup_logging()

    # Check required environment variables
    client_id = os.getenv("CONCUR_CLIENT_ID")
    client_secret = os.getenv("CONCUR_CLIENT_SECRET")
    base_url = os.getenv("CONCUR_BASE_URL", "https://us2.api.concursolutions.com")
    scope = os.getenv("CONCUR_SCOPE", "openid profile user.read identity.user.ids.read expense.report.read")

    if not client_id or not client_secret:
        logger.error("CONCUR_CLIENT_ID and CONCUR_CLIENT_SECRET must be set in .env file")
        sys.exit(1)

    logger.info("Starting Concur expense report receipt checker")

    # Initialize authenticator with OAuth 2.0 PKCE flow
    authenticator = ConcurAuthenticator(
        client_id=client_id,
        client_secret=client_secret,
        base_url=base_url,
        redirect_uri=os.getenv("REDIRECT_URI", "http://localhost:53682/callback"),
        scope=scope,
    )

    client = ConcurClient(authenticator)

    find_and_attach_receipts(client)


if __name__ == "__main__":
    main()
