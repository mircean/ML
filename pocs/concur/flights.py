"""
Main application for finding and attaching Concur expense receipts.
"""

import datetime
import logging
import os

import config
import tqdm
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from concur_auth import ConcurAuthenticator
from concur_client import ConcurClient
from dotenv import load_dotenv
from outlook_auth import OutlookAuthenticator
from outlook_client import OutlookClient

logger = logging.getLogger(__name__)

# Cache for parsed PDF documents
_pdf_parse_cache = {}


def format_expense_info(expense):
    """Format expense information for logging."""
    return f"""  - {expense["ExpenseTypeName"]}: {expense["TransactionAmount"]} on {expense["TransactionDate"]}"""


def parse_pdf_with_azure(pdf_content: bytes, filename: str) -> dict:
    """Parse PDF using Azure Form Recognizer with caching."""
    # Check if already parsed
    if filename in _pdf_parse_cache:
        logger.debug(f"Using cached parse result for {filename}")
        return _pdf_parse_cache[filename]

    endpoint = os.getenv("AZURE_ENDPOINT")
    key = os.getenv("AZURE_API_KEY")

    assert endpoint and key, "Missing Azure Form Recognizer credentials: AZURE_ENDPOINT, AZURE_API_KEY"

    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    logger.info(f"Analyzing {filename} with Azure Form Recognizer...")

    poller = client.begin_analyze_document("prebuilt-document", pdf_content)
    result = poller.result()
    total = None
    date = None
    charged_by_airline = None
    total_ticket_amount = []
    for keyvaluepair in result.key_value_pairs:
        if keyvaluepair.key.content == "Total Invoice Charge":
            try:
                total = float(keyvaluepair.value.content)
            except ValueError:
                continue
        if keyvaluepair.key.content == "Invoice Date":
            date_str = keyvaluepair.value.content
            # Try multiple date formats
            for date_format in ["%d %B %Y", "%Y-%m-%d", "%m/%d/%Y"]:
                try:
                    date = datetime.datetime.strptime(date_str, date_format).date()
                    break
                except ValueError:
                    continue
        if keyvaluepair.key.content == "Charged by Airline":
            try:
                charged_by_airline = float(keyvaluepair.value.content)
            except ValueError:
                continue
        if keyvaluepair.key.content == "Total (USD) Ticket Amount":
            try:
                total_ticket_amount.append(float(keyvaluepair.value.content))
            except ValueError:
                continue

    parsed_result = {
        "total": total,
        "date": date,
        "charged_by_airline": charged_by_airline,
        "total_ticket_amount": total_ticket_amount,
    }

    # Cache the result
    _pdf_parse_cache[filename] = parsed_result
    logger.debug(f"Cached parse result for {filename} (cache size: {len(_pdf_parse_cache)})")

    return parsed_result


def match_invoice(total, date, invoice):
    """Match invoice based on total, date, and charged by airline."""
    if date != invoice["date"]:
        return False

    if total == invoice["total"]:
        return True
    if total == invoice.get("charged_by_airline", None):
        return True
    for total_ticket_amount in invoice["total_ticket_amount"]:
        if total == total_ticket_amount:
            return True
    return False


def search(outlook_client, total, date, days_to_search=90):
    """Search emails containing 'invoice' with attachments from specified days back."""
    logger.info("Starting invoice search")

    # Calculate date to search from
    search_from_date = (datetime.datetime.now() - datetime.timedelta(days=days_to_search)).strftime("%Y-%m-%d")

    # Search for emails with attachments from Amex travel
    search_results = outlook_client.search_messages(
        query=f"hasattachments:true received>={search_from_date} from:donotreply@mytrips.amexgbt.com",
        limit=50,
        select_fields=["subject", "from", "receivedDateTime", "hasAttachments", "bodyPreview"],
    )

    if not search_results:
        logger.info("No invoice emails with attachments found")
        return None

    logger.info(f"Found {len(search_results)} invoice emails with attachments:")

    found = False
    pdf_content = None
    for search_result in tqdm.tqdm(search_results, desc="Searching for invoice"):
        message_id = search_result["hitId"]
        # message = search_result["resource"]

        # Get attachments
        attachments = outlook_client.get_message_attachments(message_id, select_fields=["name", "contentType", "size"])
        assert attachments, f"No attachments found for message {message_id}"
        for attachment in attachments:
            name = attachment.get("name")
            attachment_id = attachment.get("id")
            if name.endswith(".pdf"):
                content_type = attachment.get("contentType")
                assert content_type == "application/pdf", f"Expected application/pdf, got {content_type}"
                size = attachment.get("size", 0)
                logger.info(f"Attachment {name} size: {size} bytes")

                pdf_content = outlook_client.get_attachment_content(message_id, attachment_id)
                invoice = parse_pdf_with_azure(pdf_content, name)
                logger.info(f"Invoice: {invoice}")
                if match_invoice(total, date, invoice):
                    logger.info(f"Invoice matches: {name}")
                    found = True
                    break

        if found:
            break

    if found:
        return pdf_content
    logger.info("No invoice matches found")
    return None


def main():
    """Main application entry point."""
    cfg = config.parse_config()

    load_dotenv()
    config.setup_logging()

    logger.info("Starting Concur expense report receipt checker")

    # Initialize Concur client
    client_id = os.getenv("CONCUR_CLIENT_ID")
    client_secret = os.getenv("CONCUR_CLIENT_SECRET")
    assert client_id and client_secret, "Missing required environment variables: CONCUR_CLIENT_ID, CONCUR_CLIENT_SECRET"
    base_url = os.getenv("CONCUR_BASE_URL", "https://us2.api.concursolutions.com")
    scope = os.getenv("CONCUR_SCOPE", "openid profile user.read identity.user.ids.read expense.report.read IMAGE EXPRPT")

    authenticator = ConcurAuthenticator(
        client_id=client_id,
        client_secret=client_secret,
        base_url=base_url,
        redirect_uri=os.getenv("REDIRECT_URI", "http://localhost:53682/callback"),
        scope=scope,
    )
    concur_client = ConcurClient(authenticator)

    # Initialize Outlook client
    tenant_id = os.getenv("GRAPH_TENANT_ID")
    outlook_client_id = os.getenv("GRAPH_CLIENT_ID")
    assert tenant_id and outlook_client_id, "Missing required environment variables: GRAPH_TENANT_ID, GRAPH_CLIENT_ID"

    outlook_authenticator = OutlookAuthenticator(tenant_id=tenant_id, client_id=outlook_client_id, scopes=config.SCOPES)
    outlook_client = OutlookClient(outlook_authenticator)

    logger.info("Searching for airline expenses without receipts...")

    active_reports = concur_client.get_active_expense_reports()

    total_airline_expenses_no_receipt = 0
    matched_count = 0
    not_matched_count = 0

    for report in active_reports:
        report_id = report["ID"]
        report_name = report.get("Name", "Unknown Report")
        logger.info(f"Processing report: {report_name}")

        expenses = concur_client.get_expenses(report_id)
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

                receipt_pdf = search(outlook_client, total=amount, date=transaction_date)

                if receipt_pdf:
                    logger.info("✅ Found matching receipt in Outlook!")
                    matched_count += 1
                    if cfg.dry_run:
                        logger.info(f"[DRY RUN] Would upload receipt to expense entry {expense['ID']}")
                    else:
                        concur_client.upload_receipt_image(expense["ID"], receipt_pdf)
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


if __name__ == "__main__":
    main()
