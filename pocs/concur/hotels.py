"""
Itemize hotel expenses in Concur expense reports.

For each hotel expense without itemizations, creates nightly itemizations
with equal per-night amounts (room + tax combined).
"""

import datetime
import logging
import os

import config
from concur_auth import ConcurAuthenticator
from concur_client import ConcurClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def format_expense_info(expense):
    """Format expense information for logging."""
    return (
        f"  - {expense['ExpenseTypeName']}: {expense['TransactionAmount']} "
        f"on {expense['TransactionDate']}"
    )


def parse_date(date_str):
    """Parse a date string from the Concur API (YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD)."""
    return datetime.datetime.strptime(date_str[:10], "%Y-%m-%d").date()


def get_hotel_dates(concur_client, report_id, expense_id):
    """Get check-in and check-out dates from the v4 API.

    The v3 entries API doesn't expose hotel dates. The v4 expense detail
    endpoint returns travel.hotelCheckinDate and travel.hotelCheckoutDate.
    """
    detail = concur_client.get_expense_v4(report_id, expense_id)
    travel = detail.get("travel", {})

    checkin_str = travel.get("hotelCheckinDate")
    checkout_str = travel.get("hotelCheckoutDate")
    assert checkin_str and checkout_str, (
        f"Missing hotel dates in v4 API response for expense {expense_id}. "
        f"travel={travel}"
    )

    checkin_date = parse_date(checkin_str)
    checkout_date = parse_date(checkout_str)
    nights = (checkout_date - checkin_date).days
    assert nights > 0, (
        f"Invalid date range: checkin={checkin_date}, checkout={checkout_date}"
    )

    return checkin_date, checkout_date, nights


def itemize_hotel_expense(concur_client, expense, report_id, dry_run):
    """Create nightly itemizations for a hotel expense.

    Splits the total amount evenly across nights. Each night gets a single
    LODNG itemization combining room rate and tax.
    """
    entry_id = expense["ID"]
    expense_id = expense["ExpenseID"]
    total_amount = expense["TransactionAmount"]
    currency = expense["TransactionCurrencyCode"]

    checkin_date, checkout_date, nights = get_hotel_dates(
        concur_client, report_id, expense_id
    )
    per_night_amount = round(total_amount / nights, 2)

    # Adjust last night for rounding differences
    rounding_remainder = round(total_amount - (per_night_amount * nights), 2)

    logger.info(
        f"Itemizing {entry_id}: {nights} nights @ {per_night_amount}/night "
        f"(total: {total_amount} {currency}, "
        f"checkin: {checkin_date}, checkout: {checkout_date})"
    )

    for night_index in range(nights):
        night_date = checkin_date + datetime.timedelta(days=night_index)
        amount = per_night_amount
        if night_index == nights - 1:
            amount = round(amount + rounding_remainder, 2)

        if dry_run:
            logger.info(
                f"[DRY RUN] Would create itemization: "
                f"LODNG {night_date} {amount} {currency}"
            )
        else:
            concur_client.create_itemization(
                entry_id=entry_id,
                expense_type_code="LODNG",
                transaction_date=str(night_date),
                transaction_amount=amount,
            )

    return nights


def main():
    """Main entry point for hotel expense itemization."""
    cfg = config.parse_config()

    load_dotenv()
    config.setup_logging()

    logger.info("Starting hotel expense itemization")

    # Initialize Concur client
    client_id = os.getenv("CONCUR_CLIENT_ID")
    client_secret = os.getenv("CONCUR_CLIENT_SECRET")
    assert client_id and client_secret, (
        "Missing required environment variables: CONCUR_CLIENT_ID, CONCUR_CLIENT_SECRET"
    )
    base_url = os.getenv(
        "CONCUR_BASE_URL", "https://us2.api.concursolutions.com"
    )
    scope = os.getenv(
        "CONCUR_SCOPE",
        "openid profile user.read identity.user.ids.read expense.report.read IMAGE EXPRPT",
    )

    authenticator = ConcurAuthenticator(
        client_id=client_id,
        client_secret=client_secret,
        base_url=base_url,
        redirect_uri=os.getenv(
            "REDIRECT_URI", "http://localhost:53682/callback"
        ),
        scope=scope,
    )
    concur_client = ConcurClient(authenticator)

    active_reports = concur_client.get_active_expense_reports()

    total_hotel_expenses = 0
    itemized_count = 0
    skipped_count = 0

    for report in active_reports:
        report_id = report["ID"]
        report_name = report.get("Name", "Unknown Report")
        logger.info(f"Processing report: {report_name}")

        expenses = concur_client.get_expenses(report_id)

        for expense in expenses:
            expense_type = expense["ExpenseTypeName"]
            if expense_type != "Hotel":
                continue

            total_hotel_expenses += 1
            logger.info(f"Found hotel expense: {format_expense_info(expense)}")

            # Skip if already itemized
            has_itemizations = expense.get("HasItemizations", False)
            if has_itemizations:
                logger.info(f"Skipping already itemized expense {expense['ID']}")
                skipped_count += 1
                continue

            itemize_hotel_expense(concur_client, expense, report_id, cfg.dry_run)
            itemized_count += 1

    logger.info("=" * 80)
    logger.info("HOTEL ITEMIZATION STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total hotel expenses found: {total_hotel_expenses}")
    logger.info(f"Itemized: {itemized_count}")
    logger.info(f"Skipped (already itemized): {skipped_count}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
