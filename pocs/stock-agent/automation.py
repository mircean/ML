import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import agent
import config
import sync_data
from dotenv import load_dotenv
from outlook_auth import OutlookAuthenticator
from outlook_client import OutlookClient

logger = logging.getLogger(__name__)


def main():
    """Main automation function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Daily trading automation")
    parser.add_argument(
        "--skip-data-download",
        action="store_true",
        default=False,
        help="Skip data synchronization step (default: False)"
    )
    args = parser.parse_args()

    load_dotenv()

    # Check required environment variables for email
    tenant_id = os.getenv("GRAPH_TENANT_ID")
    client_id = os.getenv("GRAPH_CLIENT_ID")
    assert tenant_id and client_id, "Missing required environment variables: GRAPH_TENANT_ID, GRAPH_CLIENT_ID"

    config.setup_logging()

    logger.info("Starting daily trading automation")

    # Check if it's a weekday
    day_of_week = datetime.now().weekday()  # Monday=0, Sunday=6
    if day_of_week >= 5:  # Saturday=5, Sunday=6
        logger.info(f"Skipping - Weekend (day {day_of_week})")
        return

    # Step 1: Sync data (optional)
    if args.skip_data_download:
        logger.info("Skipping data synchronization (--skip-data-download flag set)")
    else:
        logger.info("Running data synchronization...")
        sync_data.main()
        logger.info("Data sync completed successfully")

    # Load portfolio data
    portfolio_path = Path("portfolio.json")
    with open(portfolio_path, "r") as f:
        portfolio = json.load(f)

    # Step 2: Run trading agent
    logger.info("Running trading agent...")
    trading_analysis = agent.main()
    logger.info("Trading agent completed successfully")

    # notifier = EmailNotifier()

    subject = f"🔔 Daily Trading Report - {datetime.now().strftime('%Y-%m-%d')}"
    body = f"""
Portfolio Value: ${portfolio.get("total_value", 0):.2f}

Summary: {trading_analysis.summary}

Market Outlook: {trading_analysis.market_outlook}

Risk Assessment: {trading_analysis.risk_assessment}

"""

    for recommendation in trading_analysis.trade_recommendations:
        body += f"""
{recommendation.action} {recommendation.symbol} - {recommendation.shares} shares at {recommendation.price} - {recommendation.reasoning} - {recommendation.confidence}
"""

    # Temporary: Write to file
    filename = subject.replace(":", "-").replace(" ", "_") + ".txt"
    with open(filename, "w") as f:
        f.write(f"Subject: {subject}\n\n{body}")
    logger.info(f"Daily trading report written to {filename}")

    authenticator = OutlookAuthenticator(tenant_id=tenant_id, client_id=client_id, scopes=config.SCOPES)

    client = OutlookClient(authenticator)

    client.send_email(
        to="mircean@outlook.com",
        subject=subject,
        body=body,
        body_type="Text",  # or "HTML"
    )

    logger.info("Daily trading report sent to email")

    logger.info("Daily trading automation completed")


if __name__ == "__main__":
    main()
