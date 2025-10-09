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
    load_dotenv()

    # Parse configuration with command line overrides
    cfg = config.parse_config()

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
    if cfg.skip_data_download:
        logger.info("Skipping data synchronization (--skip-data-download flag set)")
    else:
        logger.info("Running data synchronization...")
        sync_data.main()
        logger.info("Data sync completed successfully")

    # Load portfolio data
    portfolio_path = Path(config.PORTFOLIO_FILE)
    with open(portfolio_path, "r") as f:
        portfolio = json.load(f)

    # Step 2: Run trading agent
    logger.info("Running trading agent...")
    trading_analysis, final_portfolio = agent.main(cfg)
    logger.info("Trading agent completed successfully")

    # notifier = EmailNotifier()

    subject = f"🔔 Daily Trading Report - {datetime.now().strftime('%Y-%m-%d')}"
    body = agent.print_analysis(trading_analysis)
    body += "\n\n"
    body += agent.print_portfolio(portfolio, "Initial Portfolio")
    if final_portfolio:
        body += "\n\n"
        body += agent.print_portfolio(final_portfolio, "Final Portfolio")

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
