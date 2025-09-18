#!/usr/bin/env python3
"""
Email attachment reader for expense receipts using OAuth2 authentication.
"""

import logging
import os
import sys
from datetime import datetime, timedelta

import msal
from dotenv import load_dotenv
from exchangelib import DELEGATE, Account, Configuration, OAuth2Credentials

logger = logging.getLogger(__name__)

# Microsoft Graph/Exchange OAuth2 configuration
AUTHORITY = "https://login.microsoftonline.com/organizations"
SCOPES = ["https://outlook.office365.com/EWS.AccessAsUser.All"]

# Outlook app client ID - pre-approved in most corporate tenants
# DEFAULT_CLIENT_ID = "d3590ed6-52b3-4102-aeff-aad2292ab01c"
DEFAULT_CLIENT_ID = "9199bf20-a13f-4107-85dc-02114787ef48"


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(level=getattr(logging, log_level.upper()), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def load_email_config():
    """Load email configuration from environment variables."""
    load_dotenv()

    required_vars = ["EMAIL_ADDRESS"]

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

    config["client_id"] = os.getenv("EMAIL_CLIENT_ID") or DEFAULT_CLIENT_ID
    config["log_level"] = os.getenv("LOG_LEVEL", "INFO")
    config["days_back"] = int(os.getenv("EMAIL_DAYS_BACK", "7"))

    return config


def get_token_cache(cache_file):
    """Load token cache from file."""
    cache = msal.SerializableTokenCache()
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache.deserialize(f.read())
    return cache


def save_token_cache(cache, cache_file):
    """Save token cache to file."""
    if cache.has_state_changed:
        with open(cache_file, "w") as f:
            f.write(cache.serialize())


def authenticate_oauth2(client_id, email_address, cache_file):
    """Perform OAuth2 authentication with browser flow."""
    cache = get_token_cache(cache_file)

    app = msal.PublicClientApplication(client_id=client_id, authority=AUTHORITY, token_cache=cache)

    # Try to get token silently first
    accounts = app.get_accounts(username=email_address)
    if accounts:
        logger.info("Found cached account, attempting silent authentication...")
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result:
            save_token_cache(cache, cache_file)
            logger.info("✅ Silent authentication successful")
            return result["access_token"]

    logger.info("No cached token found, initiating interactive browser authentication...")

    print("\n" + "=" * 60)
    print("📧 EMAIL AUTHENTICATION REQUIRED")
    print("=" * 60)
    print("Opening browser for interactive authentication...")
    print(f"Please sign in with: {email_address}")
    print("=" * 60)

    # Interactive browser flow
    result = app.acquire_token_interactive(scopes=SCOPES, login_hint=email_address, prompt="select_account")

    if "access_token" not in result:
        logger.error(f"Authentication failed: {result.get('error_description', 'Unknown error')}")
        logger.error(f"Full error: {result}")
        raise Exception("Authentication failed")

    save_token_cache(cache, cache_file)
    logger.info("✅ Authentication successful!")
    return result["access_token"]


def connect_to_exchange(email_address, client_id):
    """Connect to Exchange server using OAuth2."""
    try:
        access_token = authenticate_oauth2(client_id, email_address, ".email_tokens.json")

        credentials = OAuth2Credentials(
            client_id=client_id,
            client_secret=None,  # Not needed for public client
            tenant_id=None,  # Will be determined from token
            identity=email_address,
            access_token=access_token,
        )

        config = Configuration(credentials=credentials)
        account = Account(primary_smtp_address=email_address, config=config, autodiscover=True, access_type=DELEGATE)

        # Test connection
        _ = account.root.effective_rights

        logger.info(f"Successfully connected to Exchange for {email_address}")
        return account

    except Exception as e:
        logger.error(f"Failed to connect to Exchange: {e}")
        raise


def process_messages_with_attachments(account, days_back):
    """Process messages with attachments from the last N days."""
    start_date = datetime.now() - timedelta(days=days_back)

    messages_with_attachments = account.inbox.filter(datetime_received__gte=start_date).filter(has_attachments=True).order_by("-datetime_received")

    logger.info(f"Searching for messages with attachments from the last {days_back} days...")

    message_count = 0
    attachment_count = 0

    for message in messages_with_attachments:
        try:
            message_count += 1
            num_attachments = len(message.attachments)
            attachment_count += num_attachments

            logger.info(f"📧 {message.subject}")
            logger.info(f"   From: {message.sender}")
            logger.info(f"   Date: {message.datetime_received.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"   Attachments: {num_attachments}")

            for attachment in message.attachments:
                if hasattr(attachment, "name"):
                    size_kb = len(attachment.content) // 1024 if hasattr(attachment, "content") else 0
                    logger.info(f"     • {attachment.name} ({size_kb} KB)")

            logger.info("-" * 60)

        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            continue

    return message_count, attachment_count


def main():
    """Main application entry point."""
    config = load_email_config()
    setup_logging(config["log_level"])

    logger.info("Starting email attachment reader")

    try:
        account = connect_to_exchange(config["email_address"], config["client_id"])

        message_count, attachment_count = process_messages_with_attachments(account, config["days_back"])

        if message_count == 0:
            logger.info("✅ No messages with attachments found in the specified time range")
            return

        logger.info("📊 Summary:")
        logger.info(f"   Messages with attachments: {message_count}")
        logger.info(f"   Total attachments: {attachment_count}")

    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
