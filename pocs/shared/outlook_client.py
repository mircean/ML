"""
Outlook API client for email operations.
"""

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class OutlookClient:
    """Client for interacting with Outlook API."""

    def __init__(self, authenticator):
        self.authenticator = authenticator
        self.base_url = "https://graph.microsoft.com/v1.0"
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to Outlook API."""
        token = self.authenticator.get_access_token()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, params=params, timeout=30)

        # Retry once if token expired
        if response.status_code in (401, 403):
            logger.debug("Token expired, retrying with fresh token")
            token = self.authenticator.get_access_token()
            self.session.headers.update({"Authorization": f"Bearer {token}"})
            response = self.session.request(method, url, params=params, timeout=30)

        response.raise_for_status()
        return response.json()

    def _make_post_request(self, endpoint: str, json_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make authenticated POST request to Outlook API with JSON data."""
        token = self.authenticator.get_access_token()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        })

        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, json=json_data, timeout=30)

        # Retry once if token expired
        if response.status_code in (401, 403):
            logger.debug("Token expired, retrying with fresh token")
            token = self.authenticator.get_access_token()
            self.session.headers.update({"Authorization": f"Bearer {token}"})
            response = self.session.post(url, json=json_data, timeout=30)

        response.raise_for_status()

        # Some endpoints return empty response on success
        if response.content:
            return response.json()
        return None

    def get_messages(self, folder="Inbox", limit=10, select_fields=None) -> List[Dict[str, Any]]:
        """
        Get messages from specified folder.

        Args:
            folder: Email folder name (default: "Inbox")
            limit: Number of messages to retrieve
            select_fields: List of fields to select

        Returns:
            List of message dictionaries
        """
        endpoint = f"/me/mailFolders/{folder}/messages"

        params = {
            "$orderby": "receivedDateTime DESC",
            "$top": limit
        }

        if select_fields:
            params["$select"] = ",".join(select_fields)

        logger.info(f"Fetching {limit} messages from {folder}")
        response_data = self._make_request("GET", endpoint, params)

        messages = response_data.get("value", [])
        logger.info(f"Retrieved {len(messages)} messages")
        return messages

    def get_message_attachments(self, message_id: str, select_fields=None) -> List[Dict[str, Any]]:
        """
        Get attachments for a specific message.

        Args:
            message_id: The message ID
            select_fields: List of fields to select

        Returns:
            List of attachment dictionaries
        """
        endpoint = f"/me/messages/{message_id}/attachments"

        params = {}
        if select_fields:
            params["$select"] = ",".join(select_fields)

        logger.debug(f"Fetching attachments for message {message_id}")
        response_data = self._make_request("GET", endpoint, params)

        attachments = response_data.get("value", [])
        logger.debug(f"Found {len(attachments)} attachments")
        return attachments

    def send_email(self, to: str, subject: str, body: str, body_type: str = "Text") -> str:
        """
        Send an email from the authenticated user's account.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body content
            body_type: Body content type ("Text" or "HTML")

        Returns:
            Message ID of sent email
        """
        endpoint = "/me/sendMail"

        message_data = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": body_type,
                    "content": body
                },
                "toRecipients": [
                    {
                        "emailAddress": {
                            "address": to
                        }
                    }
                ]
            }
        }

        logger.info(f"Sending email to {to} with subject: {subject}")

        self._make_post_request(endpoint, message_data)

        logger.info(f"Email sent successfully to {to}")
        return "sent"