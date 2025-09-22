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

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to Outlook API."""
        token = self.authenticator.get_access_token()
        request_headers = {"Authorization": f"Bearer {token}"}
        if headers:
            request_headers.update(headers)
        self.session.headers.update(request_headers)

        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, params=params, timeout=30)

        # Retry once if token expired
        if response.status_code in (401, 403):
            logger.debug("Token expired, retrying with fresh token")
            token = self.authenticator.get_access_token()
            request_headers["Authorization"] = f"Bearer {token}"
            self.session.headers.update(request_headers)
            response = self.session.request(method, url, params=params, timeout=30)

        response.raise_for_status()
        return response.json()

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

    def search_messages(self, search_term: str, limit=50, select_fields=None) -> List[Dict[str, Any]]:
        """
        Search for messages containing a specific term.

        Args:
            search_term: Term to search for in messages
            limit: Number of messages to retrieve
            select_fields: List of fields to select

        Returns:
            List of message dictionaries matching the search term
        """
        endpoint = "/me/messages"

        params = {
            "$search": f'"{search_term}"',
            "$top": limit
        }

        if select_fields:
            params["$select"] = ",".join(select_fields)

        logger.info(f"Searching for messages containing: {search_term}")

        # Search requires ConsistencyLevel header
        headers = {"ConsistencyLevel": "eventual"}
        response_data = self._make_request("GET", endpoint, params, headers)

        messages = response_data.get("value", [])

        # Client-side sort by receivedDateTime desc (search doesn't support $orderby)
        messages.sort(key=lambda m: m.get("receivedDateTime", ""), reverse=True)

        logger.info(f"Found {len(messages)} messages matching '{search_term}'")
        return messages