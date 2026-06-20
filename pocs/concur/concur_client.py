"""
SAP Concur API client for expense reports and receipts.
"""

import logging
from typing import Any, Dict, List, Optional

import requests
from concur_auth import ConcurAuthenticator

logger = logging.getLogger(__name__)


class ConcurClient:
    """Client for interacting with SAP Concur APIs."""

    def __init__(self, authenticator: ConcurAuthenticator):
        self.authenticator = authenticator
        self.base_url = authenticator.base_url.rstrip("/")

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to Concur API."""
        url = f"{self.base_url}{endpoint}"
        headers = self.authenticator.get_auth_headers()

        logger.info(f"Making {method} request to {endpoint}")

        response = requests.request(method, url, headers=headers, params=params)

        if response.status_code != 200:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            response.raise_for_status()

        return response.json()

    def _make_request_url(self, method: str, url: str) -> Dict[str, Any]:
        """Make authenticated request to a full URL (for pagination)."""
        headers = self.authenticator.get_auth_headers()

        logger.info(f"Making {method} request to {url}")

        response = requests.request(method, url, headers=headers)

        if response.status_code != 200:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            response.raise_for_status()

        return response.json()

    def _get_all_items(self, endpoint: str, params: Dict) -> List[Dict[str, Any]]:
        """Fetch all items from a paginated v3 API endpoint following NextPage links."""
        all_items = []

        response_data = self._make_request("GET", endpoint, params)
        all_items.extend(response_data.get("Items", []))
        logger.info(f"Fetched {len(all_items)} items")

        next_page = response_data.get("NextPage")
        while next_page:
            response_data = self._make_request_url("GET", next_page)
            items = response_data.get("Items", [])
            all_items.extend(items)
            logger.info(f"Fetched {len(items)} more items (total: {len(all_items)})")
            next_page = response_data.get("NextPage")

        return all_items

    def get_active_expense_reports(self) -> List[Dict[str, Any]]:
        """
        Retrieve active expense reports using Reports v3 API.

        Returns:
            List of expense report dictionaries
        """
        logger.info("Fetching all expense reports")
        reports = self._get_all_items("/api/v3.0/expense/reports", {"limit": 100})

        status_counts = {}
        for report in reports:
            status = report.get("ApprovalStatusCode", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        logger.info(f"Report status distribution: {status_counts}")

        active_reports = [
            report
            for report in reports
            if report.get("ApprovalStatusCode") in ["A_NOTF", "A_PEND"]
        ]

        logger.info(f"Found {len(active_reports)} active expense reports out of {len(reports)} total")
        return active_reports

    def get_expenses(self, report_id: str) -> List[Dict[str, Any]]:
        """
        Get all expenses for a report using Expense Entries v3 API.

        Args:
            report_id: The expense report ID

        Returns:
            List of expense dictionaries
        """
        logger.info(f"Fetching expense details for report {report_id}")
        expenses = self._get_all_items("/api/v3.0/expense/entries", {"reportID": report_id, "limit": 100})
        logger.info(f"Found {len(expenses)} expenses in report {report_id}")
        return expenses

    def upload_receipt_image(self, entry_id: str, image_data: bytes, content_type: str = "application/pdf") -> None:
        """
        Upload a receipt image to an expense entry using the Image v1 API.

        Note: Once an image is attached to an entry, you cannot append additional images.

        Args:
            entry_id: The expense entry ID (from the v3 entries API 'ID' field)
            image_data: Raw image/PDF bytes (max 10 MB)
            content_type: MIME type - application/pdf, image/jpeg, or image/png
        """
        url = f"https://www.concursolutions.com/api/image/v1.0/expenseentry/{entry_id}"
        headers = self.authenticator.get_auth_headers()
        headers["Content-Type"] = content_type
        headers["Accept"] = "application/xml"

        logger.info(f"Uploading receipt ({len(image_data)} bytes) to expense entry {entry_id}")

        response = requests.post(url, headers=headers, data=image_data)

        assert response.status_code == 201, (
            f"Receipt upload failed with status {response.status_code}: {response.text}"
        )

        logger.info(f"Receipt uploaded successfully to expense entry {entry_id}")

        # Save the expense entry to trigger re-validation and clear receipt-required alert
        self._save_expense_entry(entry_id)

    def _save_expense_entry(self, entry_id: str) -> None:
        """Save an expense entry via PUT to trigger re-validation after receipt upload."""
        url = f"{self.base_url}/api/v3.0/expense/entries/{entry_id}"
        headers = self.authenticator.get_auth_headers()
        headers["Content-Type"] = "application/json"

        logger.info(f"Saving expense entry {entry_id} to clear receipt-required alert")

        response = requests.put(url, headers=headers, json={})

        assert response.status_code in (200, 204), (
            f"Expense entry save failed with status {response.status_code}: {response.text}"
        )

        logger.info(f"Expense entry {entry_id} saved successfully")

    def _get_user_id(self) -> str:
        """Get the current user's UUID from the profile API. Cached after first call."""
        if not hasattr(self, "_user_id"):
            data = self._make_request("GET", "/profile/v1/me")
            self._user_id = data["id"]
            logger.info(f"Resolved user ID: {self._user_id}")
        return self._user_id

    def get_expense_v4(self, report_id: str, expense_id: str) -> Dict[str, Any]:
        """
        Get detailed expense data via the v4 Expense Reports API.

        The v4 API returns fields not available in v3, including
        travel.hotelCheckinDate and travel.hotelCheckoutDate.

        Args:
            report_id: The expense report ID
            expense_id: The ExpenseID (not the v3 entry ID)

        Returns:
            Expense detail dictionary
        """
        user_id = self._get_user_id()
        endpoint = (
            f"/expensereports/v4/users/{user_id}/context/TRAVELER"
            f"/reports/{report_id}/expenses/{expense_id}"
        )
        return self._make_request("GET", endpoint)

    def get_itemizations(self, entry_id: str) -> List[Dict[str, Any]]:
        """
        Get all itemizations for an expense entry.

        Args:
            entry_id: The expense entry ID

        Returns:
            List of itemization dictionaries
        """
        logger.info(f"Fetching itemizations for entry {entry_id}")
        itemizations = self._get_all_items(
            "/api/v3.0/expense/itemizations", {"entryID": entry_id, "limit": 100}
        )
        logger.info(f"Found {len(itemizations)} itemizations for entry {entry_id}")
        return itemizations

    def create_itemization(
        self,
        entry_id: str,
        expense_type_code: str,
        transaction_date: str,
        transaction_amount: float,
    ) -> Dict[str, Any]:
        """
        Create an itemization for an expense entry.

        Args:
            entry_id: The parent expense entry ID
            expense_type_code: Expense type code (e.g., "LODNG")
            transaction_date: Date in YYYY-MM-DD format
            transaction_amount: Amount for this itemization

        Returns:
            Response dict with ID and URI of created itemization
        """
        url = f"{self.base_url}/api/v3.0/expense/itemizations"
        headers = self.authenticator.get_auth_headers()
        headers["Content-Type"] = "application/json"

        payload = {
            "EntryID": entry_id,
            "ExpenseTypeCode": expense_type_code,
            "TransactionDate": transaction_date,
            "TransactionAmount": transaction_amount,
        }

        logger.info(
            f"Creating itemization for entry {entry_id}: "
            f"{expense_type_code} {transaction_date} ${transaction_amount}"
        )

        response = requests.post(url, headers=headers, json=payload)

        assert response.status_code in (200, 201), (
            f"Itemization creation failed with status {response.status_code}: {response.text}"
        )

        result = response.json()
        logger.info(f"Itemization created: {result.get('ID', 'unknown')}")
        return result
