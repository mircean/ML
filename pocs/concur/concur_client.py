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

    def get_active_expense_reports(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve active expense reports using Reports v3 API.

        Returns:
            List of expense report dictionaries
        """
        endpoint = "/api/v3.0/expense/reports"
        params = {"limit": limit}

        logger.info(f"Fetching active expense reports with limit {limit}")
        response_data = self._make_request("GET", endpoint, params)

        reports = response_data.get("Items", [])
        active_reports = [
            report
            for report in reports
            if report.get("ApprovalStatusCode") in ["A_NOTF", "A_PEND"]  # Active/Not Submitted or Pending
        ]

        logger.info(f"Found {len(active_reports)} active expense reports out of {len(reports)} total")
        return active_reports

    def get_expenses(self, report_id: str) -> List[Dict[str, Any]]:
        """
        Get detailed expense information for a report using Expense Entries v3 API.

        Args:
            report_id: The expense report ID

        Returns:
            List of expense dictionaries
        """
        endpoint = "/api/v3.0/expense/entries"
        params = {"reportID": report_id, "limit": 100}

        logger.info(f"Fetching expense details for report {report_id}")
        response_data = self._make_request("GET", endpoint, params)

        expenses = response_data.get("Items", [])
        logger.info(f"Found {len(expenses)} expenses in report {report_id}")
        assert len(expenses) != 100, "More than 100 expenses found in report {report_id}, need to implement pagination"
        return expenses
