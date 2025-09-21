"""
SAP Concur OAuth 2.0 authentication module.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class ConcurAuthenticator:
    """Handles OAuth 2.0 authentication with SAP Concur API."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        company_uuid: Optional[str] = None,
        request_token: Optional[str] = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")

        # Support both username/password and company request token authentication
        self.username = username
        self.password = password
        self.company_uuid = company_uuid
        self.request_token = request_token

        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.token_file = ".concur_tokens.json"

        # Try to load existing tokens
        self._load_tokens()

    def _make_token_request(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Make a token request to the OAuth endpoint."""
        url = f"{self.base_url}/oauth2/v0/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}

        logger.info(f"Making token request to {url}")
        response = requests.post(url, data=data, headers=headers)

        if response.status_code != 200:
            logger.error(f"Token request failed with status {response.status_code}: {response.text}")
            response.raise_for_status()

        return response.json()

    def _load_tokens(self):
        """Load tokens from file if they exist and are valid."""
        if not os.path.exists(self.token_file):
            return

        try:
            with open(self.token_file, "r") as f:
                token_data = json.load(f)

            self.access_token = token_data.get("access_token")
            self.refresh_token = token_data.get("refresh_token")

            if token_data.get("expires_at"):
                self.token_expires_at = datetime.fromisoformat(token_data["expires_at"])

            logger.info("Loaded existing tokens from file")

        except Exception as e:
            logger.warning(f"Could not load tokens from file: {e}")

    def _save_tokens(self):
        """Save current tokens to file."""
        token_data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None,
        }

        try:
            with open(self.token_file, "w") as f:
                json.dump(token_data, f, indent=2)
            logger.info("Saved tokens to file")
        except Exception as e:
            logger.warning(f"Could not save tokens to file: {e}")

    def authenticate(self) -> str:
        """Authenticate and return access token."""
        if self.access_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            logger.info("Using cached access token")
            return self.access_token

        if self.refresh_token:
            logger.info("Refreshing access token")
            return self._refresh_access_token()
        else:
            logger.info("Getting initial access token")
            return self._get_initial_token()

    def _get_initial_token(self) -> str:
        """Get initial access token using password grant."""
        if self.company_uuid and self.request_token:
            # Company-level authentication for SSO users
            logger.info("Using company request token authentication")
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "password",
                "username": self.company_uuid,
                "password": self.request_token,
                "credtype": "authtoken",
            }
        elif self.username and self.password:
            # Individual user authentication
            logger.info("Using username/password authentication")
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
            }
        else:
            raise ValueError("Must provide either username/password or company_uuid/request_token")

        response_data = self._make_token_request(data)
        return self._process_token_response(response_data)

    def _refresh_access_token(self) -> str:
        """Refresh access token using refresh token."""
        data = {"client_id": self.client_id, "client_secret": self.client_secret, "grant_type": "refresh_token", "refresh_token": self.refresh_token}

        try:
            response_data = self._make_token_request(data)
            return self._process_token_response(response_data)
        except requests.RequestException as e:
            logger.warning(f"Refresh token failed: {e}. Getting new token.")
            return self._get_initial_token()

    def _process_token_response(self, response_data: Dict[str, Any]) -> str:
        """Process token response and store tokens."""
        self.access_token = response_data["access_token"]

        if "refresh_token" in response_data:
            self.refresh_token = response_data["refresh_token"]
            logger.info("Stored new refresh token")

        expires_in = response_data.get("expires_in", 3600)
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

        logger.info(f"Access token obtained, expires at {self.token_expires_at}")

        # Save tokens to file for future use
        self._save_tokens()

        return self.access_token

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests."""
        token = self.authenticate()
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}
