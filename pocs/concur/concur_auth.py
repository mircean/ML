"""
SAP Concur OAuth 2.0 authentication module using Authorization Code flow with PKCE.
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class ConcurAuthenticator:
    """Handles OAuth 2.0 authentication with SAP Concur API using PKCE flow."""

    class _OAuthCallbackHandler(BaseHTTPRequestHandler):
        """HTTP handler for OAuth2 callback."""

        # Class variables to store the authorization code and state
        authorization_code: Optional[str] = None
        expected_state: Optional[str] = None
        error: Optional[str] = None

        def log_message(self, format, *args):
            """Suppress default HTTP server logging."""
            pass  # Silent by default; use logger if needed

        def do_GET(self):
            """Handle GET request to callback endpoint."""
            if not self.path.startswith("/callback"):
                self.send_error(404, "Not Found")
                return

            # Parse query parameters
            parsed = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed.query)

            # Check for error response
            if "error" in query_params:
                error_msg = query_params.get("error", ["unknown"])[0]
                error_desc = query_params.get("error_description", [""])[0]
                ConcurAuthenticator._OAuthCallbackHandler.error = f"{error_msg}: {error_desc}"
                logger.error(f"OAuth error: {ConcurAuthenticator._OAuthCallbackHandler.error}")

                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Authentication failed: {error_msg}".encode())
                return

            # Extract authorization code
            code = query_params.get("code", [None])[0]
            state = query_params.get("state", [None])[0]

            if not code:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing authorization code")
                return

            # Verify state parameter (CSRF protection)
            if state != ConcurAuthenticator._OAuthCallbackHandler.expected_state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid state parameter")
                return

            # Store authorization code
            ConcurAuthenticator._OAuthCallbackHandler.authorization_code = code
            logger.info("Authorization code received successfully")

            # Send success response
            self.send_response(200)
            self.end_headers()
            html = b"""
            <html>
            <head><title>Authentication Successful</title></head>
            <body>
                <h1>Success!</h1>
                <p>You can close this window and return to the application.</p>
            </body>
            </html>
            """
            self.wfile.write(html)

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        base_url: str,
        redirect_uri: str = "http://localhost:53682/callback",
        scope: str = "openid profile user.read identity.user.ids.read expense.report.read",
    ):
        """
        Initialize Concur authenticator with OAuth 2.0 PKCE flow.

        Args:
            client_id: OAuth client ID from Concur app registration
            client_secret: OAuth client secret from Concur app registration
            base_url: Concur API base URL (e.g., https://us2.api.concursolutions.com)
            redirect_uri: OAuth callback URI (must match app registration)
            scope: Space-separated OAuth scopes
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self.redirect_uri = redirect_uri
        self.scope = scope

        # OAuth endpoints
        self.auth_url = f"{self.base_url}/oauth2/v0/authorize"
        self.token_url = f"{self.base_url}/oauth2/v0/token"

        # Token state
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.user_id: Optional[str] = None  # Cached user ID from JWT

        # Environment file for token persistence
        self.env_path = os.path.join(os.getcwd(), ".env")

        # Load existing tokens from .env
        self._load_tokens_from_env()

    def authenticate(self) -> str:
        """
        Authenticate and return a valid access token.

        Returns:
            Valid access token

        Raises:
            RuntimeError: If authentication fails
        """
        # Check if current token is still valid
        if self._is_token_valid():
            logger.info("Using cached access token")
            return self.access_token

        # Try to refresh if we have a refresh token
        if self.refresh_token:
            logger.info("Attempting to refresh access token")
            try:
                return self._refresh_access_token()
            except Exception as e:
                logger.warning(f"Token refresh failed: {e}. Starting new authentication flow.")
                # Clear invalid tokens
                self.refresh_token = None

        # Start interactive OAuth flow
        logger.info("Starting interactive OAuth 2.0 authentication flow")
        return self._authenticate_with_pkce()

    def _is_token_valid(self) -> bool:
        """Check if current access token is valid and not expiring soon."""
        if not self.access_token or not self.token_expires_at:
            return False
        # Consider token invalid if it expires in less than 60 seconds
        return datetime.now() < self.token_expires_at - timedelta(seconds=60)

    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authorization headers for API requests.

        Returns:
            Dictionary with Authorization and Accept headers
        """
        token = self.authenticate()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }

    def _generate_pkce_pair(self) -> Tuple[str, str]:
        """
        Generate PKCE code verifier and challenge.

        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        # Generate random code verifier (43-128 characters)
        code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')

        # Generate code challenge (SHA256 hash of verifier)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')

        return code_verifier, code_challenge

    def _build_authorization_url(self, state: str, code_challenge: str) -> str:
        """
        Build OAuth2 authorization URL.

        Args:
            state: Random state parameter for CSRF protection
            code_challenge: PKCE code challenge

        Returns:
            Complete authorization URL
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": self.scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{self.auth_url}?{urllib.parse.urlencode(params)}"

    def _start_callback_server_and_get_code(
        self, state: str, code_challenge: str, timeout: int = 300
    ) -> str:
        """
        Start local HTTP server, open browser, and wait for OAuth callback.

        Args:
            state: Random state for CSRF protection
            code_challenge: PKCE code challenge
            timeout: Maximum seconds to wait for callback (default: 5 minutes)

        Returns:
            Authorization code

        Raises:
            TimeoutError: If user doesn't complete authentication in time
            RuntimeError: If OAuth error occurs
        """
        # Reset handler state
        self._OAuthCallbackHandler.authorization_code = None
        self._OAuthCallbackHandler.expected_state = state
        self._OAuthCallbackHandler.error = None

        # Parse port from redirect URI
        parsed_uri = urllib.parse.urlparse(self.redirect_uri)
        port = parsed_uri.port or 53682
        host = "127.0.0.1"

        # Start local HTTP server
        logger.info(f"Starting OAuth callback server on {host}:{port}")
        httpd = HTTPServer((host, port), self._OAuthCallbackHandler)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        try:
            # Build and open authorization URL in browser
            auth_url = self._build_authorization_url(state, code_challenge)
            logger.info(f"Opening browser for authentication: {auth_url}")
            print(f"\n{'='*70}")
            print(f"AUTHENTICATION REQUIRED")
            print(f"{'='*70}")
            print(f"Opening your web browser for Concur authentication...")
            print(f"\nIf the browser doesn't open automatically, visit:")
            print(f"  {auth_url}")
            print(f"{'='*70}\n")

            webbrowser.open(auth_url)

            # Wait for authorization code
            elapsed = 0
            while elapsed < timeout:
                if self._OAuthCallbackHandler.authorization_code:
                    return self._OAuthCallbackHandler.authorization_code

                if self._OAuthCallbackHandler.error:
                    raise RuntimeError(
                        f"OAuth authentication failed: {self._OAuthCallbackHandler.error}"
                    )

                time.sleep(1)
                elapsed += 1

            raise TimeoutError(
                f"Authentication timed out after {timeout} seconds. "
                f"Please try again."
            )

        finally:
            httpd.shutdown()
            logger.info("OAuth callback server stopped")

    def _authenticate_with_pkce(self) -> str:
        """
        Perform interactive OAuth2 authorization code flow with PKCE.

        Returns:
            Access token

        Raises:
            RuntimeError: If authentication fails
        """
        # Generate PKCE parameters
        state = secrets.token_urlsafe(16)
        code_verifier, code_challenge = self._generate_pkce_pair()

        # Get authorization code via browser
        auth_code = self._start_callback_server_and_get_code(state, code_challenge)

        # Exchange authorization code for tokens
        logger.info("Exchanging authorization code for tokens")
        response_data = self._exchange_code_for_tokens(auth_code, code_verifier)

        # Process and save tokens
        return self._process_token_response(response_data)

    def _exchange_code_for_tokens(
        self, code: str, code_verifier: str
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for access and refresh tokens.

        Args:
            code: Authorization code from OAuth callback
            code_verifier: PKCE code verifier

        Returns:
            Token response data

        Raises:
            requests.HTTPError: If token exchange fails
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code_verifier": code_verifier,
        }

        logger.info(f"Exchanging authorization code at {self.token_url}")
        response = requests.post(
            self.token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(
                f"Token exchange failed: {response.status_code} - {response.text}"
            )
            response.raise_for_status()

        return response.json()

    def _refresh_access_token(self) -> str:
        """
        Refresh access token using refresh token.

        Returns:
            New access token

        Raises:
            requests.HTTPError: If token refresh fails
        """
        if not self.refresh_token:
            raise ValueError("No refresh token available")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        logger.info("Refreshing access token")
        response = requests.post(
            self.token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(
                f"Token refresh failed: {response.status_code} - {response.text}"
            )
            response.raise_for_status()

        return self._process_token_response(response.json())

    def _process_token_response(self, response_data: Dict[str, Any]) -> str:
        """
        Process token response and store tokens.

        Args:
            response_data: Token response from OAuth server

        Returns:
            Access token
        """
        self.access_token = response_data["access_token"]

        # Update refresh token if provided
        if "refresh_token" in response_data:
            self.refresh_token = response_data["refresh_token"]
            logger.info("Received new refresh token")

        # Calculate expiration time
        expires_in = response_data.get("expires_in", 3600)
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

        logger.info(f"Access token obtained, expires at {self.token_expires_at}")

        # Extract user ID from JWT
        self.user_id = self._decode_jwt_sub(self.access_token)
        if self.user_id:
            logger.info(f"Extracted user ID from token: {self.user_id}")

        # Save tokens to .env file
        self._save_tokens_to_env()

        return self.access_token

    def _save_tokens_to_env(self):
        """Save tokens to .env file."""
        if not os.path.exists(self.env_path):
            logger.warning(f".env file not found at {self.env_path}, creating new file")

        # Prepare token data
        updates = {
            "CONCUR_ACCESS_TOKEN": self.access_token or "",
            "CONCUR_REFRESH_TOKEN": self.refresh_token or "",
            "CONCUR_ACCESS_TOKEN_EXPIRES_AT": (
                str(int(self.token_expires_at.timestamp()))
                if self.token_expires_at
                else "0"
            ),
        }

        # Update .env file
        for key, value in updates.items():
            self._upsert_env_var(key, value)

        logger.info("Tokens saved to .env file")

    def _upsert_env_var(self, key: str, value: str):
        """
        Update or insert environment variable in .env file.

        Args:
            key: Environment variable name
            value: Environment variable value
        """
        lines = []
        if os.path.exists(self.env_path):
            with open(self.env_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()

        # Find and update existing key, or append new one
        key_prefix = f"{key}="
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(key_prefix):
                lines[i] = f"{key}={value}"
                updated = True
                break

        if not updated:
            lines.append(f"{key}={value}")

        # Write back to file
        with open(self.env_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _load_tokens_from_env(self):
        """Load existing tokens from .env file."""
        access_token = os.getenv("CONCUR_ACCESS_TOKEN")
        refresh_token = os.getenv("CONCUR_REFRESH_TOKEN")
        expires_at_str = os.getenv("CONCUR_ACCESS_TOKEN_EXPIRES_AT", "0")

        if access_token:
            self.access_token = access_token
            logger.info("Loaded access token from environment")

        if refresh_token:
            self.refresh_token = refresh_token
            logger.info("Loaded refresh token from environment")

        try:
            expires_at_timestamp = int(expires_at_str)
            if expires_at_timestamp > 0:
                self.token_expires_at = datetime.fromtimestamp(expires_at_timestamp)
                logger.info(f"Token expires at: {self.token_expires_at}")
        except (ValueError, OSError) as e:
            logger.warning(f"Could not parse token expiration: {e}")

        # Extract user ID from token if available
        if self.access_token:
            self.user_id = self._decode_jwt_sub(self.access_token)

    def _decode_jwt_sub(self, token: str) -> Optional[str]:
        """
        Extract 'sub' (user ID) from JWT access token.

        Args:
            token: JWT access token

        Returns:
            User ID (UUID) or None if decoding fails
        """
        try:
            # JWT format: header.payload.signature
            parts = token.split(".")
            if len(parts) != 3:
                return None

            payload_segment = parts[1]

            # Add padding if needed (JWT uses base64url without padding)
            padding = 4 - (len(payload_segment) % 4)
            if padding != 4:
                payload_segment += "=" * padding

            # Decode payload
            payload_bytes = base64.urlsafe_b64decode(payload_segment)
            payload = json.loads(payload_bytes.decode("utf-8"))

            # Extract and return 'sub' claim
            sub = payload.get("sub")
            if isinstance(sub, str):
                return sub

            return None

        except Exception as e:
            logger.debug(f"Could not decode JWT: {e}")
            return None

    def get_user_id(self) -> str:
        """
        Get current user's UUID (required for ERS v4 API calls).

        Returns:
            User UUID

        Raises:
            RuntimeError: If user ID cannot be determined
        """
        # Ensure we have a valid token
        self.authenticate()

        # Try to get from cached user_id
        if self.user_id:
            return self.user_id

        # Try to decode from current token
        if self.access_token:
            self.user_id = self._decode_jwt_sub(self.access_token)
            if self.user_id:
                logger.info(f"Resolved user ID from JWT: {self.user_id}")
                return self.user_id

        # If we still don't have a user ID, this is an error
        raise RuntimeError(
            "Could not extract user ID from access token. "
            "The token may be invalid or in an unexpected format."
        )
