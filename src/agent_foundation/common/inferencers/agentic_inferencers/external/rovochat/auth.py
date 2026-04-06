# pyre-strict
from __future__ import annotations

"""RovoChat Inferencer — authentication utilities.

Provides ``RovoChatAuth`` for managing authentication tokens when calling
the RovoChat API. Supports:

1. **Direct UCT token** — pass a pre-generated User-Context Token.
2. **ASAP token generation** — generate JWT tokens using ``atlassian-jwt-auth``
   (soft dependency; ``ImportError`` is raised only when ASAP generation is
   attempted without the library installed).
3. **Environment variable fallback** — reads from ``ROVOCHAT_UCT_TOKEN``,
   ``ROVOCHAT_ASAP_ISSUER``, ``ROVOCHAT_ASAP_PRIVATE_KEY``, ``ROVOCHAT_ASAP_KEY_ID``.

Usage::

    # Direct token
    auth = RovoChatAuth(uct_token="my-uct-token")

    # ASAP-based
    auth = RovoChatAuth(
        asap_issuer="my-service",
        asap_private_key="-----BEGIN RSA PRIVATE KEY-----...",
        asap_key_id="my-key-id",
    )

    # From environment
    auth = RovoChatAuth()  # Reads ROVOCHAT_UCT_TOKEN or ASAP env vars

    headers = auth.get_auth_headers()
"""

import base64
import json
import logging
import os
import time
from typing import Optional

from rich_python_utils.common_utils.map_helper import get__

from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.common import (
    ENV_API_TOKEN,
    ENV_ASAP_AUDIENCE,
    ENV_ASAP_ISSUER,
    ENV_ASAP_KEY_ID,
    ENV_ASAP_PRIVATE_KEY,
    ENV_EMAIL,
    ENV_FALLBACK_API_TOKEN,
    ENV_FALLBACK_EMAIL,
    ENV_UCT_TOKEN,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.exceptions import (
    RovoChatAuthError,
)

logger: logging.Logger = logging.getLogger(__name__)

# Default ASAP audience (the service we're calling)
DEFAULT_ASAP_AUDIENCE: str = "conversational-ai-platform"

# Token cache TTL in seconds (refresh before expiry)
_TOKEN_CACHE_TTL: int = 30


class RovoChatAuth:
    """Manages authentication for RovoChat API calls.

    Supports three authentication modes:

    1. **Basic Auth** (email + API token) — simplest, works via the
       Atlassian gateway (e.g., ``https://hello.atlassian.net/gateway/...``).
       Get an API token from https://id.atlassian.com/manage-profile/security/api-tokens
    2. **Direct UCT token** — pass a pre-generated User-Context Token.
    3. **ASAP token generation** — generate JWT tokens using ``atlassian-jwt-auth``
       (soft dependency; only needed when this mode is used).

    Priority order:
    1. Basic Auth (``email`` + ``api_token``)
    2. Direct ``uct_token``
    3. ASAP token generation
    4. Environment variables as fallback

    Attributes:
        email: Atlassian account email for Basic Auth.
        api_token: Atlassian API token for Basic Auth.
        uct_token: Pre-generated User-Context Token.
        asap_issuer: ASAP token issuer identifier.
        asap_private_key: ASAP RSA private key (PEM format or data URI).
        asap_key_id: ASAP key identifier.
        asap_audience: ASAP token audience (target service).
    """

    def __init__(
        self,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        uct_token: Optional[str] = None,
        asap_issuer: Optional[str] = None,
        asap_private_key: Optional[str] = None,
        asap_key_id: Optional[str] = None,
        asap_audience: Optional[str] = None,
    ) -> None:
        self.email = email
        self.api_token = api_token
        self.uct_token = uct_token
        self.asap_issuer = asap_issuer
        self.asap_private_key = asap_private_key
        self.asap_key_id = asap_key_id
        self.asap_audience = asap_audience or DEFAULT_ASAP_AUDIENCE

        # Token cache
        self._cached_token: Optional[str] = None
        self._cached_token_time: float = 0.0

        # Resolve from environment if not provided
        self._resolve_from_env()

    @property
    def auth_mode(self) -> str:
        """Return the active authentication mode as a string."""
        if self.email and self.api_token:
            return "basic"
        if self.uct_token:
            return "uct"
        if self.asap_issuer and self.asap_private_key:
            return "asap"
        return "none"

    def _resolve_from_env(self) -> None:
        """Fill in missing credentials from environment variables.

        Checks ``ROVOCHAT_*`` vars first, then falls back to common
        Atlassian credential env vars (``JIRA_EMAIL``, ``JIRA_API_TOKEN``,
        ``ATLASSIAN_EMAIL``, ``ATLASSIAN_API_TOKEN``).
        """
        if not self.email:
            self.email = get__(os.environ, ENV_EMAIL, *ENV_FALLBACK_EMAIL, default=None)
        if not self.api_token:
            self.api_token = get__(os.environ, ENV_API_TOKEN, *ENV_FALLBACK_API_TOKEN, default=None)

        if not self.uct_token:
            self.uct_token = os.environ.get(ENV_UCT_TOKEN)

        if not self.asap_issuer:
            self.asap_issuer = os.environ.get(ENV_ASAP_ISSUER)
        if not self.asap_private_key:
            self.asap_private_key = os.environ.get(ENV_ASAP_PRIVATE_KEY)
        if not self.asap_key_id:
            self.asap_key_id = os.environ.get(ENV_ASAP_KEY_ID)
        if not self.asap_audience or self.asap_audience == DEFAULT_ASAP_AUDIENCE:
            env_audience = os.environ.get(ENV_ASAP_AUDIENCE)
            if env_audience:
                self.asap_audience = env_audience

    def get_token(self) -> str:
        """Return a valid authentication token (for UCT/ASAP modes).

        Returns the direct UCT token if available, otherwise generates
        an ASAP JWT token. Not used in Basic Auth mode.

        Returns:
            A token string suitable for the ``User-Context`` header.

        Raises:
            RovoChatAuthError: If no valid credentials are available.
        """
        # Direct UCT token
        if self.uct_token:
            return self.uct_token

        # ASAP token generation
        if self.asap_issuer and self.asap_private_key:
            return self._get_asap_token()

        raise RovoChatAuthError(
            "No authentication credentials available. Provide one of:\n"
            f"  - Basic Auth (email + api_token, or {ENV_EMAIL} + {ENV_API_TOKEN} env vars)\n"
            f"  - A UCT token (param or {ENV_UCT_TOKEN} env var)\n"
            f"  - ASAP credentials (params or {ENV_ASAP_ISSUER}, "
            f"{ENV_ASAP_PRIVATE_KEY}, {ENV_ASAP_KEY_ID} env vars)"
        )

    def get_auth_headers(self) -> dict[str, str]:
        """Return HTTP headers for RovoChat API authentication.

        Returns headers appropriate for the active auth mode:

        - **Basic Auth**: ``Authorization: Basic base64(email:token)``
        - **UCT/ASAP**: ``Authorization: Bearer <token>`` +
          ``User-Context: <token>``

        Returns:
            Dictionary of authentication headers.
        """
        # Basic Auth (highest priority — simplest, works via gateway)
        if self.email and self.api_token:
            b64 = base64.b64encode(
                f"{self.email}:{self.api_token}".encode()
            ).decode()
            return {"Authorization": f"Basic {b64}"}

        # UCT / ASAP token
        token = self.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "User-Context": token,
        }

    def _get_asap_token(self) -> str:
        """Generate or return cached ASAP JWT token.

        Uses TTL-based caching to avoid generating a new token on every call.

        Returns:
            ASAP JWT token string.

        Raises:
            RovoChatAuthError: If token generation fails.
        """
        now = time.time()
        if self._cached_token and (now - self._cached_token_time) < _TOKEN_CACHE_TTL:
            return self._cached_token

        token = self._generate_asap_token()
        self._cached_token = token
        self._cached_token_time = now
        return token

    def _generate_asap_token(self) -> str:
        """Generate an ASAP JWT token using atlassian-jwt-auth.

        Supports multiple private key formats:
        - PEM (``-----BEGIN RSA PRIVATE KEY-----``)
        - PKCS8 data URI (``data:application/pkcs8;base64,...``)
        - Generic data URI

        Returns:
            JWT token string.

        Raises:
            RovoChatAuthError: If the atlassian-jwt-auth library is not
                installed or token generation fails.
        """
        try:
            from atlassian_jwt_auth.signer import create_signer
        except ImportError as e:
            raise RovoChatAuthError(
                f"ASAP token generation requires 'atlassian-jwt-auth' package: {e}. "
                "Install it with: pip install atlassian-jwt-auth\n"
                "Alternatively, provide a pre-generated UCT token via "
                f"the uct_token parameter or {ENV_UCT_TOKEN} env var."
            ) from e

        try:
            private_key = self.asap_private_key
            key_id = self.asap_key_id
            issuer = self.asap_issuer

            if not private_key or not issuer:
                raise RovoChatAuthError(
                    "ASAP credentials incomplete: need asap_issuer and asap_private_key."
                )

            # Handle PKCS8 data URI format
            if private_key.startswith("data:application/pkcs8;base64,"):
                pem_key = _convert_pkcs8_data_uri_to_pem(private_key)
                signer = create_signer(issuer, key_id, pem_key)
            elif private_key.startswith("data:"):
                # Generic data URI — let atlassian-jwt-auth handle it
                from atlassian_jwt_auth.key import DataUriPrivateKeyRetriever

                retriever = DataUriPrivateKeyRetriever(private_key)
                loaded_key_id, loaded_key = retriever.load(issuer)
                signer = create_signer(issuer, loaded_key_id, loaded_key)
            else:
                # Direct PEM format
                signer = create_signer(issuer, key_id, private_key)

            jwt_token = signer.generate_jwt(self.asap_audience)
            token_str = (
                jwt_token.decode("utf-8")
                if isinstance(jwt_token, bytes)
                else str(jwt_token)
            )

            logger.debug(
                "Generated ASAP token for issuer=%s, audience=%s (length=%d)",
                issuer,
                self.asap_audience,
                len(token_str),
            )
            return token_str

        except RovoChatAuthError:
            raise
        except Exception as e:
            raise RovoChatAuthError(
                f"ASAP token generation failed: {e}",
                details={"issuer": self.asap_issuer, "audience": self.asap_audience},
            ) from e

    @property
    def has_credentials(self) -> bool:
        """Check if any authentication credentials are available."""
        return bool(
            (self.email and self.api_token)
            or self.uct_token
            or (self.asap_issuer and self.asap_private_key)
        )

    def __repr__(self) -> str:
        mode = self.auth_mode
        if mode == "basic":
            return f"RovoChatAuth(basic={self.email})"
        elif mode == "uct":
            return f"RovoChatAuth(uct_token=...{self.uct_token[-8:]})"
        elif mode == "asap":
            return f"RovoChatAuth(asap_issuer={self.asap_issuer})"
        return "RovoChatAuth(no_credentials)"


def _convert_pkcs8_data_uri_to_pem(data_uri: str) -> str:
    """Convert a PKCS8 data URI to PEM format.

    Args:
        data_uri: A string starting with ``data:application/pkcs8;base64,``.

    Returns:
        PEM-formatted private key string.
    """
    prefix = "data:application/pkcs8;base64,"
    b64_data = data_uri[len(prefix) :]
    der_bytes = base64.b64decode(b64_data)
    b64_pem = base64.b64encode(der_bytes).decode("ascii")

    # Split into 64-character lines
    lines = [b64_pem[i : i + 64] for i in range(0, len(b64_pem), 64)]
    pem_body = "\n".join(lines)

    return f"-----BEGIN PRIVATE KEY-----\n{pem_body}\n-----END PRIVATE KEY-----\n"
