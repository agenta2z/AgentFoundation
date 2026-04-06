# pyre-strict
"""RovoChat Inferencer — custom exception hierarchy.

Provides structured exceptions for authentication, connection, timeout,
and API-level errors when interacting with the RovoChat service.
"""

from __future__ import annotations

from typing import Dict, Optional


class RovoChatError(Exception):
    """Base exception for all RovoChat inferencer errors.

    Attributes:
        message: Human-readable error description.
        details: Optional additional error context (e.g. response body, status code).
    """

    def __init__(self, message: str, details: Optional[Dict] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


class RovoChatAuthError(RovoChatError):
    """Authentication or authorization failure.

    Raised when:
    - UCT token is missing, expired, or invalid
    - ASAP token generation fails (missing keys, bad format)
    - The server returns 401/403
    """


class RovoChatConnectionError(RovoChatError):
    """Network-level connection failure.

    Raised when:
    - Cannot reach the RovoChat API host
    - DNS resolution fails
    - Connection is refused or reset
    """


class RovoChatTimeoutError(RovoChatError):
    """Operation timed out.

    Raised when:
    - HTTP request times out
    - Streaming response exceeds total timeout
    - Idle timeout between stream chunks is exceeded
    """


class RovoChatAPIError(RovoChatError):
    """API-level error from the RovoChat service.

    Raised when the server returns an error response (4xx/5xx)
    that is not an auth error.

    Attributes:
        status_code: HTTP status code from the server.
    """

    def __init__(
        self, message: str, status_code: int = 0, details: Optional[Dict] = None
    ) -> None:
        self.status_code = status_code
        super().__init__(message, details)

    def __str__(self) -> str:
        base = f"[HTTP {self.status_code}] {self.message}" if self.status_code else self.message
        if self.details:
            return f"{base} | details={self.details}"
        return base
