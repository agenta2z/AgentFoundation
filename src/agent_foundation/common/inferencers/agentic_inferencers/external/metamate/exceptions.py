# pyre-strict
"""
Metamate Integration Exceptions.

This module defines exception types for the Metamate integration,
providing specific error handling for different failure modes.
"""


class MetamateError(Exception):
    """Base exception for Metamate integration errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class MetamateConnectionError(MetamateError):
    """Raised when connection to Metamate service fails."""

    pass


class MetamateUnavailableError(MetamateError):
    """Raised when Metamate service is unavailable."""

    def __init__(
        self,
        message: str = "Metamate service is unavailable",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)


class MetamateTimeoutError(MetamateError):
    """Raised when a Metamate operation times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.timeout_seconds = timeout_seconds


class MetamateToolError(MetamateError):
    """Raised when a Metamate tool execution fails."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        parameters: dict[str, object] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.tool_name = tool_name
        self.parameters = parameters or {}


class MetamateSecurityError(MetamateError):
    """Raised when a security policy violation is detected."""

    def __init__(
        self,
        message: str,
        permission_required: str,
        operation: str,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.permission_required = permission_required
        self.operation = operation


class MetamateValidationError(MetamateError):
    """Raised when validation of inputs or outputs fails."""

    def __init__(
        self,
        message: str,
        validation_errors: list[str] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.validation_errors = validation_errors or []


class ResearchError(MetamateError):
    """Raised when a research operation fails."""

    def __init__(
        self,
        message: str,
        research_plan_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.research_plan_id = research_plan_id


class KnowledgeSearchError(MetamateError):
    """Raised when a knowledge search operation fails."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        query: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.source = source
        self.query = query


class FallbackActivatedError(MetamateError):
    """Raised when fallback is activated due to primary service failure."""

    def __init__(
        self,
        message: str,
        primary_error: Exception,
        fallback_service: str,
    ) -> None:
        super().__init__(message, cause=primary_error)
        self.primary_error = primary_error
        self.fallback_service = fallback_service
