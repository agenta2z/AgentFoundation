# pyre-strict
"""RovoChat Streaming Inferencer.

Wraps the RovoChat REST API as an async-native ``StreamingInferencerBase``
implementation with NDJSON streaming and auto-continuation support.

Sends natural language queries to Atlassian's RovoChat conversational AI
service, creates conversations automatically, and streams responses as
text deltas.

Runtime Dependencies:
    Requires ``httpx`` for HTTP requests. This is a soft dependency —
    the module imports successfully without it. ``RuntimeError`` is raised
    only when ``_ainfer_streaming()`` is called without ``httpx`` installed.

    Optionally requires ``atlassian-jwt-auth`` for ASAP token generation.
    Not needed if a pre-generated UCT token is provided.

Usage::

    # Basic usage with UCT token:
    inferencer = RovoChatInferencer(
        cloud_id="my-cloud-id",
        uct_token="my-uct-token",
    )
    result = inferencer("What is Atlassian Rovo?")

    # Streaming:
    async for chunk in inferencer.ainfer_streaming("Explain Rovo agents"):
        print(chunk, end="", flush=True)

    # Multi-turn conversation:
    r1 = inferencer.new_session("My project uses Jira and Confluence")
    r2 = inferencer("What integrations are available for my tools?")

    # With ASAP authentication:
    inferencer = RovoChatInferencer(
        cloud_id="my-cloud-id",
        asap_issuer="my-service",
        asap_private_key="-----BEGIN RSA PRIVATE KEY-----...",
        asap_key_id="my-key-id",
    )

    # With a specific Rovo agent:
    inferencer = RovoChatInferencer(
        cloud_id="my-cloud-id",
        uct_token="...",
        agent_named_id="my-agent-uuid",
    )

    # With a custom base URL (default is staging):
    inferencer = RovoChatInferencer(
        base_url="https://convo-ai.us-east-1.staging.atl-paas.net",
        cloud_id="my-cloud-id",
        uct_token="...",
    )
"""

import asyncio
import logging
import uuid as uuid_mod
from typing import Any, AsyncIterator, Optional

from attr import attrib, attrs

from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
    SDKInferencerResponse,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.auth import (
    RovoChatAuth,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.client import (
    RovoChatClient,
)
from rich_python_utils.common_utils.map_helper import get__

from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.common import (
    AUTO_CONTINUE_REPLY,
    DEFAULT_BASE_URL,
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_TOTAL_TIMEOUT,
    ENV_BASE_URL,
    ENV_CLOUD_ID,
    ENV_FALLBACK_BASE_URL,
    MAX_CONTINUATIONS,
    extract_text_from_event,
    is_terminal_event,
    needs_continuation,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.types import (
    RovoChatConfig,
)

logger: logging.Logger = logging.getLogger(__name__)


@attrs
class RovoChatInferencer(StreamingInferencerBase):
    """RovoChat as an async-native streaming inferencer with auto-continuation.

    Sends natural language queries to the RovoChat REST API and streams
    responses via NDJSON parsing. Creates conversations automatically on
    first use and supports multi-turn conversations via session management.

    Inherits from ``StreamingInferencerBase`` which provides:

    - ``ainfer_streaming()`` with idle timeout and optional caching
    - ``infer_streaming()`` sync bridge via thread + queue
    - Session management: ``new_session``, ``anew_session``, ``resume_session``,
      ``aresume_session``
    - ``active_session_id`` property

    This class implements ``_ainfer_streaming()`` (the abstract primitive) and
    overrides ``_ainfer()`` to support session kwargs and ``SDKInferencerResponse``.

    Usage Patterns::

        # Single query:
        inferencer = RovoChatInferencer(
            cloud_id="...", uct_token="..."
        )
        result = inferencer("What is Rovo?")

        # Streaming:
        async for chunk in inferencer.ainfer_streaming("Explain this"):
            print(chunk, end="", flush=True)

        # Multi-turn with auto-resume:
        r1 = inferencer.new_session("I'm working on project X")
        r2 = inferencer("What tools should I use?")  # Auto-resumes!

    Attributes:
        base_url: Base URL of the RovoChat API.
        cloud_id: Atlassian Cloud ID for the target tenant.
        uct_token: Pre-generated User-Context Token for authentication.
        asap_issuer: ASAP token issuer (alternative to UCT).
        asap_private_key: ASAP RSA private key (alternative to UCT).
        asap_key_id: ASAP key identifier (alternative to UCT).
        agent_named_id: Named identifier for the recipient Rovo agent.
        agent_id: Agent ID for routing to a specific agent.
        lanyard_config: Lanyard configuration ID for authorization.
        store_message: Whether to persist messages server-side.
        citations_enabled: Whether to request citations in responses.
        auto_continue: If True, auto-reply when agent asks clarification.
        max_continuations: Max auto-continue follow-ups per query.
        total_timeout_seconds: Max total time for entire operation.
        idle_timeout_seconds: Max idle time between chunks.
    """

    # === Connection Configuration ===
    # Defaults fall back to ROVOCHAT_BASE_URL / ROVOCHAT_CLOUD_ID env vars,
    # then to JIRA_URL (stripping any /browse path).
    base_url: str = attrib(default="")
    cloud_id: str = attrib(default="")

    # === Authentication (one of Basic Auth, UCT, or ASAP must be provided) ===
    email: Optional[str] = attrib(default=None)
    api_token: Optional[str] = attrib(default=None)
    uct_token: Optional[str] = attrib(default=None)
    asap_issuer: Optional[str] = attrib(default=None)
    asap_private_key: Optional[str] = attrib(default=None)
    asap_key_id: Optional[str] = attrib(default=None)

    # === Chat Configuration ===
    agent_named_id: str = attrib(default="")
    agent_id: str = attrib(default="")
    lanyard_config: str = attrib(default="")
    product: str = attrib(default="rovo")
    experience_id: str = attrib(default="ai-mate")
    store_message: bool = attrib(default=True)
    citations_enabled: bool = attrib(default=True)

    # === Behavior ===
    auto_continue: bool = attrib(default=True)
    max_continuations: int = attrib(default=MAX_CONTINUATIONS)
    total_timeout_seconds: int = attrib(default=DEFAULT_TOTAL_TIMEOUT)
    idle_timeout_seconds: int = attrib(default=DEFAULT_IDLE_TIMEOUT)

    # === Internal State ===
    _conversation_id: Optional[str] = attrib(default=None, init=False, repr=False)
    _last_token_count: int = attrib(default=0, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self._resolve_connection_from_env()
        auth_mode = "Basic" if (self.email and self.api_token) else ("UCT" if self.uct_token else ("ASAP" if self.asap_issuer else "none"))
        self.log_info(
            f"base_url={self.base_url}, cloud_id={self.cloud_id}, "
            f"auth={auth_mode}, agent_named_id={self.agent_named_id or '(default)'}, "
            f"auto_continue={self.auto_continue}, "
            f"idle_timeout={self.idle_timeout_seconds}s, "
            f"total_timeout={self.total_timeout_seconds}s",
            "Config",
        )

    # === Internal Helpers ===

    def _resolve_connection_from_env(self) -> None:
        """Fill in missing connection config from environment variables.

        Checks ``ROVOCHAT_BASE_URL`` / ``ROVOCHAT_CLOUD_ID`` first,
        then falls back to ``JIRA_URL`` (extracting the site base URL).
        If no base_url is resolved, defaults to ``DEFAULT_BASE_URL``.
        """
        import os

        if not self.base_url:
            raw_url = get__(os.environ, ENV_BASE_URL, *ENV_FALLBACK_BASE_URL, default="")
            if raw_url:
                # Strip /browse or trailing paths: "https://x.atlassian.net/browse/PROJ" -> "https://x.atlassian.net"
                from urllib.parse import urlparse

                parsed = urlparse(raw_url)
                if parsed.scheme and parsed.netloc:
                    self.base_url = f"{parsed.scheme}://{parsed.netloc}"

        if not self.base_url:
            self.base_url = DEFAULT_BASE_URL

        if not self.cloud_id:
            self.cloud_id = get__(os.environ, ENV_CLOUD_ID, default="")

    def _create_auth(self) -> RovoChatAuth:
        """Create an authentication manager from configured credentials."""
        return RovoChatAuth(
            email=self.email,
            api_token=self.api_token,
            uct_token=self.uct_token,
            asap_issuer=self.asap_issuer,
            asap_private_key=self.asap_private_key,
            asap_key_id=self.asap_key_id,
        )

    def _create_config(self, auth: RovoChatAuth) -> RovoChatConfig:
        """Create an API configuration from inferencer attributes.

        Auto-detects gateway mode: if Basic Auth is the resolved auth
        mode AND the base_url looks like an Atlassian site (contains
        ``.atlassian.net`` or ``.jira-dev.com``), gateway mode is
        enabled automatically.

        Args:
            auth: The resolved auth object (needed to check auth mode
                after env var resolution).
        """
        use_gateway = bool(
            auth.auth_mode == "basic"
            and any(
                domain in self.base_url
                for domain in (".atlassian.net", ".jira-dev.com", ".atlassian.com")
            )
        )
        return RovoChatConfig(
            base_url=self.base_url,
            cloud_id=self.cloud_id,
            lanyard_config=self.lanyard_config,
            product=self.product,
            experience_id=self.experience_id,
            store_message=self.store_message,
            citations_enabled=self.citations_enabled,
            use_gateway=use_gateway,
        )

    def _create_client(self) -> RovoChatClient:
        """Create a fresh HTTP client for this call.

        Creates auth first, then config (which needs to know the auth
        mode for gateway auto-detection).
        """
        auth = self._create_auth()
        config = self._create_config(auth)
        return RovoChatClient(config=config, auth=auth)

    # === Streaming Primitive ===

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield text deltas by streaming NDJSON from RovoChat.

        Creates a ``RovoChatClient`` per call. Creates a new conversation
        (or reuses an existing one), sends the message, and yields text
        deltas as NDJSON events arrive.

        Supports auto-continuation: if the agent asks a clarification
        question, automatically replies and continues streaming.

        Args:
            prompt: The prompt string.
            **kwargs: Additional arguments:
                - conversation_id: Conversation ID for multi-turn resume.
                - agent_named_id: Override agent for this call.
                - agent_id: Override agent ID for this call.

        Yields:
            Text deltas as they arrive from RovoChat.
        """
        client = self._create_client()

        conv_id: Optional[str] = kwargs.get("conversation_id")
        call_agent_named_id: str = kwargs.get("agent_named_id", self.agent_named_id)
        call_agent_id: str = kwargs.get("agent_id", self.agent_id)

        # Create conversation if needed
        if not conv_id:
            conv_info = await client.create_conversation(
                agent_id=call_agent_id or None,
            )
            conv_id = conv_info.conversation_id
            self.log_info(
                f"Created conversation: id={conv_id}",
                "ConversationCreated",
            )
        else:
            self.log_info(
                f"Resuming conversation: id={conv_id}",
                "ConversationResumed",
            )

        # Send message and stream response
        accumulated_text = ""
        continuations_sent = 0

        async def _stream_response(message: str) -> AsyncIterator[str]:
            """Stream a single message exchange, yielding text deltas.

            Handles two streaming patterns:
            - **Incremental**: Each event contains a new chunk of text
              (typical for NDJSON streaming). Each chunk is yielded directly.
            - **Accumulated**: Each event contains the full text so far
              (typical for polling). Delta is computed by slicing off the
              previously seen prefix.

            The heuristic: if the new text starts with the current
            accumulated text, it's accumulated mode; otherwise it's a
            fresh incremental chunk.
            """
            nonlocal accumulated_text

            async for event in client.send_message_stream(
                conversation_id=conv_id,  # pyre-ignore[6]
                text=message,
                agent_named_id=call_agent_named_id,
                agent_id=call_agent_id,
            ):
                text = extract_text_from_event(event)
                if text:
                    if accumulated_text and text.startswith(accumulated_text):
                        # Accumulated mode: text grows monotonically
                        delta = text[len(accumulated_text):]
                    else:
                        # Incremental mode: each event is a new chunk
                        delta = text

                    if delta:
                        yield delta
                        accumulated_text += delta

                if is_terminal_event(event):
                    self.log_info(
                        f"Terminal event: type={event.event_type} "
                        f"({len(accumulated_text)} chars accumulated)",
                        "StreamComplete",
                    )
                    break

        # Initial message exchange
        async for delta in _stream_response(prompt):
            yield delta
            self._last_token_count += len(delta)

        # Auto-continuation loop
        while (
            self.auto_continue
            and needs_continuation(accumulated_text)
            and continuations_sent < self.max_continuations
        ):
            continuations_sent += 1
            self.log_info(
                f"Auto-continuing (turn {continuations_sent}/{self.max_continuations})",
                "AutoContinue",
            )

            async for delta in _stream_response(AUTO_CONTINUE_REPLY):
                yield delta
                self._last_token_count += len(delta)

        # Save conversation ID for session management
        self._conversation_id = conv_id
        self._session_id = conv_id

    # === Overrides ===

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Override for session management and SDKInferencerResponse support.

        Resolves session kwargs (``new_session``, ``session_id``,
        ``auto_resume``) into ``conversation_id`` and injects it into
        kwargs BEFORE calling ``super()._ainfer()``.

        Args:
            inference_input: Input for inference (string).
            inference_config: Optional configuration (unused).
            **kwargs: Additional arguments:
                - return_sdk_response: If True, return SDKInferencerResponse.
                - session_id: Conversation ID to resume.
                - new_session: If True, forces a new conversation.

        Returns:
            Response text string, or ``SDKInferencerResponse`` if
            ``return_sdk_response=True``.
        """
        new_session = kwargs.pop("new_session", False)
        explicit_session_id = kwargs.pop("session_id", None)
        return_sdk_response = kwargs.pop("return_sdk_response", False)

        if new_session:
            kwargs["conversation_id"] = None
            logger.debug("Starting new conversation (new_session=True)")
        elif explicit_session_id:
            kwargs["conversation_id"] = explicit_session_id
            logger.debug(
                "Resuming conversation: %s",
                explicit_session_id[:8] if explicit_session_id else None,
            )
        elif self.auto_resume and self._conversation_id:
            kwargs["conversation_id"] = self._conversation_id
            logger.debug(
                "Auto-resuming conversation: %s",
                self._conversation_id[:8] if self._conversation_id else None,
            )
        else:
            kwargs["conversation_id"] = None
            logger.debug("Starting fresh conversation (no previous session)")

        self._last_token_count = 0
        response_text = await super()._ainfer(
            inference_input, inference_config, **kwargs
        )

        if return_sdk_response:
            return SDKInferencerResponse(
                content=response_text,
                session_id=self._session_id,
                tokens_received=self._last_token_count,
            )
        return response_text

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Sync wrapper for async ``_ainfer()``.

        Uses ``_run_async()`` from the common utils helper. Note: cannot be
        called from an async context (would raise ``RuntimeError``). Use
        ``_ainfer()`` or ``ainfer_streaming()`` directly in async code.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration.
            **kwargs: Additional arguments passed to ``_ainfer()``.

        Returns:
            Response text string, or ``SDKInferencerResponse`` if
            ``return_sdk_response=True``.
        """
        from rich_python_utils.common_utils.async_function_helper import _run_async

        return _run_async(self._ainfer(inference_input, inference_config, **kwargs))
