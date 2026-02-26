"""
Generic LLM Chat Service for Dash applications.

This module provides a conversation-aware LLM service wrapper that can be used
with any LLM API that supports the message format pattern (system/user/assistant roles).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class BaseLLMConfig:
    """
    Base configuration for LLM services.

    Attributes:
        system_prompt: Optional system prompt to set context for the conversation.
        temperature: Controls randomness in responses (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum number of tokens to generate.
        model_name: Name/identifier of the model to use.
    """

    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    model_name: str = ""
    extra_params: Dict[str, Any] = field(default_factory=dict)


class LLMChatService:
    """
    A generic, conversation-aware LLM chat service.

    This service wraps any LLM API function and provides:
    - Conversation history management
    - System prompt injection
    - Message format conversion (from internal format to LLM API format)
    - Error handling with user-friendly messages

    The service is designed to be plugged into DashInteractiveApp as a message handler,
    bridging the gap between the Dash UI and LLM backends.

    Example:
        >>> from meta_modeling_tools.apis.metagen_llm import generate_text, MetaGenModels
        >>>
        >>> # Create service with Metagen backend
        >>> service = LLMChatService(
        ...     llm_func=generate_text,
        ...     config=BaseLLMConfig(
        ...         model_name=MetaGenModels.CLAUDE_4_SONNET,
        ...         system_prompt="You are a helpful assistant.",
        ...         temperature=0.7
        ...     )
        ... )
        >>>
        >>> # Use as message handler
        >>> app = DashInteractiveApp(title="Chatbot")
        >>> app.set_message_handler(lambda msg: service.process_message(msg, history=[]))
    """

    def __init__(
        self,
        llm_func: Callable[..., str],
        config: Optional[BaseLLMConfig] = None,
        message_key: str = "prompt_or_messages",
        model_key: str = "model",
        temperature_key: str = "temperature",
        max_tokens_key: str = "max_new_tokens",
    ):
        """
        Initialize the LLM chat service.

        Args:
            llm_func: The LLM API function to call. Should accept messages and return a string.
                     Example: meta_modeling_tools.apis.metagen_llm.generate_text
            config: Configuration object with model settings.
            message_key: Parameter name for messages in the LLM function.
            model_key: Parameter name for model in the LLM function.
            temperature_key: Parameter name for temperature in the LLM function.
            max_tokens_key: Parameter name for max tokens in the LLM function.
        """
        self.llm_func = llm_func
        self.config = config or BaseLLMConfig()
        self.message_key = message_key
        self.model_key = model_key
        self.temperature_key = temperature_key
        self.max_tokens_key = max_tokens_key

    def _convert_history_to_messages(
        self, history: List[Dict[str, Any]], include_system_prompt: bool = True
    ) -> List[Dict[str, str]]:
        """
        Convert internal message history to LLM API message format.

        Internal format: [{"role": "user/assistant", "content": "...", "timestamp": "..."}]
        LLM API format: [{"role": "system/user/assistant", "content": "..."}]

        Args:
            history: List of messages in internal format (with optional extra fields).
            include_system_prompt: Whether to prepend the system prompt.

        Returns:
            List of messages in LLM API format.
        """
        messages = []

        if include_system_prompt and self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        for msg in history:
            messages.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )

        return messages

    def process_message(
        self, user_message: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Process a user message and return an LLM response.

        This is the main entry point for handling messages. It:
        1. Converts history to API format
        2. Appends the new user message
        3. Calls the LLM API
        4. Returns the response (or error message)

        Args:
            user_message: The new message from the user.
            history: Previous conversation history in internal format.
                    If None, starts a new conversation.

        Returns:
            The LLM's response string, or an error message if the call fails.
        """
        try:
            history = history or []
            messages = self._convert_history_to_messages(history)
            messages.append({"role": "user", "content": user_message})

            kwargs = {
                self.message_key: messages,
                self.temperature_key: self.config.temperature,
            }

            if self.config.model_name:
                kwargs[self.model_key] = self.config.model_name

            if self.config.max_tokens:
                kwargs[self.max_tokens_key] = self.config.max_tokens

            kwargs.update(self.config.extra_params)

            logger.debug(f"Calling LLM with {len(messages)} messages")
            response = self.llm_func(**kwargs)

            return response

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"⚠️ Error: Unable to get response from the model. {str(e)}"

    def process_with_full_history(
        self, user_message: str, history: List[Dict[str, Any]]
    ) -> str:
        """
        Process a message with the full conversation history.

        This is an alias for process_message that makes the intent clearer
        when you want to include the full conversation context.

        Args:
            user_message: The new message from the user.
            history: Full conversation history from the session.

        Returns:
            The LLM's response string.
        """
        return self.process_message(user_message, history)

    def create_handler(
        self, get_history_func: Optional[Callable[[], List[Dict[str, Any]]]] = None
    ) -> Callable[[str], str]:
        """
        Create a simple message handler function suitable for DashInteractiveApp.

        This method returns a callable that can be passed directly to
        DashInteractiveApp.set_message_handler() or the constructor.

        Args:
            get_history_func: Optional function that returns current conversation history.
                            If None, the handler will process messages without history context.

        Returns:
            A callable that takes a message string and returns a response string.

        Example:
            >>> service = LLMChatService(llm_func=generate_text, config=config)
            >>> handler = service.create_handler()
            >>> app = DashInteractiveApp(message_handler=handler)
        """
        if get_history_func:

            def handler_with_history(message: str) -> str:
                history = get_history_func()
                return self.process_message(message, history)

            return handler_with_history
        else:

            def handler_no_history(message: str) -> str:
                return self.process_message(message, history=[])

            return handler_no_history

    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update.
                     Valid keys: system_prompt, temperature, max_tokens, model_name, extra_params
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")


class MockLLMService(LLMChatService):
    """
    A mock LLM service for testing and development.

    Returns canned responses based on simple keyword matching,
    useful for UI development without actual LLM API calls.
    """

    def __init__(self, config: Optional[BaseLLMConfig] = None):
        """Initialize with a mock LLM function."""
        super().__init__(
            llm_func=self._mock_llm,
            config=config or BaseLLMConfig(model_name="mock-model"),
        )

    def _mock_llm(self, **kwargs) -> str:
        """Generate mock responses for testing."""
        messages = kwargs.get(self.message_key, [])
        if not messages:
            return "Hello! How can I help you today?"

        last_message = messages[-1].get("content", "").lower()

        if "hello" in last_message or "hi" in last_message:
            return "Hello! I'm a mock assistant. How can I help you today?"
        elif "help" in last_message:
            return "I'm here to help! This is a mock response for testing purposes."
        elif "bye" in last_message or "goodbye" in last_message:
            return "Goodbye! Have a great day!"
        else:
            return f"Mock response to: {messages[-1].get('content', 'your message')}"
