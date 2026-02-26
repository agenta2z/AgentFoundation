====================================
LLM Integration
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

OpenManus abstracts LLM communication through a unified ``LLM`` class
that supports multiple providers (OpenAI, Azure OpenAI, AWS Bedrock)
with a consistent API. The LLM layer handles message formatting, tool
calling, token counting, retry logic, and streaming.


LLM Client
===========

**File**: ``app/llm.py``

The ``LLM`` class is a singleton-per-config-name wrapper around the
OpenAI SDK client.

Singleton Pattern
-----------------

.. code-block:: python

    class LLM:
        _instances: Dict[str, "LLM"] = {}

        @classmethod
        def get_instance(cls, config_name: str = "default") -> "LLM":
            if config_name not in cls._instances:
                cls._instances[config_name] = cls(config_name)
            return cls._instances[config_name]

This ensures one LLM client per configuration section, allowing
different agents to use different models (e.g., ``[llm]`` for the
main model, ``[llm.vision]`` for vision tasks).

Configuration
-------------

LLM settings come from the TOML config:

.. code-block:: toml

    [llm]
    model = "gpt-4o"
    base_url = "https://api.openai.com/v1"
    api_key = "sk-..."
    max_tokens = 4096
    temperature = 0.0
    timeout = 60
    max_retries = 3

Supported Providers
-------------------

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Provider
     - Config
     - Notes
   * - OpenAI
     - ``base_url = "https://api.openai.com/v1"``
     - Default provider, full feature support
   * - Azure OpenAI
     - ``api_type = "azure"``
     - Uses ``AzureOpenAI`` client, requires ``api_version``
   * - AWS Bedrock
     - ``api_type = "aws"``
     - Uses custom ``BedrockClient`` adapter
   * - Ollama
     - ``base_url = "http://localhost:11434/v1"``
     - Local models via OpenAI-compatible API
   * - Anthropic
     - ``base_url`` pointing to Anthropic-compatible endpoint
     - Via OpenAI-compatible proxy
   * - Google
     - ``base_url`` pointing to Google-compatible endpoint
     - Via OpenAI-compatible proxy


Core Methods
============

ask() -- Text Generation
-------------------------

.. code-block:: python

    async def ask(
        self,
        messages: List[dict],
        system_msgs: Optional[List[dict]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text response from LLM."""

- Combines system messages with conversation messages
- Supports streaming (default) and non-streaming modes
- Returns the complete response text

ask_with_images() -- Multimodal
---------------------------------

.. code-block:: python

    async def ask_with_images(
        self,
        messages: List[dict],
        images: List[str],
        system_msgs: Optional[List[dict]] = None,
    ) -> str:
        """Generate response with image context."""

- Injects base64-encoded images into the last user message
- Uses the ``[llm.vision]`` model configuration if available

ask_tool() -- Function Calling
---------------------------------

.. code-block:: python

    async def ask_tool(
        self,
        messages: List[dict],
        system_msgs: Optional[List[dict]] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
    ) -> ChatCompletionMessage:
        """Generate response with tool/function calling."""

This is the primary method used by ``ToolCallAgent.think()``:

1. Formats messages with system prompt
2. Calls OpenAI API with tool schemas
3. Returns the raw ``ChatCompletionMessage`` (may contain tool_calls)
4. The agent then parses tool_calls and dispatches to tools

Parameters:

- ``tools``: List of tool schemas in OpenAI format (from
  ``ToolCollection.to_params()``)
- ``tool_choice``: ``"auto"`` (LLM decides), ``"required"``
  (must call a tool), ``"none"`` (text only)


Message Formatting
==================

The ``format_messages()`` method converts internal ``Message`` objects
to the OpenAI API format:

.. code-block:: python

    def format_messages(
        self,
        messages: List[Message],
        system_msgs: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Convert Message objects to OpenAI API format."""

Conversion rules:

- ``Message.role`` → ``role`` field
- ``Message.content`` → ``content`` field
- ``Message.tool_calls`` → ``tool_calls`` list with function name and
  arguments
- ``Message.tool_call_id`` → ``tool_call_id`` for tool result messages
- ``Message.base64_image`` → injected as image_url content part


Token Counting
==============

The ``LLM`` class includes a ``TokenCounter`` for tracking API token
usage:

.. code-block:: python

    class TokenCounter:
        def __init__(self):
            self.total_input_tokens: int = 0
            self.total_output_tokens: int = 0

        def update(self, input_tokens: int, output_tokens: int):
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

Token counting serves two purposes:

1. **Budget Management**: Track total API costs across a session
2. **Context Window Management**: Ensure messages fit within the
   model's context window

The ``tiktoken`` library is used for local token estimation.


Retry Logic
===========

API calls use ``tenacity`` for retry with exponential backoff:

.. code-block:: python

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(max_retries),
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        )),
    )
    async def _call_api(self, ...):
        ...

This handles transient API failures gracefully, with configurable
``max_retries`` (default: 3) and randomized exponential backoff.


AWS Bedrock Adapter
===================

**File**: ``app/bedrock.py``

The ``BedrockClient`` adapts the AWS Bedrock Converse API to the
OpenAI message format:

::

    OpenAI Message Format ──► BedrockClient ──► Bedrock Converse API
                                                       │
    OpenAI Response Format ◄── BedrockClient ◄──────────┘

Key conversions:

- OpenAI ``role: "assistant"`` → Bedrock ``role: "assistant"``
- OpenAI ``tool_calls`` → Bedrock ``toolUse`` blocks
- OpenAI ``role: "tool"`` → Bedrock ``toolResult`` blocks
- Bedrock ``converse()`` response → OpenAI ``ChatCompletionMessage``

This enables all OpenManus agents to work with AWS Bedrock models
(Claude, Llama, etc.) transparently.


Prompt Management
=================

Prompt Templates
----------------

All prompts are static strings in ``app/prompt/``:

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - File
     - Constants
     - Used By
   * - ``manus.py``
     - ``SYSTEM_PROMPT``, ``NEXT_STEP_PROMPT``
     - Manus, SandboxManus
   * - ``toolcall.py``
     - ``SYSTEM_PROMPT``, ``NEXT_STEP_PROMPT``
     - ToolCallAgent base
   * - ``browser.py``
     - ``SYSTEM_PROMPT``, ``NEXT_STEP_PROMPT``
     - BrowserAgent
   * - ``swe.py``
     - ``SYSTEM_PROMPT``
     - SWEAgent
   * - ``planning.py``
     - ``PLANNING_SYSTEM_PROMPT``, ``NEXT_STEP_PROMPT``
     - PlanningFlow
   * - ``mcp.py``
     - ``SYSTEM_PROMPT``, ``NEXT_STEP_PROMPT``,
       ``TOOL_ERROR_PROMPT``, ``MULTIMEDIA_RESPONSE_PROMPT``
     - MCPAgent
   * - ``visualization.py``
     - ``SYSTEM_PROMPT``, ``NEXT_STEP_PROMPT``
     - DataAnalysis

Prompt Injection Points
-----------------------

Prompts enter the LLM context at two points:

1. **System Prompt**: Set once at agent creation, prepended to every
   LLM call via ``format_messages()``

2. **Next Step Prompt**: Appended as a user message before each
   ``think()`` call, providing per-step context like current working
   directory

Dynamic Prompt Modification
----------------------------

The ``is_stuck()`` method in ``BaseAgent`` can modify the
``next_step_prompt`` when stuck detection triggers:

.. code-block:: python

    if self.is_stuck():
        self.next_step_prompt = (
            "Your previous approaches didn't work. "
            "Try a completely different strategy."
        )

This is the only runtime prompt modification mechanism.
