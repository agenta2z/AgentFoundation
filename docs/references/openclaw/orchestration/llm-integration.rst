.. _orchestration-llm-integration:

===============
LLM Integration
===============

OpenClaw integrates with multiple LLM providers through a unified model
resolution and API key management system.

Supported Providers
===================

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Provider
     - Provider ID
     - Notes
   * - Anthropic
     - ``anthropic``
     - Default provider; Claude models; OAuth + API key auth
   * - OpenAI
     - ``openai``
     - GPT models; Codex models (OAuth)
   * - Google
     - ``google``
     - Gemini models (preview and stable)
   * - OpenRouter
     - ``openrouter``
     - Multi-model proxy service
   * - Ollama
     - ``ollama``
     - Local model hosting
   * - GitHub Copilot
     - ``github-copilot``
     - Via GitHub Copilot auth tokens
   * - Qwen
     - ``qwen-portal``
     - Alibaba Qwen models
   * - Xiaomi
     - ``xiaomi``
     - MiLM models
   * - MiniMax
     - ``minimax``
     - MiniMax models
   * - Z.AI
     - ``zai``
     - Z.AI models
   * - Volcengine
     - ``volcengine``
     - ByteDance/Doubao models
   * - Kimi
     - ``kimi-coding``
     - Moonshot Kimi models

Provider IDs are normalized via ``normalizeProviderId()`` in
``src/agents/model-selection.ts`` (e.g., ``"qwen"`` → ``"qwen-portal"``,
``"bytedance"`` → ``"volcengine"``).

Model Resolution Chain
======================

Models are resolved through a precedence chain (first non-empty wins):

.. code-block:: text

   1. Agent-specific model   →  agents.list[].model
   2. Session override       →  sessions.patch / session directive
   3. Channel override       →  Per-channel model config
   4. Global default         →  agents.defaults.model
   5. Hardcoded default      →  "claude-opus-4-6"

Source: ``src/agents/defaults.ts:4``

Model Aliases
-------------

Shorthand aliases map to full model references:

.. list-table::
   :header-rows: 1
   :widths: 20 40

   * - Alias
     - Resolves To
   * - ``opus``
     - ``anthropic/claude-opus-4-6``
   * - ``sonnet``
     - ``anthropic/claude-sonnet-4-6``
   * - ``gpt``
     - ``openai/gpt-5.2``
   * - ``gpt-mini``
     - ``openai/gpt-5-mini``
   * - ``gemini``
     - ``google/gemini-3-pro-preview``
   * - ``gemini-flash``
     - ``google/gemini-3-flash-preview``

Source: ``src/config/defaults.ts:15-27``

Model Catalog
=============

The model catalog (``src/agents/model-catalog.ts``) maintains definitions for
known models including:

- Context window size
- Max output tokens
- Input modalities (text, image, audio)
- Cost per token (input/output/cache)
- API type (``anthropic-messages``, ``openai-chat``, etc.)
- Reasoning capability flag

Provider configuration (``src/agents/models-config.ts``) handles per-provider
setup including API endpoints, authentication, and custom model registration.

Auth Profile System
===================

**Source**: ``src/agents/auth-profiles.ts``

Multiple API keys can be configured per provider. The auth profile system:

1. **Selection**: Picks the first non-cooled-down profile from the configured
   order (``auth.order.<provider>``)
2. **Rotation**: When a profile hits a rate limit, it's marked in cooldown and
   the system advances to the next profile
3. **Probe recovery**: When all profiles are in cooldown, the system
   periodically probes (every 30s) to detect recovery
4. **Persistence**: Cooldown state persists across runs to avoid re-hitting
   the same rate limits

Extended Thinking
=================

OpenClaw supports extended thinking (chain-of-thought reasoning) via the
``thinkLevel`` parameter:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Level
     - Description
   * - ``off``
     - No extended thinking
   * - ``minimal``
     - Minimal reasoning
   * - ``low``
     - Brief reasoning
   * - ``medium``
     - Moderate reasoning
   * - ``high``
     - Detailed reasoning
   * - ``xhigh``
     - Maximum reasoning depth

Think level can be set via:

- Agent config (``agents.defaults.thinkLevel``)
- Session override
- User directive (``/think high``)

When a model fails at a high thinking level, the system falls back to lower
levels automatically (see :doc:`error-handling`).

Reasoning streaming is supported via the ``onReasoningStream`` callback, which
pipes the model's internal reasoning to the Control UI when available.

Usage Tracking
==============

Token usage is tracked per-run and accumulated across tool-loop iterations:

- **Input tokens**: Prompt tokens sent to the model
- **Output tokens**: Completion tokens received
- **Cache tokens**: Tokens served from provider-side cache (read/write)

The system distinguishes "last call" metrics from accumulated totals to avoid
inflating context-size calculations during the tool loop.

Source: ``src/agents/pi-embedded-runner/run.ts``, ``src/agents/model-selection.ts``
