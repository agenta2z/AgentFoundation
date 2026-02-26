.. _glossary:

========
Glossary
========

.. glossary::
   :sorted:

   AgentBinding
      A routing rule that maps incoming messages to specific agents based on
      channel, account ID, peer ID, guild/server, team, or Discord role.
      Configured in the ``agents.bindings`` array of ``openclaw.json``.
      Defined as the ``AgentBinding`` type in
      ``src/config/types.agents.ts``.

   Agent
      A configured AI entity within OpenClaw that processes messages and
      executes tasks. Agents are declaratively defined in the OpenClaw config
      (``openclaw.json``) with properties like model, skills, workspace, and
      tool access. Each agent has an ID (normalized to lowercase), its own
      workspace directory, and isolated session state. See
      :doc:`orchestration/agent-definition`.

   Auth Profile
      An API key or OAuth credential entry used to authenticate with an LLM
      provider. Multiple auth profiles can be configured per provider, enabling
      automatic rotation when one key hits rate limits. Profiles track cooldown
      state, usage counts, and failure reasons. Managed by
      ``src/agents/auth-profiles.ts``.

   Batch Embedding
      Asynchronous embedding operations for indexing large numbers of memory
      chunks. Supported by OpenAI, Gemini, and Voyage APIs. Includes failure
      tracking with auto-disable after 2 consecutive failures. Implemented
      across ``src/memory/batch-*.ts`` (14 files).

   Channel
      A messaging platform integration (e.g., WhatsApp, Telegram, Discord,
      Slack, iMessage, Signal). Channels are implemented either as core modules
      (``src/telegram/``, ``src/discord/``, etc.) or as extension plugins
      (``extensions/``). All channels implement the ``ChannelMessagingAdapter``
      interface. See :doc:`channels/index`.

   Chunk
      A segment of text extracted from a memory file during indexing. Created by
      ``chunkMarkdown()`` with configurable size (default: 400 tokens / ~1600
      chars) and overlap (default: 80 tokens). Each chunk is stored in the
      SQLite ``chunks`` table with its embedding vector and source location
      (path, start_line, end_line).

   Compaction
      The process of summarizing older conversation history to free context
      window tokens. Triggered automatically when the session approaches the
      context limit. Compaction mode defaults to ``"safeguard"``. A
      :term:`Memory Flush` may fire before compaction to persist important
      knowledge.

   Directive
      An inline command in a user message that modifies agent behavior for that
      session. Examples: ``/think high`` (set thinking level), ``/model gpt``
      (switch model), ``/verbose on`` (enable verbose output). Directives are
      parsed during the request lifecycle before the agent run.

   Embedding
      A dense vector representation of text, used for semantic similarity
      search. OpenClaw supports five embedding providers: Local
      (node-llama-cpp), OpenAI, Gemini, Voyage, and Mistral. All embeddings are
      L2-normalized. See :doc:`knowledge/embeddings`.

   Gateway
      The Express v5 + WebSocket server that serves as the central hub for all
      OpenClaw operations. It handles HTTP API requests, WebSocket connections
      for the Control UI, channel authentication, agent prompt orchestration,
      and inter-agent communication. Default port is configurable. See
      :doc:`gateway/index`.

   Hook
      An event-driven extension point that executes custom logic in response to
      system events. Hooks can be bundled (e.g., ``boot-md``,
      ``session-memory``, ``command-logger``) or workspace-defined. Configured
      under the ``hooks`` section of ``openclaw.json``.

   Lane
      A named execution queue in the command queue system
      (``src/process/command-queue.ts``). Lanes serialize task execution with
      configurable concurrency. Standard lanes: ``main`` (primary auto-reply),
      ``cron`` (scheduled tasks), ``subagent`` (sub-agent runs), ``nested``
      (nested agent steps), and per-session lanes (``session:<id>``).

   Memory
      The persistent knowledge store that agents read from and write to. Backed
      by Markdown files (``MEMORY.md``, ``memory/*.md``) indexed into a SQLite
      database with vector embeddings for semantic search. See
      :doc:`knowledge/index`.

   Memory Flush
      An automatic mechanism that fires before context :term:`Compaction` to
      give the agent a chance to persist durable memories to
      ``memory/YYYY-MM-DD.md``. Triggered when session tokens approach the
      context window limit. Implemented in
      ``src/auto-reply/reply/memory-flush.ts``.

   Model Catalog
      The system that resolves model definitions, capabilities, and provider
      configurations. Manages model aliases (e.g., ``opus`` →
      ``anthropic/claude-opus-4-6``), default parameters, and cost metadata.
      Implemented in ``src/agents/model-catalog.ts`` and
      ``src/agents/models-config.ts``.

   Model Fallback
      A multi-level error recovery mechanism. When the primary model fails,
      the system tries: (1) alternative auth profiles, (2) fallback models,
      (3) lower thinking levels, (4) session reset. Implemented in
      ``src/agents/model-fallback.ts``. See :doc:`orchestration/error-handling`.

   Pi Agent Core
      The underlying agent runtime library (``@mariozechner/pi-agent-core``)
      that manages the LLM tool loop. It handles sending messages to LLMs,
      parsing tool calls, executing tools, appending results, and repeating
      until the LLM produces a final text response (ReAct-style loop).

   Plugin
      An extension package that adds channels, tools, or other capabilities to
      OpenClaw. Plugins live under ``extensions/`` with an
      ``openclaw.plugin.json`` manifest. They use the Plugin SDK
      (``src/plugin-sdk/``) to register tools and channel adapters. See
      :doc:`plugins/index`.

   Provider
      An LLM service that hosts AI models. Supported providers include
      Anthropic, OpenAI, Google (Gemini), OpenRouter, Ollama, GitHub Copilot,
      Qwen, Xiaomi, and MiniMax. Providers are configured via auth profiles
      and model definitions.

   Provider Key
      An embedding model identifier used to partition the embedding cache
      (``embedding_cache`` table). Composed of provider name, model ID, and
      API key hash. Ensures cached embeddings are not reused across different
      providers or API keys.

   QMD
      An external local-first search engine used as an alternative memory
      backend. Invoked via CLI commands (``qmd update``, ``qmd embed``,
      ``qmd query``). Collections are stored in
      ``~/.openclaw/state/agents/<id>/qmd/``. See :doc:`knowledge/qmd-backend`.

   Session
      A conversation context between a user (or system) and an agent. Each
      session has a unique key (e.g., ``main``, ``agent:ops:main``,
      ``agent:ops:subagent:<uuid>``), persisted state in ``sessions.json``,
      and a JSONL transcript file. See :doc:`orchestration/state-management`.

   Session Key
      A string identifier for a :term:`Session`. Format varies by context:
      ``main`` (default CLI session), ``agent:<id>:main`` (agent-scoped),
      ``agent:<id>:subagent:<uuid>`` (sub-agent session). Used for routing
      and state lookup.

   Skill
      A markdown-defined agent capability module. Each skill is a directory
      containing a ``SKILL.md`` file with YAML frontmatter that specifies
      metadata, requirements, and the prompt content. Skills are loaded from
      6 precedence layers and composed into the agent's system prompt. See
      :doc:`skills/index`.

   Spawn Mode
      The execution mode for sub-agent creation. ``run`` mode creates a
      one-shot task that completes and announces results. ``session`` mode
      creates a persistent session that remains active for follow-up messages.
      See :doc:`orchestration/multi-agent`.

   Sub-agent
      A child agent spawned by a parent agent to handle a delegated task.
      Sub-agents run in isolated sessions with their own system prompts and
      tool sets. Spawning is depth-limited (default:
      ``DEFAULT_SUBAGENT_MAX_SPAWN_DEPTH = 1``) and concurrency-limited
      (default: ``maxChildrenPerAgent = 5``).

   Think Level
      The extended thinking capability level passed to supporting LLMs.
      Values: ``off``, ``minimal``, ``low``, ``medium``, ``high``, ``xhigh``.
      Can be set via config, session override, or the ``/think`` directive.
      Used for chain-of-thought reasoning in models that support it.

   Tool
      An executable capability exposed to an agent during its LLM tool loop.
      Tools implement the ``AgentTool`` interface with name, description,
      TypeBox parameter schema, and ``execute()`` method. Built-in tools
      include file I/O, bash execution, web fetch, browser automation, memory
      search, and inter-session messaging. See :doc:`orchestration/tool-system`.

   Workspace
      The working directory for an agent, where it can read/write files,
      execute commands, and store project-level data. Defaults to
      ``~/.openclaw/workspace`` for the default agent, or
      ``~/.openclaw/workspace-<id>`` for named agents. Configurable per-agent.
