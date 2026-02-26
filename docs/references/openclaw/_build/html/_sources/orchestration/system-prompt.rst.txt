.. _orchestration-system-prompt:

=============
System Prompt
=============

The system prompt is the foundational instruction set that defines an agent's
behavior. OpenClaw assembles it dynamically from multiple sections based on the
agent's configuration, available tools, and runtime context.

Assembly Function
=================

The system prompt is built by ``buildEmbeddedSystemPrompt()`` in
``src/agents/pi-embedded-runner/system-prompt.ts``, which delegates to
``buildAgentSystemPrompt()`` in ``src/agents/system-prompt.ts``.

.. code-block:: typescript

   function buildEmbeddedSystemPrompt(params: {
     workspaceDir: string;
     tools: AgentTool[];
     runtimeInfo: RuntimeInfo;
     skillsPrompt?: string;
     extraSystemPrompt?: string;
     workspaceNotes?: string[];
     sandboxInfo?: EmbeddedSandboxInfo;
     contextFiles?: EmbeddedContextFile[];
     memoryCitationsMode?: MemoryCitationsMode;
     // ... many more parameters
   }): string

Prompt Sections
===============

The system prompt is assembled from these sections in order:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Section
     - Content
   * - Runtime Info
     - Agent ID, hostname, OS, architecture, Node.js version, model name,
       provider, capabilities, channel name
   * - Tool Names
     - List of available tool names
   * - Tool Summaries
     - Descriptions of each tool's purpose and parameters (built via
       ``buildToolSummaryMap()``)
   * - Workspace Directory
     - The agent's working directory path
   * - Workspace Notes
     - Project-specific notes from ``AGENTS.md`` or ``CLAUDE.md`` files in the
       workspace
   * - Skills Prompt
     - Formatted content from all eligible skills (see
       :doc:`../skills/skill-composition`)
   * - Extra System Prompt
     - Custom additions from subagent context or user configuration
   * - Identity
     - Owner/sender information (optionally hashed for privacy)
   * - Thinking Hints
     - Guidance for the default thinking level
   * - Heartbeat Prompt
     - Instructions for periodic heartbeat messages
   * - Docs Path
     - Path to documentation files the agent can reference
   * - TTS Hints
     - Text-to-speech formatting guidance
   * - Reaction Guidance
     - When and how to use emoji reactions (minimal or extensive)
   * - Timezone & Time
     - User's timezone and current time
   * - Context Files
     - Embedded file contents for additional context
   * - Memory Citations
     - Citation mode guidance (auto/on/off)
   * - Model Aliases
     - Available model shorthand names and what they resolve to
   * - Sandbox Info
     - Sandbox mode details (mounted paths, restrictions)
   * - Channel Capabilities
     - Channel-specific features (threading, reactions, media types, actions)
   * - Message Tool Hints
     - Hints for channel-specific messaging tools

Parameters
==========

Key parameters that shape the prompt:

.. code-block:: typescript

   type RuntimeInfo = {
     agentId?: string;       // Current agent identifier
     host: string;           // Hostname
     os: string;             // Operating system
     arch: string;           // CPU architecture
     node: string;           // Node.js version
     model: string;          // Active model name
     provider?: string;      // Provider name
     capabilities?: string[];// Model capabilities
     channel?: string;       // Current channel name
     channelActions?: string[]; // Available channel actions
   };

Token Budgeting
===============

The system prompt competes for context window space with conversation history,
tool results, and model output. Key budgeting mechanisms:

1. **Skills prompt limit**: ``maxSkillsPromptChars = 30,000`` caps the total
   skill content
2. **Tool summary compaction**: Tool descriptions are kept concise
3. **Path compaction**: Home directory paths replaced with ``~``
4. **Context file limits**: Embedded files are size-bounded

The total system prompt size varies but is typically 2,000-10,000 tokens
depending on the number of active tools and skills.

Source: ``src/agents/pi-embedded-runner/system-prompt.ts``,
``src/agents/system-prompt.ts``, ``src/agents/tool-summaries.ts``
