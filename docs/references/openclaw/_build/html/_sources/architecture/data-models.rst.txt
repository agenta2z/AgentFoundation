.. _architecture-data-models:

===========
Data Models
===========

This section documents the core TypeScript types and schemas that define
OpenClaw's data structures.

Agent Configuration Types
=========================

AgentEntry
----------

Defines a single agent in the configuration (``agents.list[]``):

.. code-block:: typescript

   type AgentEntry = {
     id: string;              // Unique identifier (normalized to lowercase)
     name?: string;           // Display name
     default?: boolean;       // Whether this is the default agent
     workspace?: string;      // Working directory
     agentDir?: string;       // Persistent agent state directory
     model?: string | {       // Model selection
       primary: string;
       fallbacks?: string[];
     };
     skills?: string[];       // Skill filter list
     memorySearch?: unknown;  // Memory integration config
     humanDelay?: unknown;    // Simulated typing delay
     heartbeat?: unknown;     // Periodic heartbeat config
     identity?: unknown;      // Name/avatar overrides
     groupChat?: unknown;     // Group chat behavior
     subagents?: {
       allowAgents?: string[];  // Sub-agent spawning ACLs ("*" = all)
     };
     sandbox?: unknown;       // Sandbox mode config
     tools?: unknown;         // Tool allow/deny lists
   };

Source: ``src/agents/agent-scope.ts``

Session Types
=============

SessionEntry
------------

Persisted per-session state in ``sessions.json``:

.. code-block:: typescript

   type SessionEntry = {
     sessionId: string;
     totalTokens?: number;
     totalTokensFresh?: number;
     compactionCount?: number;
     memoryFlushCompactionCount?: number;
     model?: string;
     thinkLevel?: ThinkLevel;
     // ... additional fields for delivery context, skill snapshots,
     //     auth profile state, usage stats, group metadata
   };

Source: ``src/config/sessions.ts``

Memory Types
============

MemorySearchResult
------------------

Returned by the ``memory_search`` tool:

.. code-block:: typescript

   type MemorySearchResult = {
     path: string;        // File path relative to workspace
     startLine: number;   // Start line of the matching chunk
     endLine: number;     // End line of the matching chunk
     score: number;       // Relevance score (0-1)
     snippet: string;     // Matching text content
     source: "memory" | "sessions";  // Source type
     citation?: string;   // Formatted citation string
   };

MemorySearchManager
-------------------

The interface implemented by both builtin and QMD backends:

.. code-block:: typescript

   interface MemorySearchManager {
     search(
       query: string,
       opts?: { maxResults?: number; minScore?: number; sessionKey?: string },
     ): Promise<MemorySearchResult[]>;

     readFile(params: {
       relPath: string;
       from?: number;
       lines?: number;
     }): Promise<{ text: string; path: string }>;

     status(): MemoryProviderStatus;
     sync?(params?: { reason?: string; force?: boolean; ... }): Promise<void>;
     probeEmbeddingAvailability(): Promise<MemoryEmbeddingProbeResult>;
     probeVectorAvailability(): Promise<boolean>;
     close?(): Promise<void>;
   }

MemoryProviderStatus
--------------------

Status information about the memory subsystem:

.. code-block:: typescript

   type MemoryProviderStatus = {
     backend: "builtin" | "qmd";
     provider: string;
     model?: string;
     files?: number;
     chunks?: number;
     dirty?: boolean;
     workspaceDir?: string;
     dbPath?: string;
     vector?: {
       enabled: boolean;
       available?: boolean;
       dims?: number;
     };
     batch?: {
       enabled: boolean;
       failures: number;
       limit: number;
     };
     fts?: { enabled: boolean; available: boolean };
     cache?: { enabled: boolean; entries?: number; maxEntries?: number };
   };

Source: ``src/memory/types.ts``

Skill Types
===========

.. code-block:: typescript

   type SkillEntry = {
     skill: Skill;                        // from @mariozechner/pi-coding-agent
     frontmatter: ParsedSkillFrontmatter; // YAML key-value pairs
     metadata?: OpenClawSkillMetadata;    // OpenClaw-specific metadata
     invocation?: SkillInvocationPolicy;  // Invocation control
   };

See :doc:`../skills/skill-definition` for the full type specifications.

Source: ``src/agents/skills/types.ts``

Gateway Protocol Schemas
=========================

Protocol schemas are defined in ``src/gateway/protocol/schema/`` using
TypeBox:

- **Primitives**: Base types for messages, sessions, and errors
- **Nodes**: Schema for node (companion app) connections
- **Logs/Chat**: Chat log entries
- **Error codes**: Structured error code definitions

See :doc:`../gateway/rpc-protocol` for the full protocol documentation.

Plugin SDK Types
================

The Plugin SDK (``src/plugin-sdk/index.ts``) exports typed interfaces for
plugin development:

- ``ChannelMessagingAdapter`` — Message send/receive
- ``ChannelConfigAdapter`` — Configuration UI
- ``ChannelPairingAdapter`` — Authentication/pairing

See :doc:`../plugins/extension-development` for details.
