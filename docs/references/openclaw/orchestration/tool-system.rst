.. _tool-system:

==================
Tool Architecture
==================

This document describes how tools are defined, registered, filtered,
and adapted for different LLM providers in OpenClaw.  The primary
source files are:

- ``src/agents/pi-tools.ts`` -- tool creation and assembly
- ``src/agents/tools/common.ts`` -- shared tool types and helpers
- ``src/agents/tool-catalog.ts`` -- built-in tool catalog
- ``src/agents/tool-policy-pipeline.ts`` -- layered policy filtering
- ``src/agents/pi-tools.schema.ts`` -- schema adaptation per provider

.. contents:: On this page
   :local:
   :depth: 2


The AgentTool Interface
========================

Every tool in OpenClaw implements the ``AgentTool`` interface from
the ``@mariozechner/pi-agent-core`` library:

.. code-block:: typescript

   interface AgentTool<TParams = unknown, TResult = unknown> {
     name: string;
     description: string;
     parameters: JSONSchema;
     execute: (params: TParams) => Promise<AgentToolResult<TResult>>;
   }

OpenClaw extends this with the ``AnyAgentTool`` type
(``src/agents/tools/common.ts``:8-10):

.. code-block:: typescript
   :caption: src/agents/tools/common.ts:8-10

   export type AnyAgentTool = AgentTool<any, unknown> & {
     ownerOnly?: boolean;
   };

The ``ownerOnly`` flag restricts the tool to authorized senders only.


Common Tool Helpers
====================

``src/agents/tools/common.ts`` provides shared utilities:

**Parameter Reading**

.. code-block:: typescript

   readStringParam(params, "key")                // optional string
   readStringParam(params, "key", { required: true })  // required
   readParamRaw(params, key)                     // raw value

Parameters are read with automatic ``camelCase`` to ``snake_case``
fallback (``toSnakeCaseKey()``, line 56-61), so both
``sessionKey`` and ``session_key`` are accepted.

**Error Types**

.. code-block:: typescript

   class ToolInputError extends Error { ... }     // 400
   class ToolAuthorizationError extends ToolInputError { ... }  // 403

**Action Gating**

.. code-block:: typescript

   const gate = createActionGate(config.actions);
   if (!gate("send")) { /* action disabled */ }


Built-in Tool Catalog
======================

The tool catalog (``src/agents/tool-catalog.ts``) defines all
built-in tools with their metadata:

.. list-table:: Core Tool Catalog
   :header-rows: 1
   :widths: 18 12 15 55

   * - Tool ID
     - Section
     - Profiles
     - Description
   * - ``read``
     - Files
     - coding
     - Read file contents
   * - ``write``
     - Files
     - coding
     - Create or overwrite files
   * - ``edit``
     - Files
     - coding
     - Make precise edits
   * - ``apply_patch``
     - Files
     - coding
     - Patch files (OpenAI format)
   * - ``exec``
     - Runtime
     - coding
     - Run shell commands
   * - ``process``
     - Runtime
     - coding
     - Manage background processes
   * - ``web_search``
     - Web
     - (all)
     - Search the web
   * - ``web_fetch``
     - Web
     - (all)
     - Fetch web content
   * - ``memory_search``
     - Memory
     - coding
     - Semantic memory search
   * - ``memory_get``
     - Memory
     - coding
     - Read memory files
   * - ``sessions_list``
     - Sessions
     - coding, messaging
     - List sessions
   * - ``sessions_history``
     - Sessions
     - coding, messaging
     - Fetch session history
   * - ``sessions_send``
     - Sessions
     - coding, messaging
     - Send to another session
   * - ``sessions_spawn``
     - Sessions
     - coding
     - Spawn a sub-agent
   * - ``subagents``
     - Sessions
     - coding
     - Manage sub-agent runs
   * - ``session_status``
     - Sessions
     - minimal, coding, messaging
     - Show session status card
   * - ``browser``
     - UI
     - (all)
     - Control web browser
   * - ``canvas``
     - UI
     - (all)
     - Control canvases
   * - ``message``
     - Messaging
     - messaging
     - Send messages and channel actions
   * - ``cron``
     - Automation
     - (all)
     - Schedule tasks and reminders
   * - ``gateway``
     - Automation
     - (all)
     - Gateway control (restart, update)
   * - ``nodes``
     - Nodes
     - (all)
     - List/control paired devices
   * - ``agents_list``
     - Agents
     - (all)
     - List configured agent IDs
   * - ``image``
     - Media
     - coding
     - Image understanding
   * - ``tts``
     - Media
     - (all)
     - Text-to-speech conversion


Tool Profiles
===============

Tools are organized into four profiles that control which tools
are available by default:

.. list-table:: Tool Profiles
   :header-rows: 1
   :widths: 15 85

   * - Profile
     - Included Tools
   * - ``minimal``
     - ``session_status`` only
   * - ``coding``
     - File tools + runtime + memory + sessions + image
   * - ``messaging``
     - Session tools + message
   * - ``full``
     - All tools (no restrictions)

Profiles are resolved by ``resolveCoreToolProfilePolicy()``
(line 287-302), which returns an allow/deny policy.


Tool Groups
=============

Tools can be referenced as groups in policy configurations:

- ``group:openclaw`` -- all tools marked ``includeInOpenClawGroup``
- ``group:fs`` -- file tools (read, write, edit, apply_patch)
- ``group:runtime`` -- exec, process
- ``group:web`` -- web_search, web_fetch
- ``group:memory`` -- memory_search, memory_get
- ``group:sessions`` -- sessions_list, sessions_history,
  sessions_send, sessions_spawn, subagents, session_status
- ``group:ui`` -- browser, canvas
- ``group:messaging`` -- message
- ``group:automation`` -- cron, gateway
- ``group:nodes`` -- nodes
- ``group:agents`` -- agents_list
- ``group:media`` -- image, tts


Plugin Tool Registration
=========================

Plugins can register additional tools via the plugin SDK.  Plugin
tools are integrated into the tool pipeline through
``getPluginToolMeta()`` (``src/plugins/tools.ts``), which returns
metadata including the ``pluginId`` for each plugin-provided tool.

Plugin tools go through the same policy pipeline as built-in tools
and can be individually allowed or denied in configuration.


Tool Policy Pipeline
=====================

The tool policy pipeline
(``src/agents/tool-policy-pipeline.ts``) applies a series of
allow/deny filters to determine which tools are available for a
given run:

.. code-block:: typescript
   :caption: src/agents/tool-policy-pipeline.ts:17-63

   export function buildDefaultToolPolicyPipelineSteps(params: {
     profilePolicy?: ToolPolicyLike;
     globalPolicy?: ToolPolicyLike;
     agentPolicy?: ToolPolicyLike;
     groupPolicy?: ToolPolicyLike;
     // ...
   }): ToolPolicyPipelineStep[]

The pipeline applies steps in order:

1. **Profile policy** -- ``tools.profile`` (e.g., ``"coding"``)
2. **Provider profile policy** -- ``tools.byProvider.profile``
3. **Global policy** -- ``tools.allow`` / ``tools.deny``
4. **Global provider policy** -- ``tools.byProvider.allow``
5. **Agent policy** -- ``agents.<id>.tools.allow``
6. **Agent provider policy** -- ``agents.<id>.tools.byProvider.allow``
7. **Group policy** -- group-specific tool restrictions

Each step can:

- **Allow** specific tools (whitelist).
- **Deny** specific tools (blacklist).
- Use ``tools.alsoAllow`` for additive plugin tool enablement.

The ``applyToolPolicyPipeline()`` function (line 65-) iterates
through the steps, progressively filtering the tool list.

.. warning::

   If an allowlist contains only plugin tool names (not core tools),
   the pipeline strips it to avoid accidentally disabling all core
   tools.  A warning is logged to alert the operator.


Schema Adaptation per Provider
===============================

Different LLM providers have different requirements for tool
schemas.  OpenClaw adapts schemas automatically:

**Anthropic / Claude**

- Supports the full JSON Schema specification.
- The ``patchToolSchemaForClaudeCompatibility()`` function
  (``src/agents/pi-tools.read.ts``) adds ``claude_``-prefixed
  parameter aliases for tools that need backward-compatible
  parameter names.

**Google / Gemini**

- Does not support ``anyOf``, ``oneOf``, ``allOf`` in tool schemas.
- ``cleanToolSchemaForGemini()``
  (``src/agents/pi-tools.schema.ts``) removes unsupported constructs
  and flattens enum types.

**OpenAI**

- Requires ``type: "object"`` at the top level.
- ``normalizeToolParameters()``
  (``src/agents/pi-tools.schema.ts``) ensures compliance.

.. note::

   Tool schemas should avoid ``Type.Union`` in input schemas.
   Use ``stringEnum`` / ``optionalStringEnum`` for string lists and
   ``Type.Optional(...)`` instead of ``| null``.  The ``format``
   property name should also be avoided as some validators treat it
   as reserved.


Tool Wrappers
==============

Before tools are registered with the agent session, they pass
through several wrapping layers:

1. **Workspace root guard** -- ``wrapToolWorkspaceRootGuard()``
   restricts file operations to the configured workspace.
2. **Abort signal** -- ``wrapToolWithAbortSignal()`` propagates
   the run's abort signal to tool execution.
3. **Before-tool-call hook** -- ``wrapToolWithBeforeToolCallHook()``
   allows plugins to intercept tool calls.
4. **Parameter normalization** -- ``wrapToolParamNormalization()``
   handles camelCase/snake_case conversion.
5. **Owner-only gating** -- ``applyOwnerOnlyToolPolicy()``
   restricts tools marked ``ownerOnly`` to authorized senders.
6. **Image sanitization** -- tool results containing images are
   sanitized via ``sanitizeToolResultImages()``.


Tool Assembly
==============

The ``createOpenClawCodingTools()`` function
(``src/agents/pi-tools.ts``:171-) assembles the complete tool set:

1. Start with pi-coding-agent's ``codingTools`` (read, write, edit,
   grep, find, ls).
2. Add ``exec`` and ``process`` tools with configured defaults.
3. Add ``apply_patch`` if the model supports it.
4. Add OpenClaw-specific tools (sessions, subagents, message, cron,
   etc.) via ``createOpenClawTools()``.
5. Add channel-specific tools via ``listChannelAgentTools()``.
6. Apply the tool policy pipeline to filter the final set.
7. Wrap each tool with guards, hooks, and signal handling.
