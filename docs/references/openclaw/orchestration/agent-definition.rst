.. _agent-definition:

====================
Agent Configuration
====================

This document describes how agents are defined, identified, and
configured in OpenClaw.  The primary source files are:

- ``src/agents/agent-scope.ts`` -- agent entry resolution, workspace
  and directory helpers
- ``src/agents/agent-paths.ts`` -- default agent directory resolution
- ``src/config/agent-limits.ts`` -- concurrency and depth constants

.. contents:: On this page
   :local:
   :depth: 2


The AgentEntry Type
====================

Every agent is described by an ``AgentEntry`` object, which is the
element type of the ``agents.list`` array in the OpenClaw configuration.

.. code-block:: typescript
   :caption: src/agents/agent-scope.ts:18

   type AgentEntry = NonNullable<
     NonNullable<OpenClawConfig["agents"]>["list"]
   >[number];

Each entry may contain the following fields:

.. list-table:: AgentEntry Fields
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``id``
     - ``string``
     - Agent identifier (normalized to lowercase). **Required**.
   * - ``name``
     - ``string``
     - Human-readable display name.
   * - ``default``
     - ``boolean``
     - If ``true``, this agent is selected when no explicit agent is
       specified. If multiple entries have ``default: true``, the first
       one wins and a warning is logged (line 70-73).
   * - ``workspace``
     - ``string``
     - Custom workspace directory path. Supports ``~`` expansion.
   * - ``agentDir``
     - ``string``
     - Custom agent state directory. Supports ``~`` expansion.
   * - ``model``
     - ``string | object``
     - Model configuration. A bare string sets the primary model.
       An object supports ``{ primary, fallbacks }`` for per-agent
       fallback chains.
   * - ``skills``
     - ``string[]``
     - Skill filter list. Only the named skills are activated for
       this agent.
   * - ``memorySearch``
     - ``object``
     - Memory search configuration override.
   * - ``humanDelay``
     - ``object``
     - Simulated human typing delay settings.
   * - ``heartbeat``
     - ``object``
     - Heartbeat / proactive messaging configuration.
   * - ``identity``
     - ``object``
     - Agent persona and identity overrides.
   * - ``groupChat``
     - ``object``
     - Group chat behavior configuration.
   * - ``subagents``
     - ``object``
     - Sub-agent spawning constraints (``allowAgents``, ``thinking``).
   * - ``sandbox``
     - ``object``
     - Sandbox / Docker configuration for isolated execution.
   * - ``tools``
     - ``object``
     - Per-agent tool policy and exec configuration.

The ``ResolvedAgentConfig`` type (line 20-34) mirrors these fields
after resolution, with optional values for every field.


Agent ID Normalization
=======================

All agent IDs are **lowercased** before comparison or storage.
Normalization is performed by ``normalizeAgentId()`` from
``src/routing/session-key.ts``, re-exported through ``agent-scope.ts``:

.. code-block:: typescript
   :caption: src/agents/agent-scope.ts:8

   import {
     DEFAULT_AGENT_ID,
     normalizeAgentId,
     parseAgentSessionKey,
   } from "../routing/session-key.js";

The constant ``DEFAULT_AGENT_ID`` (typically ``"default"``) is used
when no agents are configured or when no explicit ID can be resolved.


Default Agent Selection
========================

The function ``resolveDefaultAgentId()`` (line 64-76) determines which
agent is the default:

1. If ``agents.list`` is empty or absent, return ``DEFAULT_AGENT_ID``.
2. Filter entries where ``default === true``.
3. If multiple defaults exist, log a warning and pick the first.
4. If no explicit default, pick the first entry in the list.
5. Normalize the chosen ID to lowercase.

.. code-block:: typescript
   :caption: src/agents/agent-scope.ts:64-76

   export function resolveDefaultAgentId(cfg: OpenClawConfig): string {
     const agents = listAgentEntries(cfg);
     if (agents.length === 0) {
       return DEFAULT_AGENT_ID;
     }
     const defaults = agents.filter((agent) => agent?.default);
     if (defaults.length > 1 && !defaultAgentWarned) {
       defaultAgentWarned = true;
       log.warn("Multiple agents marked default=true; ...");
     }
     const chosen = (defaults[0] ?? agents[0])?.id?.trim();
     return normalizeAgentId(chosen || DEFAULT_AGENT_ID);
   }


Session-to-Agent Resolution
=============================

When a message arrives, the system must determine which agent handles
the session.  ``resolveSessionAgentIds()`` (line 78-96) resolves both
the default agent and the session-specific agent:

1. Compute the default agent ID from config.
2. If an explicit ``agentId`` parameter is provided, use it.
3. Otherwise, parse the session key (e.g.,
   ``agent:researcher:subagent:abc123``) to extract the embedded
   agent ID.
4. Fall back to the default agent ID.


Per-Agent Workspace Resolution
================================

Each agent gets its own workspace directory, resolved by
``resolveAgentWorkspaceDir()`` (line 213-229):

.. code-block:: text

   Priority chain:
   1. Explicit agent.workspace in config
   2. agents.defaults.workspace (for default agent only)
   3. Default workspace from environment (OPENCLAW_WORKSPACE_DIR)
   4. <stateDir>/workspace-<agentId>  (for non-default agents)

The state directory is resolved from ``resolveStateDir()`` in
``src/config/paths.ts``, typically ``~/.openclaw``.


Per-Agent State Directory
==========================

Agent-specific state (sessions, transcripts, auth profiles) is stored
under a dedicated directory, resolved by ``resolveAgentDir()``
(line 231-239):

.. code-block:: typescript
   :caption: src/agents/agent-scope.ts:231-239

   export function resolveAgentDir(cfg: OpenClawConfig, agentId: string) {
     const id = normalizeAgentId(agentId);
     const configured = resolveAgentConfig(cfg, id)?.agentDir?.trim();
     if (configured) {
       return resolveUserPath(configured);
     }
     const root = resolveStateDir(process.env);
     return path.join(root, "agents", id, "agent");
   }

Default layout: ``~/.openclaw/agents/<agentId>/agent/``

The legacy default agent directory can also be set via the
``OPENCLAW_AGENT_DIR`` or ``PI_CODING_AGENT_DIR`` environment
variables (``src/agents/agent-paths.ts``:6-14).


Concurrency Limits
===================

Three constants in ``src/config/agent-limits.ts`` govern parallelism:

.. list-table:: Agent Concurrency Defaults
   :header-rows: 1
   :widths: 45 10 45

   * - Constant
     - Value
     - Description
   * - ``DEFAULT_AGENT_MAX_CONCURRENT``
     - ``4``
     - Maximum concurrent agent runs across all sessions.
       Configurable via ``agents.defaults.maxConcurrent``.
   * - ``DEFAULT_SUBAGENT_MAX_CONCURRENT``
     - ``8``
     - Maximum concurrent sub-agent runs.
       Configurable via ``agents.defaults.subagents.maxConcurrent``.
   * - ``DEFAULT_SUBAGENT_MAX_SPAWN_DEPTH``
     - ``1``
     - Maximum nesting depth for sub-agent spawning.
       Depth-1 sub-agents are leaves unless config opts into nesting.
       Configurable via ``agents.defaults.subagents.maxSpawnDepth``.

The resolver functions clamp values to a minimum of 1:

.. code-block:: typescript
   :caption: src/config/agent-limits.ts:8-14

   export function resolveAgentMaxConcurrent(cfg?: OpenClawConfig): number {
     const raw = cfg?.agents?.defaults?.maxConcurrent;
     if (typeof raw === "number" && Number.isFinite(raw)) {
       return Math.max(1, Math.floor(raw));
     }
     return DEFAULT_AGENT_MAX_CONCURRENT;
   }


Model Resolution per Agent
============================

Each agent can specify its own primary model and fallback chain:

.. code-block:: yaml
   :caption: Example config

   agents:
     list:
       - id: researcher
         model:
           primary: "anthropic/claude-opus-4-6"
           fallbacks:
             - "anthropic/claude-sonnet-4-5"
       - id: coder
         model: "google/gemini-2.5-pro"

Resolution functions (line 146-211):

- ``resolveAgentExplicitModelPrimary()`` -- returns only the per-agent
  primary, or ``undefined``.
- ``resolveAgentEffectiveModelPrimary()`` -- falls back to
  ``agents.defaults.model`` if the per-agent primary is not set.
- ``resolveAgentModelFallbacksOverride()`` -- returns the per-agent
  fallback list. An explicitly empty array ``[]`` disables global
  fallbacks for that agent.

See :doc:`llm-integration` for the full model resolution chain.
