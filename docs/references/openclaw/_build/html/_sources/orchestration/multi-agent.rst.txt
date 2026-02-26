.. _multi-agent:

====================================
Sub-agent Spawning & Coordination
====================================

OpenClaw supports a hierarchical multi-agent architecture where a
parent agent can spawn child sub-agents to handle specialized tasks
in parallel.  The primary source files are:

- ``src/agents/subagent-spawn.ts`` -- spawn logic and validation
- ``src/agents/subagent-registry.ts`` -- run tracking and lifecycle
- ``src/agents/subagent-announce.ts`` -- result announcement back
  to parent

.. contents:: On this page
   :local:
   :depth: 2


Architecture Overview
======================

.. code-block:: text

   +-------------------+
   |   Main Agent      |  (depth 0)
   |   session: main   |
   +--------+----------+
            |
            | sessions_spawn(task="Research X")
            |
            v
   +-------------------+         +-------------------+
   | Sub-agent A       |         | Sub-agent B       |
   | depth: 1          |         | depth: 1          |
   | session: agent:   |         | session: agent:   |
   |  default:subagent:|         |  default:subagent: |
   |  <uuid-a>         |         |  <uuid-b>         |
   +--------+----------+         +-------------------+
            |
            | (if maxSpawnDepth > 1)
            v
   +-------------------+
   | Sub-sub-agent     |
   | depth: 2          |
   +-------------------+

The parent-child relationship is tracked by the **sub-agent registry**
(an in-memory ``Map<string, SubagentRunRecord>`` persisted to disk).
Each child run records its ``requesterSessionKey`` so results can be
announced back to the correct parent.


Session Key Format
===================

Sub-agent sessions use a deterministic key format:

.. code-block:: text

   agent:<targetAgentId>:subagent:<uuid>

For example:

.. code-block:: text

   agent:default:subagent:a1b2c3d4-e5f6-7890-abcd-ef1234567890

This format is constructed in ``spawnSubagentDirect()``
(``src/agents/subagent-spawn.ts``:259):

.. code-block:: typescript

   const childSessionKey =
     `agent:${targetAgentId}:subagent:${crypto.randomUUID()}`;

The ``agent:`` prefix enables agent ID extraction via
``resolveAgentIdFromSessionKey()``, and the ``subagent:`` segment
identifies the session as a sub-agent (not a regular user session).


Two Spawn Modes
================

The ``sessions_spawn`` tool supports two modes, defined in
``src/agents/subagent-spawn.ts``:22-23:

.. code-block:: typescript

   export const SUBAGENT_SPAWN_MODES = ["run", "session"] as const;

.. list-table:: Spawn Modes
   :header-rows: 1
   :widths: 15 15 70

   * - Mode
     - Lifecycle
     - Description
   * - ``run``
     - One-shot
     - The sub-agent executes the task and terminates.  Its session
       is archived (or deleted, depending on ``cleanup`` setting)
       after the result is announced.  **Default when no thread
       binding is requested.**
   * - ``session``
     - Persistent
     - The sub-agent stays alive in a dedicated channel thread after
       completing its initial task.  Users can continue interacting
       with it via the thread.  **Requires ``thread=true``.**

Mode resolution logic (``resolveSpawnMode()``, line 80-89):

- If ``mode`` is explicitly ``"run"`` or ``"session"``, use it.
- If ``thread=true`` is requested, default to ``"session"``.
- Otherwise, default to ``"run"``.

.. warning::

   ``mode="session"`` requires ``thread=true``.  Attempting to create
   a persistent session without thread binding returns an error
   (line 176-180).


Depth and Children Limits
==========================

Two limits prevent unbounded sub-agent proliferation:

**Maximum Spawn Depth**

.. code-block:: typescript
   :caption: src/config/agent-limits.ts:6

   export const DEFAULT_SUBAGENT_MAX_SPAWN_DEPTH = 1;

The default depth of 1 means sub-agents are leaf workers -- they
cannot spawn further sub-agents.  The limit is configurable via
``agents.defaults.subagents.maxSpawnDepth``.

Depth is checked at spawn time (``src/agents/subagent-spawn.ts``:219-227):

.. code-block:: typescript

   const callerDepth = getSubagentDepthFromSessionStore(
     requesterInternalKey, { cfg }
   );
   const maxSpawnDepth = cfg.agents?.defaults?.subagents?.maxSpawnDepth
     ?? DEFAULT_SUBAGENT_MAX_SPAWN_DEPTH;
   if (callerDepth >= maxSpawnDepth) {
     return {
       status: "forbidden",
       error: `sessions_spawn is not allowed at this depth ...`,
     };
   }

**Maximum Children per Agent**

.. code-block:: typescript
   :caption: src/agents/subagent-spawn.ts:229

   const maxChildren =
     cfg.agents?.defaults?.subagents?.maxChildrenPerAgent ?? 5;

Each parent session can have at most 5 active child runs by default.
This prevents a single agent from overwhelming the system with
parallel tasks.


Spawn Flow
===========

The full spawn sequence in ``spawnSubagentDirect()``
(line 162-527):

1. **Validate mode** -- ensure ``session`` mode has thread binding.
2. **Check depth limit** -- compare caller depth against max.
3. **Check children limit** -- count active children.
4. **Validate agent allowlist** -- if the target agent differs from
   the requester, check ``subagents.allowAgents``.
5. **Generate child session key** --
   ``agent:<targetId>:subagent:<uuid>``.
6. **Set spawn depth** on child session via gateway RPC
   ``sessions.patch``.
7. **Apply model override** (if specified) via ``sessions.patch``.
8. **Apply thinking override** (if specified) via ``sessions.patch``.
9. **Bind thread** (if ``thread=true``) via
   ``ensureThreadBindingForSubagentSpawn()``.
10. **Build child system prompt** via ``buildSubagentSystemPrompt()``.
11. **Dispatch agent run** via gateway RPC ``agent`` method with
    ``lane: AGENT_LANE_SUBAGENT`` and ``deliver: false``.
12. **Register in sub-agent registry** via
    ``registerSubagentRun()``.
13. **Emit lifecycle hooks** via ``runSubagentSpawned()``.
14. **Return** ``{ status: "accepted", childSessionKey, runId }``.


Agent Allowlist
================

When a sub-agent targets a different agent ID than the requester,
the system checks the ``subagents.allowAgents`` configuration:

.. code-block:: typescript
   :caption: src/agents/subagent-spawn.ts:243-258

   const allowAgents = resolveAgentConfig(
     cfg, requesterAgentId
   )?.subagents?.allowAgents ?? [];
   const allowAny = allowAgents.some(
     (value) => value.trim() === "*"
   );

- ``"*"`` allows spawning any configured agent.
- An empty or absent list disallows cross-agent spawning.
- Listed agent IDs are normalized to lowercase for comparison.


Result Announcement
====================

When a sub-agent run completes, the **announce flow**
(``runSubagentAnnounceFlow()`` in ``src/agents/subagent-announce.ts``:978-1300)
delivers the result back to the parent:

1. **Wait for completion** -- poll via gateway ``agent.wait`` RPC.
2. **Read output** -- extract the latest assistant reply from the
   child session's transcript.
3. **Build trigger message** -- format the result as a system
   message including task name, status, findings, and stats.
4. **Resolve delivery origin** -- determine the channel/thread
   where the announcement should be sent.
5. **Deliver** -- try steering (inject into active parent run),
   queuing (enqueue for follow-up), or direct send (gateway RPC).
6. **Cleanup** -- delete the child session (if ``cleanup="delete"``),
   or keep it for archival.

The trigger message format:

.. code-block:: text

   [System Message] [sessionId: <id>] A subagent task "<label>"
   just completed successfully.

   Result:
   <sub-agent's final output>

   Stats: runtime 12s - tokens 4.2k (in 3.1k / out 1.1k)

   <reply instruction for parent agent>


Sub-agent Registry
===================

The registry (``src/agents/subagent-registry.ts``) maintains an
in-memory map of all active and recently completed sub-agent runs:

.. code-block:: typescript
   :caption: src/agents/subagent-registry.ts:42

   const subagentRuns = new Map<string, SubagentRunRecord>();

Key operations:

- ``registerSubagentRun()`` -- add a new run record.
- ``completeSubagentRun()`` -- mark as ended, trigger announce flow.
- ``countActiveRunsForSession()`` -- count active children.
- ``markSubagentRunTerminated()`` -- force-terminate (kill).
- ``listSubagentRunsForRequester()`` -- list children for a parent.

The registry is persisted to disk and restored on gateway restart
via ``persistSubagentRunsToDisk()`` and
``restoreSubagentRunsFromDisk()``.

A background **sweeper** runs every 60 seconds to archive completed
runs after a configurable TTL
(``agents.defaults.subagents.archiveAfterMinutes``, default 60).

The registry listens for ``lifecycle`` events on the agent event bus
(see :doc:`communication`) to detect when sub-agent runs end:

.. code-block:: typescript
   :caption: src/agents/subagent-registry.ts:376-416

   listenerStop = onAgentEvent((evt) => {
     if (evt.stream !== "lifecycle") return;
     const entry = subagentRuns.get(evt.runId);
     if (!entry) return;
     // Handle "start", "end", "error" phases
   });


Sub-agent System Prompt
========================

Sub-agents receive a specialized system prompt
(``buildSubagentSystemPrompt()`` in
``src/agents/subagent-announce.ts``:860-950) that constrains their
behavior:

- Identifies them as a sub-agent with a specific task.
- Instructs them to stay focused and not initiate conversations.
- Tells them their final message will auto-announce to the parent.
- At depth < maxSpawnDepth, enables further sub-agent spawning.
- At max depth, marks them as leaf workers who cannot spawn.
- Includes session context (label, requester session, channel).

.. code-block:: text
   :caption: Example sub-agent system prompt excerpt

   # Subagent Context

   You are a **subagent** spawned by the main agent for a specific task.

   ## Your Role
   - You were created to handle: Research the latest API changes
   - Complete this task. That's your entire purpose.
   - You are NOT the main agent. Don't try to be.

   ## Rules
   1. **Stay focused** - Do your assigned task, nothing else
   2. **Complete the task** - Your final message will be
      automatically reported to the main agent
   ...
