.. _tutorial-multi-agent-workflow:

=======================
Multi-Agent Workflow
=======================

This tutorial walks through a multi-agent workflow where a parent agent
spawns sub-agents, monitors their progress, steers one mid-task, and
aggregates results.

Scenario
========

A user asks: "Research the top 3 JavaScript frameworks and compare their
performance benchmarks, then write a summary report."

The parent agent decides to parallelize by spawning sub-agents for each
framework while it coordinates the overall task.

Step 1: Parent Receives the Request
=====================================

The parent agent (the default agent) receives the user's message through
the normal request lifecycle. After analyzing the task, it decides to
delegate research to sub-agents.

The parent agent has access to the ``subagents`` tool, which provides
sub-agent management capabilities.

Step 2: Spawning Sub-Agents
============================

The parent calls ``spawnSubagentDirect()`` to create child agents. Each
sub-agent gets its own session with a predictable key format:

.. code-block:: text

   Parent session:  whatsapp:+1234567890
   Child 1 session: agent:default:subagent:a1b2c3d4-...
   Child 2 session: agent:default:subagent:e5f6g7h8-...
   Child 3 session: agent:default:subagent:i9j0k1l2-...

Two spawn modes are available:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Mode
     - Behavior
   * - ``run``
     - One-shot execution. The sub-agent processes the task and returns a
       result. The session is not reusable.
   * - ``session``
     - Persistent session. The sub-agent can receive follow-up messages
       and maintains conversation context.

For our research task, the parent uses ``run`` mode since each sub-agent
has a self-contained task:

.. code-block:: text

   Spawn 1: "Research React's performance benchmarks, bundle size,
             rendering speed, and real-world case studies"

   Spawn 2: "Research Vue.js's performance benchmarks, bundle size,
             rendering speed, and real-world case studies"

   Spawn 3: "Research Svelte's performance benchmarks, bundle size,
             rendering speed, and real-world case studies"

Source: ``src/agents/subagent-spawn.ts``

Step 3: Depth and Concurrency Limits
======================================

Before spawning, the system enforces safety limits:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Limit
     - Default
     - Description
   * - ``maxSpawnDepth``
     - 1
     - Maximum nesting depth. Sub-agents cannot spawn their own
       sub-agents by default.
   * - ``maxChildrenPerAgent``
     - 5
     - Maximum concurrent children per parent. Our 3 sub-agents
       are within this limit.

The spawn depth is tracked via ``getSpawnDepth()``, which counts the
``subagent:`` segments in the session key:

.. code-block:: text

   agent:default:subagent:uuid1                     → depth 1 ✓
   agent:default:subagent:uuid1:subagent:uuid2      → depth 2 ✗ (blocked)

Access control is enforced via the agent's ``subagents.allowAgents``
configuration. A value of ``["*"]`` allows spawning any agent.

Source: ``src/config/agent-limits.ts``, ``src/agents/subagent-depth.ts``

Step 4: Sub-Agent Registration
================================

Each spawned sub-agent is registered in the Subagent Registry — an
in-memory ``Map`` that tracks all running children:

.. code-block:: typescript

   {
     runId: "run-abc123",
     childSessionKey: "agent:default:subagent:a1b2c3d4-...",
     requesterSessionKey: "whatsapp:+1234567890",
     task: "Research React's performance benchmarks...",
     status: "running",
     model: "claude-opus-4-6",
     startedAt: 1708700000000,
   }

The registry is queryable by the parent via the ``subagents`` tool:

.. code-block:: json

   {
     "tool": "subagents",
     "params": { "action": "list" }
   }

This returns a list of all children with their current status.

Source: ``src/agents/subagent-registry.ts``

Step 5: Concurrent Execution
==============================

All three sub-agents run concurrently in the ``subagent`` command lane:

.. code-block:: text

   Command Queue
   ┌─────────────────────────────────────────────┐
   │ Lane "main"      [parent agent run]         │
   │ Lane "subagent"  [react] [vue] [svelte]     │
   └─────────────────────────────────────────────┘

Each sub-agent independently:

1. Assembles its own system prompt
2. Resolves its model (inherits from parent unless overridden)
3. Selects an auth profile
4. Runs the ReAct tool loop
5. Uses tools (``web_search``, ``web_fetch``, etc.) to research

The parent agent continues running in the ``main`` lane. It can check
on children, process other messages, or wait.

Source: ``src/process/command-queue.ts``

Step 6: Steering a Sub-Agent
==============================

While the sub-agents are running, the user sends a follow-up message:
"Also include Solid.js instead of Svelte."

The parent agent decides to steer the Svelte sub-agent:

.. code-block:: json

   {
     "tool": "subagents",
     "params": {
       "action": "steer",
       "runId": "run-svelte-456",
       "message": "Change focus: research Solid.js instead of Svelte"
     }
   }

Steering injects a new user message into the sub-agent's active run via
``queueEmbeddedPiMessage()``. The sub-agent sees this as a mid-conversation
redirection and adjusts its work accordingly.

Alternatively, the parent could kill the Svelte sub-agent and spawn a new
one:

.. code-block:: json

   {
     "tool": "subagents",
     "params": {
       "action": "kill",
       "runId": "run-svelte-456"
     }
   }

Source: ``src/agents/tools/subagents-tool.ts``

Step 7: Sub-Agent Completion
==============================

As each sub-agent completes, the result flows back to the parent:

1. The sub-agent's final response is captured as the ``outcome``
2. The registry entry is updated to ``status: "completed"``
3. A completion event is emitted on the Agent Events bus
4. The ``subagent-announce`` module notifies the parent

.. code-block:: text

   Timeline:
   ──────────────────────────────────────────────►
   t=0    Parent spawns 3 sub-agents
   t=5    User steers: "Solid.js instead of Svelte"
   t=8    React sub-agent completes → result announced
   t=10   Vue sub-agent completes → result announced
   t=12   Solid.js sub-agent completes → result announced
   t=13   Parent aggregates and responds

Source: ``src/agents/subagent-announce.ts``

Step 8: Result Aggregation
============================

The parent agent now has three research results. Since it's an LLM-driven
orchestrator, it doesn't need explicit aggregation logic — it naturally
synthesizes the results:

.. code-block:: text

   Parent's context now contains:
   ├── Original user request
   ├── React research result (from sub-agent 1)
   ├── Vue.js research result (from sub-agent 2)
   ├── Solid.js research result (from sub-agent 3)
   └── User's steering message ("Solid.js instead of Svelte")

The parent produces a comparative summary report and delivers it to the
user through the channel.

Step 9: Cleanup
================

After the response is delivered:

1. Completed sub-agent entries remain in the registry for reference
2. Sub-agent session transcripts are saved to
   ``~/.openclaw/agents/<agentId>/sessions/``
3. Token usage from all sub-agents is tracked in the parent's session

Event Flow Diagram
===================

The complete event flow across all agents:

.. code-block:: text

   User            Parent Agent         Sub-Agents        Events Bus
    │                  │                    │                  │
    │─── request ─────►│                    │                  │
    │                  │                    │                  │
    │                  │── spawn react ────►│                  │
    │                  │── spawn vue ──────►│                  │
    │                  │── spawn svelte ───►│    lifecycle:    │
    │                  │                    │──── start ──────►│
    │                  │                    │                  │
    │                  │                    │    tool:         │
    │                  │                    │──── web_search ─►│
    │                  │                    │                  │
    │─── "use Solid" ─►│                    │                  │
    │                  │── steer svelte ───►│                  │
    │                  │                    │                  │
    │                  │◄── react result ──│    lifecycle:    │
    │                  │◄── vue result ────│──── complete ───►│
    │                  │◄── solid result ──│                  │
    │                  │                    │                  │
    │◄── summary ──────│                    │                  │

Key Concepts
=============

1. **No explicit planner**: The LLM is the planner. It decides when to
   spawn sub-agents, how to distribute work, and how to aggregate results.

2. **Session isolation**: Each sub-agent has its own session with separate
   context, tools, and token tracking.

3. **Depth limits prevent runaway**: ``DEFAULT_SUBAGENT_MAX_SPAWN_DEPTH = 1``
   prevents sub-agents from spawning their own children by default.

4. **Concurrent execution**: Sub-agents run in parallel via the command
   lane queue, maximizing throughput.

5. **Mid-flight steering**: Running sub-agents can be redirected without
   killing and restarting them.

6. **Graceful failure**: If a sub-agent fails, the parent receives an error
   outcome and can retry or work with partial results.

Key Source Files
=================

- ``src/agents/subagent-spawn.ts`` — Spawning mechanics
- ``src/agents/subagent-registry.ts`` — Running child tracking
- ``src/agents/subagent-announce.ts`` — Result announcement
- ``src/agents/subagent-depth.ts`` — Depth limit enforcement
- ``src/agents/tools/subagents-tool.ts`` — Agent-facing management tool
- ``src/config/agent-limits.ts`` — Default limits
- ``src/process/command-queue.ts`` — Lane-based concurrency
- ``src/infra/agent-events.ts`` — Event distribution
