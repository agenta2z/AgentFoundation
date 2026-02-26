.. _orchestration-state-management:

================
State Management
================

OpenClaw manages state across multiple layers — from persistent session stores
to in-memory queues — to support concurrent agent runs, follow-up messages,
and crash recovery.

Session Store
=============

**Location**: ``~/.openclaw/sessions.json`` (per agent)

The session store persists session entries with atomic read-modify-write via
``updateSessionStore()``. Each entry contains:

- Session ID and key
- Model and thinking level overrides
- Token usage counters (``totalTokens``, ``totalTokensFresh``)
- Compaction count and memory flush tracking
- Delivery context (channel, sender info)
- Skill snapshot reference
- Auth profile state

Session Transcripts
===================

**Location**: ``~/.openclaw/agents/<agentId>/sessions/*.jsonl``

Full conversation history stored as JSONL files. Each message has:

- A unique ID
- A ``parentId`` forming a directed acyclic graph (DAG)
- Role (system, user, assistant, tool)
- Content blocks (text, tool calls, tool results)
- Timestamps and metadata

The ``SessionManager`` from ``@mariozechner/pi-coding-agent`` handles
reading, writing, and compacting transcripts.

**Compaction**: When the transcript grows too large for the context window, older
turns are summarized into a condensed history, freeing tokens for new messages.

Subagent Registry
=================

**Type**: In-memory ``Map`` (with optional crash recovery persistence)

The subagent registry (``src/agents/subagent-registry.ts``) tracks all spawned
sub-agent runs:

.. code-block:: typescript

   type SubagentRunEntry = {
     runId: string;
     childSessionKey: string;
     requesterSessionKey: string;
     task: string;
     status: "running" | "completed" | "failed" | "killed";
     outcome?: string;
     model?: string;
     startedAt: number;
     completedAt?: number;
     deliveryContext?: unknown;
   };

The registry is queried by the ``subagents`` tool to list, kill, or steer
running children.

Command Lane Queue
==================

**Source**: ``src/process/command-queue.ts``

An in-process task queue with named lanes that serializes command execution:

.. code-block:: text

   ┌──────────────────────────────────────────────┐
   │              Command Queue                   │
   │                                              │
   │  Lane "main"     [task] → [task] → ...      │
   │  Lane "cron"     [task] → ...               │
   │  Lane "subagent" [task] → [task] → ...      │
   │  Lane "nested"   [task] → ...               │
   │  Lane "session:abc" [task] → ...            │
   └──────────────────────────────────────────────┘

Each lane has configurable concurrency (default: 1) and drains tasks in FIFO
order. Tasks that wait too long trigger ``onWait`` callbacks with queue-ahead
warnings.

Lane types:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Lane
     - Purpose
   * - ``main``
     - Primary auto-reply workflow
   * - ``cron``
     - Scheduled/cron tasks
   * - ``subagent``
     - Sub-agent runs
   * - ``nested``
     - Nested agent steps (e.g., ``agent_step`` tool)
   * - ``session:<id>``
     - Per-session serialization

``CommandLaneClearedError`` is thrown when a lane is cleared while tasks are
queued, allowing callers to handle cancellation gracefully.

Follow-up Queue
===============

**Source**: ``src/auto-reply/reply/queue.ts``

When a message arrives while an agent run is already active for a session, it
enters the follow-up queue with one of two modes:

- **Steer mode**: Inject the message into the active run via
  ``queueEmbeddedPiMessage()``, redirecting the agent's current work
- **Queue mode**: Wait for the current run to finish, then process the
  message as a new run

The queue supports:

- Configurable dedup (drop duplicate messages)
- Drop policies (discard messages that arrive too fast)
- Sequential drain after the active run completes

Agent Events Bus
================

**Source**: ``src/infra/agent-events.ts``

An in-process pub/sub event system with typed streams:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Stream
     - Description
   * - ``lifecycle``
     - Run start, end, abort events
   * - ``tool``
     - Tool call start, result, error events
   * - ``assistant``
     - Partial and complete assistant messages
   * - ``error``
     - Error conditions during runs
   * - ``compaction``
     - Compaction start/complete events

Each event carries:

- ``runId`` — unique run identifier
- Monotonic sequence number (per-run)
- Timestamp
- Event-specific payload

Events are used for:

- Streaming tool events to the Control UI via WebSocket
- Run job tracking (``agent-job.ts`` caches run snapshots for wait/poll)
- Subagent completion signaling

Auth Profile Store
==================

**Source**: ``src/agents/auth-profiles.ts``

Per-agent persistent store of API key profiles:

- Cooldown timestamps (when a profile was rate-limited)
- Usage counts (how many times each profile has been used)
- Failure reasons (why a profile was put in cooldown)
- Last-used timestamps (for round-robin ordering)

Profiles are rotated automatically when one hits a rate limit, and cooldown
state persists across agent runs to avoid re-hitting the same limit.
