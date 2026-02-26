.. _communication:

===============================
Inter-Agent Communication
===============================

This document covers the communication pathways between agents,
sessions, and the gateway.  The primary source files are:

- ``src/agents/tools/sessions-send-tool.ts`` -- the ``sessions_send``
  tool implementation
- ``src/agents/pi-embedded-messaging.ts`` -- messaging tool detection
  and classification
- ``src/infra/agent-events.ts`` -- the agent event bus

.. contents:: On this page
   :local:
   :depth: 2


Communication Pathways
=======================

OpenClaw supports several communication pathways between agents
and sessions:

.. code-block:: text

   +-----------------+     Gateway RPC     +-----------------+
   |  Agent A        | ------------------> |  Gateway        |
   |  (session X)    | <------------------ |  (process)      |
   +-----------------+   agent / send /    +-----------------+
          |              sessions.patch           |
          |                                       |
          |    Agent Events Bus                   |
          +-----(in-process pub/sub)------+       |
          |                               |       |
          v                               v       v
   +-----------------+             +-----------------+
   |  Sub-agent B    |             |  Channel        |
   |  (session Y)    |             |  Adapter        |
   +-----------------+             +-----------------+


Gateway RPC
============

The primary mechanism for cross-session communication is the
**gateway RPC** system (``src/gateway/call.ts``).  The
``callGateway()`` function sends JSON-RPC requests to the running
gateway process.

Key RPC methods used for communication:

.. list-table:: Gateway RPC Methods for Communication
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Purpose
   * - ``agent``
     - Dispatch an agent run for a session.  Used to inject messages
       (including sub-agent announcements) into a session.
   * - ``send``
     - Send a message directly to a channel without triggering an
       agent run.  Used for direct completion delivery.
   * - ``sessions.patch``
     - Update session metadata (model, thinking level, spawn depth,
       label).
   * - ``sessions.delete``
     - Delete a session and optionally its transcript.
   * - ``agent.wait``
     - Wait for an agent run to complete (blocking RPC with timeout).
   * - ``chat.history``
     - Retrieve conversation history for a session.

Example: injecting a sub-agent announcement into the parent session:

.. code-block:: typescript
   :caption: src/agents/subagent-announce.ts:497-510

   await callGateway({
     method: "agent",
     params: {
       sessionKey: item.sessionKey,
       message: item.prompt,
       channel: requesterIsSubagent ? undefined : origin?.channel,
       accountId: requesterIsSubagent ? undefined : origin?.accountId,
       to: requesterIsSubagent ? undefined : origin?.to,
       deliver: !requesterIsSubagent,
       idempotencyKey,
     },
     timeoutMs: announceTimeoutMs,
   });


Session Messaging
==================

The ``sessions_send`` Tool
---------------------------

The ``sessions_send`` tool allows an agent to send a text message to
another session.  This is the primary mechanism for cross-session
communication initiated by the LLM.

The tool is classified as a **messaging tool** by the detection
system in ``src/agents/pi-embedded-messaging.ts``:

.. code-block:: typescript
   :caption: src/agents/pi-embedded-messaging.ts:10-11

   const CORE_MESSAGING_TOOLS = new Set([
     "sessions_send", "message"
   ]);

When the LLM calls ``sessions_send(sessionKey, message)``, the tool
dispatches the message to the target session via the gateway ``agent``
RPC, which triggers a new agent run in the target session's context.


Message Steering
-----------------

When a new message arrives for a session that already has an active
run, the system can **steer** the message into the running context
instead of queuing it:

.. code-block:: typescript

   queueEmbeddedPiMessage(sessionId, message);

This function (from ``src/agents/pi-embedded.ts``) injects the
message into the active run's pi-agent-core session, causing the
model to see it as a new user turn in the same conversation context.

Steering is used for:

- Sub-agent announcement delivery to an active parent session.
- Follow-up user messages when queue mode is ``"steer"``.
- Real-time context injection during long-running tasks.


Messaging Tool Classification
-------------------------------

``isMessagingTool()`` (``src/agents/pi-embedded-messaging.ts``:13-19)
determines whether a tool call is a messaging action:

.. code-block:: typescript

   export function isMessagingTool(toolName: string): boolean {
     if (CORE_MESSAGING_TOOLS.has(toolName)) {
       return true;
     }
     const providerId = normalizeChannelId(toolName);
     return Boolean(
       providerId && getChannelPlugin(providerId)?.actions
     );
   }

This classification is used to:

- Track which messages were sent via tools (to avoid duplicate
  delivery).
- Apply the ``SILENT_REPLY_TOKEN`` suppression when the agent
  already sent its reply via a messaging tool.


Agent Events Bus
=================

The agent event bus (``src/infra/agent-events.ts``) is an in-process
publish-subscribe system for real-time agent activity events.

Event Payload Type
-------------------

.. code-block:: typescript
   :caption: src/infra/agent-events.ts:3-12

   export type AgentEventStream =
     | "lifecycle"
     | "tool"
     | "assistant"
     | "error"
     | (string & {});

   export type AgentEventPayload = {
     runId: string;
     seq: number;          // monotonically increasing per runId
     stream: AgentEventStream;
     ts: number;           // timestamp (Date.now())
     data: Record<string, unknown>;
     sessionKey?: string;
   };

Event Streams
--------------

.. list-table:: Agent Event Streams
   :header-rows: 1
   :widths: 20 80

   * - Stream
     - Events
   * - ``lifecycle``
     - ``phase: "start"`` -- agent run started, with ``startedAt``.
       ``phase: "end"`` -- agent run completed, with ``endedAt``.
       ``phase: "error"`` -- agent run failed, with error details.
       ``phase: "fallback"`` -- model fallback occurred.
       ``phase: "fallback_cleared"`` -- returned to primary model.
   * - ``tool``
     - ``phase: "start"`` -- tool execution started, with ``name``.
       ``phase: "update"`` -- tool execution progress.
       ``phase: "end"`` -- tool execution completed.
   * - ``assistant``
     - Text deltas from the model's response.
   * - ``compaction``
     - ``phase: "end"`` -- auto-compaction completed.
   * - ``error``
     - Error events from the agent runtime.


Emitting Events
----------------

.. code-block:: typescript
   :caption: src/infra/agent-events.ts:57-78

   export function emitAgentEvent(
     event: Omit<AgentEventPayload, "seq" | "ts">
   ) {
     const nextSeq = (seqByRun.get(event.runId) ?? 0) + 1;
     seqByRun.set(event.runId, nextSeq);
     // ... enrich with sessionKey from run context ...
     for (const listener of listeners) {
       listener(enriched);
     }
   }

Events are fire-and-forget.  Listener errors are silently caught
to prevent one subscriber from breaking others.


Subscribing to Events
----------------------

.. code-block:: typescript
   :caption: src/infra/agent-events.ts:80-83

   export function onAgentEvent(
     listener: (evt: AgentEventPayload) => void
   ) {
     listeners.add(listener);
     return () => listeners.delete(listener);  // unsubscribe fn
   }

The returned function removes the listener when called.


Run Context Registry
---------------------

Each agent run can register contextual metadata that is automatically
enriched onto emitted events:

.. code-block:: typescript
   :caption: src/infra/agent-events.ts:25-43

   export function registerAgentRunContext(
     runId: string,
     context: AgentRunContext
   ) { ... }

   export type AgentRunContext = {
     sessionKey?: string;
     verboseLevel?: VerboseLevel;
     isHeartbeat?: boolean;
   };

This allows event consumers to correlate events with sessions
without the emitter needing to pass the session key every time.


Event Consumers
================

The agent event bus is consumed by several subsystems:

1. **Sub-agent registry** -- listens for ``lifecycle`` events to
   detect when sub-agent runs complete and trigger the announce flow
   (see :doc:`multi-agent`).

2. **Agent runner** -- emits ``lifecycle``, ``tool``, ``assistant``,
   and ``compaction`` events during the run for real-time monitoring.

3. **Control UI / TUI** -- subscribes to events for live status
   display and streaming output.

4. **WebSocket server** -- forwards events to connected web clients
   for real-time updates.

5. **Diagnostic system** -- records usage and performance metrics
   when diagnostics are enabled.
