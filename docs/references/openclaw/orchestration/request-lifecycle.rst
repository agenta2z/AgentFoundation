.. _request-lifecycle:

=================================
Full 13-Step Request Flow
=================================

This tutorial-style document traces a single user message from the
moment it arrives at a channel adapter to the moment the response
is delivered back.  The primary source files are:

- ``src/auto-reply/reply/get-reply.ts`` -- top-level reply
  orchestration
- ``src/auto-reply/reply/agent-runner.ts`` -- agent run setup and
  post-processing
- ``src/auto-reply/reply/agent-runner-execution.ts`` -- model
  fallback and the embedded run loop

.. contents:: On this page
   :local:
   :depth: 2


Flow Diagram
=============

.. code-block:: text

    User sends "Summarize this PR" on Telegram
         |
         v
   [1] Channel Inbound   -- Telegram adapter receives update
         |
   [2] Routing            -- map chat ID -> session key
         |
   [3] Session Resolution -- load / create session entry
         |
   [4] Auth Check         -- is sender authorized?
         |
   [5] Directive Parse    -- extract !model, !think, !verbose
         |
   [6] Queue Check        -- steer into active run, or enqueue?
         |
   [7] Memory Flush       -- persist pending memory before run
         |
   [8] Model Resolution   -- pick provider + model
         |
   [9] Auth Profile       -- select API key / OAuth token
         |
   [10] Agent Run         -- ReAct tool loop (LLM + tools)
         |
   [11] Streaming         -- typing indicators, block replies
         |
   [12] Compaction        -- auto-compact if near context limit
         |
   [13] Response/Delivery -- format and send back to Telegram


Step-by-Step Walkthrough
=========================


Step 1: Channel Inbound
------------------------

Each messaging channel (Telegram, Discord, Slack, Signal, iMessage,
Web, WhatsApp, IRC, MS Teams, etc.) has an adapter that normalizes
the incoming message into a ``MsgContext`` object.

The context includes:

- ``Body`` -- the message text
- ``From`` / ``To`` -- sender and recipient identifiers
- ``Provider`` -- channel identifier (``"telegram"``, ``"discord"``, ...)
- ``SessionKey`` -- session routing key (may be pre-computed by the
  channel)
- ``ThreadId`` -- thread/topic identifier for threaded channels
- ``MediaUrl`` -- attached media URLs
- ``CommandSource`` -- ``"native"`` for slash commands, ``"text"`` for
  inline commands

The ``finalizeInboundContext()`` function
(``src/auto-reply/reply/inbound-context.ts``) normalizes and enriches
the raw context before further processing.


Step 2: Routing
----------------

The session key determines which agent and session handle the message.
``resolveSessionAgentId()`` (``src/agents/agent-scope.ts``:98-103)
extracts the agent ID from the session key, falling back to the
default agent.

For session keys of the form ``agent:<id>:subagent:<uuid>``, the
embedded agent ID is extracted.  Otherwise, the default agent is used.


Step 3: Session Resolution
---------------------------

``initSessionState()`` (``src/auto-reply/reply/session.ts``)
loads or creates the session entry from the session store
(``sessions.json``):

- Loads the session store from disk (per-agent path).
- Looks up or creates the ``SessionEntry`` for this session key.
- Assigns a new ``sessionId`` (UUID) for new sessions.
- Resolves the transcript file path (JSONL).
- Detects session resets (``/new`` command).
- Determines group vs. DM context.
- Returns the ``sessionKey``, ``sessionId``, ``sessionEntry``,
  ``sessionStore``, ``storePath``, and related state.


Step 4: Authorization
----------------------

``resolveCommandAuthorization()``
(``src/auto-reply/command-auth.ts``) checks whether the sender is
authorized to use the agent:

- Owner-only commands are gated by the configured owner identifiers.
- DM policy and group activation rules are evaluated.
- The ``commandAuthorized`` flag is propagated to downstream steps.


Step 5: Directive Parsing
--------------------------

``resolveReplyDirectives()``
(``src/auto-reply/reply/get-reply-directives.ts``) scans the message
body for inline directives:

- ``!model <name>`` -- switch model for this session
- ``!think <level>`` -- set thinking/reasoning level
- ``!verbose`` -- toggle verbose output
- ``!reset`` -- reset session
- ``!elevated`` -- enable elevated bash execution

Directives are stripped from the message body before it is passed to
the agent.  Some directives produce an immediate reply (e.g.,
``!status``) and short-circuit the pipeline.


Step 6: Queue Check
--------------------

OpenClaw supports several queue modes for handling messages that arrive
while an agent run is already in progress:

.. list-table:: Queue Modes
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Behavior
   * - ``steer``
     - Inject the new message into the active run's context via
       ``queueEmbeddedPiMessage()``.
   * - ``followup``
     - Enqueue the message; run it after the current run completes.
   * - ``collect``
     - Batch multiple messages into a single follow-up run.
   * - ``steer-backlog``
     - Try steering first; if not possible, fall back to follow-up.
   * - ``interrupt``
     - Abort the current run and start a new one.

Queue settings are resolved per-channel via ``resolveQueueSettings()``
(``src/auto-reply/reply/queue/settings.ts``).

If the message is steered or enqueued, the function returns early --
no new agent run is started.


Step 7: Memory Flush
---------------------

``runMemoryFlushIfNeeded()``
(``src/auto-reply/reply/agent-runner-memory.ts``) persists any
pending memory writes before the agent run starts.  This ensures the
agent sees the most recent memory state when it calls
``memory_search`` or ``memory_get`` during the run.


Step 8: Model Resolution
--------------------------

The model is resolved through a priority chain:

.. code-block:: text

   Per-agent explicit model (agents.list[].model)
     |
     v  (fallback)
   Session model override (sessionEntry.modelOverride)
     |
     v  (fallback)
   Channel model override (channels.<channel>.model)
     |
     v  (fallback)
   Global default (agents.defaults.model)
     |
     v  (fallback)
   Hardcoded default: anthropic/claude-opus-4-6

This is implemented across ``resolveDefaultModel()``
(``src/auto-reply/reply/directive-handling.ts``) and
``resolveAgentEffectiveModelPrimary()``
(``src/agents/agent-scope.ts``:170-178).

The default provider is ``"anthropic"`` and the default model is
``"claude-opus-4-6"`` (``src/agents/defaults.ts``:3-4).


Step 9: Auth Profile Selection
-------------------------------

``getApiKeyForModel()`` / ``resolveAuthProfileOrder()``
(``src/agents/model-auth.ts``, ``src/agents/auth-profiles.ts``)
select the appropriate API key or OAuth credential:

- Multiple auth profiles can be configured per provider.
- Profiles in cooldown (from rate-limit errors) are skipped.
- Round-robin ordering prevents any single profile from being
  over-utilized.
- The selected profile ID is recorded in ``FollowupRun.run.authProfileId``.

See :doc:`llm-integration` for details on the auth profile system.


Step 10: Agent Run
-------------------

This is the core of the pipeline -- the ReAct tool loop.

``runAgentTurnWithFallback()``
(``src/auto-reply/reply/agent-runner-execution.ts``:67-586) wraps
the run in the model fallback chain:

1. Build the candidate list (primary + fallbacks) via
   ``resolveFallbackCandidates()``.
2. For each candidate, call ``runEmbeddedPiAgent()`` or
   ``runCliAgent()``.
3. If the run succeeds, break out of the loop.
4. If the run fails with a failover-eligible error, record the
   attempt and try the next candidate.
5. If all candidates fail, throw a summary error.

Inside ``runEmbeddedPiAgent()``
(``src/agents/pi-embedded-runner/run.ts``), the actual tool loop runs:

- The system prompt is assembled (see :doc:`system-prompt`).
- Tool definitions are registered (see :doc:`tool-system`).
- The pi-agent-core ``AgentSession`` drives the ReAct loop.
- Tool calls are executed, results appended, and the model is called
  again.
- Streaming callbacks emit partial text and tool events.

.. code-block:: typescript
   :caption: src/auto-reply/reply/agent-runner-execution.ts:172-174

   const fallbackResult = await runWithModelFallback({
     ...resolveModelFallbackOptions(params.followupRun.run),
     run: (provider, model) => { ... },
   });


Step 11: Streaming
-------------------

During the agent run, several streaming mechanisms are active:

- **Typing indicators**: ``createTypingSignaler()`` sends platform
  typing events at configured intervals.
- **Block replies**: ``createBlockReplyPipeline()`` splits long
  responses into paragraph-sized chunks and delivers them
  incrementally.
- **Partial replies**: ``onPartialReply`` callbacks emit text deltas
  as the model generates tokens.
- **Tool events**: ``onAgentEvent`` emits structured events for tool
  start/end, compaction, and lifecycle phases.


Step 12: Compaction
--------------------

If the conversation context approaches the model's context window
limit, auto-compaction is triggered:

- ``compactEmbeddedPiSessionDirect()`` summarizes older messages to
  free tokens.
- The compaction result is tracked via ``autoCompactionCompleted``.
- A post-compaction workspace context injection and read audit may
  follow.
- If compaction itself fails (context still too large), the session
  is auto-reset (line 347-352).


Step 13: Response and Delivery
-------------------------------

After the agent run completes:

1. ``buildReplyPayloads()`` normalizes the raw payloads -- stripping
   silent tokens, heartbeat markers, and applying reply-to-mode
   filters.
2. Fallback state transitions are recorded (if the model fell back
   to an alternate).
3. Usage statistics are persisted via ``persistRunSessionUsage()``.
4. Verbose notices (new session, compaction, fallback) are prepended
   if verbose mode is on.
5. Response usage lines are appended if configured.
6. ``finalizeWithFollowup()`` checks for queued follow-up messages
   and schedules the next run if needed.
7. The final ``ReplyPayload`` (or array of payloads) is returned to
   the channel adapter, which delivers it to the user.

.. code-block:: typescript
   :caption: src/auto-reply/reply/agent-runner.ts:726-730

   return finalizeWithFollowup(
     finalPayloads.length === 1 ? finalPayloads[0] : finalPayloads,
     queueKey,
     runFollowupTurn,
   );

.. warning::

   Streaming and partial replies are only sent to internal UIs
   (control channel, TUI, WebSocket).  External messaging surfaces
   (WhatsApp, Telegram, etc.) receive only the final consolidated
   reply to avoid message flooding.
