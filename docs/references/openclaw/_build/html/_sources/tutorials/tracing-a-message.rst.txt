.. _tutorial-tracing-a-message:

==================
Tracing a Message
==================

This tutorial traces a single inbound message through OpenClaw's entire
processing pipeline — from channel arrival to response delivery. Follow along
to understand how every layer interacts.

Scenario
========

A user sends "What's the weather in Tokyo?" via WhatsApp. The default agent
processes it using the ``web_search`` tool and responds with current weather
data.

Step 1: Channel Receives the Message
=====================================

The WhatsApp channel plugin receives an inbound webhook from the WhatsApp
Business API. The plugin's ``ChannelMessagingAdapter`` normalizes the raw
payload into OpenClaw's internal message format:

.. code-block:: typescript

   {
     text: "What's the weather in Tokyo?",
     sender: { id: "whatsapp:+1234567890", name: "Alice" },
     channel: "whatsapp",
     sessionKey: "whatsapp:+1234567890",
     timestamp: 1708700000000,
   }

The adapter calls the gateway's ``agent`` RPC method to submit the message.

Source: Channel-specific adapter in ``src/channels/`` or ``extensions/``

Step 2: Gateway Routing
========================

The gateway server (``src/gateway/server.impl.ts``) receives the JSON-RPC
``agent`` call. It determines which agent should handle the message based on
the session key pattern.

For a direct WhatsApp message, the default agent is selected — resolved by
scanning ``agents.list[]`` for the entry with ``default: true``, or falling
back to the first entry.

Source: ``src/agents/agent-scope.ts``

Step 3: Session Resolution
===========================

The system loads or creates a session entry from ``~/.openclaw/sessions.json``.
The session is keyed by the normalized session key (e.g.,
``whatsapp:+1234567890``).

If a session already exists, it carries forward:

- Previous model and thinking level overrides
- Token usage counters
- Compaction count
- Skill snapshot reference

If this is a new conversation, a fresh session entry is created with defaults.

Source: ``src/config/sessions.ts``

Step 4: Authorization Check
============================

The system checks send policies to determine if the sender is authorized to
interact with this agent. Policies can restrict access by:

- Sender identity (allowlist/blocklist)
- Channel type
- Time-of-day rules

If the sender is unauthorized, the message is rejected with an appropriate
error response.

Step 5: Directive Parsing
==========================

The message text is scanned for directives — special prefixes that modify
agent behavior:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Directive
     - Effect
   * - ``/think high``
     - Sets extended thinking to high for this run
   * - ``/model opus``
     - Overrides the model for this session
   * - ``/verbose``
     - Enables verbose output mode

In our example, "What's the weather in Tokyo?" contains no directives, so
defaults are used.

Step 6: Queue Check
====================

The system checks whether an agent run is already active for this session.

- **No active run**: The message proceeds directly to processing.
- **Active run, steer mode**: The message is injected into the running
  conversation via ``queueEmbeddedPiMessage()``, redirecting the agent's
  current work.
- **Active run, queue mode**: The message waits in the follow-up queue until
  the current run completes.

Our message is the first in this session, so it proceeds immediately.

Source: ``src/auto-reply/reply/queue.ts``

Step 7: Memory Flush Check
============================

Before starting a new run, the system checks if a memory flush is needed.
This happens when:

1. The session's total token usage exceeds a soft threshold
   (``DEFAULT_MEMORY_FLUSH_SOFT_TOKENS = 4000``)
2. Enough new conversation has occurred since the last flush

If triggered, the agent is given a special prompt asking it to summarize
important information from the conversation into memory files
(``MEMORY.md`` or ``memory/YYYY-MM-DD.md``).

For our fresh session, no flush is needed.

Source: ``src/auto-reply/reply/memory-flush.ts``

Step 8: Model Resolution
==========================

The model is resolved through a precedence chain:

.. code-block:: text

   1. Agent-specific model   →  agents.list[].model      (not set)
   2. Session override       →  from /model directive     (not set)
   3. Channel override       →  per-channel config        (not set)
   4. Global default         →  agents.defaults.model     (not set)
   5. Hardcoded default      →  "claude-opus-4-6"         ← selected

Result: The agent will use ``claude-opus-4-6`` via the ``anthropic`` provider.

Source: ``src/agents/model-selection.ts``, ``src/agents/defaults.ts:4``

Step 9: Auth Profile Selection
================================

The auth profile system selects an API key for the chosen provider:

1. Load all configured profiles for ``anthropic``
2. Filter out profiles currently in cooldown (rate-limited)
3. Select the first available profile from the configured order

If all profiles are in cooldown, the system enters probe mode — retrying
every 30 seconds until a profile recovers.

Source: ``src/agents/auth-profiles.ts``

Step 10: System Prompt Assembly
================================

``buildEmbeddedSystemPrompt()`` constructs the system prompt from multiple
sections:

.. code-block:: text

   ┌─────────────────────────────────────┐
   │ Runtime Info (agent ID, OS, model)  │
   │ Available Tool Names                │
   │ Tool Summaries (descriptions)       │
   │ Workspace Directory                 │
   │ Workspace Notes (AGENTS.md)         │
   │ Skills Prompt (eligible skills)     │
   │ Identity (owner/sender info)        │
   │ Thinking Hints                      │
   │ Timezone & Current Time             │
   │ Memory Citations Mode               │
   │ Channel Capabilities                │
   └─────────────────────────────────────┘

The total prompt is typically 2,000–10,000 tokens depending on active tools
and skills.

Source: ``src/agents/pi-embedded-runner/system-prompt.ts``

Step 11: Agent Run (Tool Loop)
===============================

The core agent loop from ``pi-agent-core`` begins. This is a ReAct-style
loop where the LLM alternates between reasoning and acting:

.. code-block:: text

   Iteration 1:
   ├── LLM receives: system prompt + user message
   ├── LLM responds: "I'll search for current weather in Tokyo"
   ├── LLM calls: web_search({ query: "current weather Tokyo" })
   ├── Tool executes: fetches search results
   └── Tool result injected into conversation

   Iteration 2:
   ├── LLM receives: previous context + tool results
   ├── LLM responds: "The current weather in Tokyo is..."
   └── No tool calls → loop ends

The loop continues until:

- The LLM produces a response with no tool calls (natural completion)
- Maximum iterations reached (32–160, scaled by profile count)
- An unrecoverable error occurs

Source: ``src/agents/pi-embedded-runner/run.ts``

Step 12: Streaming
===================

During the agent run, partial results stream to the user in real-time:

1. **Text chunks**: Partial assistant text streams to the channel as typing
   indicators or incremental messages
2. **Tool events**: Tool call start/result events stream to the Control UI
   via WebSocket
3. **Reasoning**: If extended thinking is enabled, reasoning tokens stream
   via the ``onReasoningStream`` callback

The Agent Events bus (``src/infra/agent-events.ts``) distributes these events
to all subscribers.

Step 13: Response Delivery
===========================

Once the agent run completes:

1. The final response text is sent back through the channel adapter to
   WhatsApp
2. Token usage is recorded in the session entry
3. The conversation transcript is appended to the session's JSONL file
4. If compaction is needed (context approaching limits), older turns are
   summarized
5. The follow-up queue drains any messages that arrived during the run

The user sees: "The current weather in Tokyo is 12°C with partly cloudy
skies..."

Error Scenarios
================

What if something goes wrong at various stages?

**Model API error (429 rate limit)**:

1. Current auth profile enters cooldown
2. System selects next available profile
3. Request retries with new API key
4. If all profiles exhausted, enters probe mode

**Model API error (overloaded)**:

1. If thinking level is high, falls back: high → medium → low → minimal → off
2. Retries with reduced thinking

**Context overflow**:

1. Session auto-resets
2. Previous context is compacted (summarized)
3. Run retries with fresh context

**Tool execution error**:

1. Error is formatted as a ``ToolInputError``
2. Error result is injected back to the LLM
3. LLM can retry with corrected parameters

Summary
========

The complete path for our message:

.. code-block:: text

   WhatsApp webhook
     → Channel adapter normalizes
       → Gateway routes to default agent
         → Session loaded/created
           → Authorization passes
             → No directives found
               → No active run (proceeds)
                 → No memory flush needed
                   → Model: claude-opus-4-6
                     → Auth profile selected
                       → System prompt assembled
                         → Agent tool loop (2 iterations)
                           → web_search called
                             → Response streamed
                               → Delivered to WhatsApp
                                 → Session updated

Key source files in the path:

- ``src/gateway/server.impl.ts`` — Gateway entry point
- ``src/auto-reply/reply/get-reply.ts`` — Reply orchestration
- ``src/auto-reply/reply/agent-runner.ts`` — Agent runner setup
- ``src/auto-reply/reply/agent-runner-execution.ts`` — Execution with fallbacks
- ``src/agents/pi-embedded-runner/run.ts`` — Core tool loop
- ``src/agents/model-selection.ts`` — Model resolution
- ``src/agents/auth-profiles.ts`` — API key management
