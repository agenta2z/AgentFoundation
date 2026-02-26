.. _orchestration-overview:

=======================
Orchestration Overview
=======================

What "Orchestration" Means in OpenClaw
=======================================

In OpenClaw, **orchestration** is the entire machinery that transforms an
inbound user message into an agent-produced response -- spanning channel
ingestion, session resolution, model selection, tool execution, sub-agent
coordination, and final delivery back to the user.

Unlike traditional agent frameworks that rely on an explicit planner
component, OpenClaw's architecture embodies a single key insight:

.. note::

   **The LLM *is* the planner.**  There is no separate planning module,
   DAG scheduler, or state machine that decides which tools to call.
   The language model itself decides what to do next by emitting tool
   calls in a ReAct-style loop until it produces a final text response.

This means the orchestration layer is primarily concerned with
*infrastructure*: routing messages, managing sessions, resolving models,
enforcing limits, and shuttling tool calls between the model and the
runtime environment.


The ReAct-Style Tool Loop
==========================

OpenClaw uses the ``pi-agent-core`` library to drive a ReAct (Reason +
Act) loop.  Each iteration:

1. The model receives the conversation history plus available tool
   definitions.
2. The model either produces a final text response (terminating the
   loop) or emits one or more **tool calls**.
3. The runtime executes each tool call and appends the results back
   into the conversation.
4. The model is called again with the updated history.

This loop continues until the model emits a final assistant message
(with no tool calls), or until a guard condition fires (timeout,
iteration limit, context overflow, or abort signal).

The loop is executed inside ``runEmbeddedPiAgent()``
(``src/agents/pi-embedded-runner/run.ts``) for the embedded provider
path, or delegated to an external CLI backend via ``runCliAgent()``
(``src/agents/cli-runner.ts``) for CLI providers like ``claude-cli``
or ``codex-cli``.


Full Agent Pipeline
====================

The following diagram shows the full path a message takes from channel
arrival to response delivery.  Each numbered step is detailed in
:doc:`request-lifecycle`.

.. code-block:: text

   User Message
        |
        v
   +--------------------+
   | 1. Channel Inbound |  (Telegram / Discord / Slack / Signal / Web / ...)
   +--------------------+
        |
        v
   +--------------------+
   | 2. Routing          |  Determine target session & agent
   +--------------------+
        |
        v
   +--------------------+
   | 3. Session          |  Resolve / create session entry (sessions.json)
   |    Resolution       |
   +--------------------+
        |
        v
   +--------------------+
   | 4. Auth Check       |  Command authorization, owner gating
   +--------------------+
        |
        v
   +--------------------+
   | 5. Directive Parse  |  Inline directives (/model, /think, /verbose ...)
   +--------------------+
        |
        v
   +--------------------+
   | 6. Queue Check      |  steer / followup / collect / interrupt
   +--------------------+
        |
        v
   +--------------------+
   | 7. Memory Flush     |  Persist pending memory writes before run
   +--------------------+
        |
        v
   +--------------------+
   | 8. Model Resolution |  agent -> session -> channel -> global -> default
   |    (claude-opus-4-6)|  (see: model-selection.ts, model-catalog.ts)
   +--------------------+
        |
        v
   +--------------------+
   | 9. Auth Profile     |  API key / OAuth token selection & rotation
   +--------------------+
        |
        v
   +======================+
   | 10. Agent Run        |  <--- ReAct tool loop lives here
   |     (pi-agent-core)  |
   |                      |
   |  System Prompt       |
   |  + User Message      |
   |  + Tool Definitions  |
   |       |              |
   |       v              |
   |  [LLM Call] <-----+  |
   |       |            |  |
   |   tool_use?        |  |
   |    yes -> execute --+ |
   |    no  -> final text  |
   +======================+
        |
        v
   +--------------------+
   | 11. Streaming       |  Block replies, partial text, typing indicators
   +--------------------+
        |
        v
   +--------------------+
   | 12. Compaction      |  Auto-compact if context nears window limit
   +--------------------+
        |
        v
   +--------------------+
   | 13. Response /      |  Format payloads, deliver to originating channel
   |     Delivery        |
   +--------------------+
        |
        v
   User receives reply


Key Architectural Properties
=============================

**Stateless model calls, stateful sessions.**
  Each LLM API call is stateless.  Conversation history is reconstructed
  from JSONL transcript files on disk before every call.

**Session-scoped isolation.**
  Each session has its own transcript, model override, thinking level,
  and workspace.  Sub-agents get their own sessions (see :doc:`multi-agent`).

**Provider-agnostic model layer.**
  The model selection chain (see :doc:`llm-integration`) supports
  Anthropic, OpenAI, Google, and many other providers through a
  unified interface.

**Fail-safe fallback.**
  When the primary model fails (rate-limit, auth error, timeout),
  the fallback chain (see :doc:`error-handling`) tries alternative
  models before surfacing an error.

**Plugin-extensible tools.**
  The tool catalog (see :doc:`tool-system`) includes built-in tools
  and supports runtime registration of plugin tools with a layered
  policy pipeline.


Document Map
=============

.. toctree::
   :maxdepth: 1
   :caption: Orchestration Topics

   agent-definition
   request-lifecycle
   multi-agent
   communication
   tool-system
   error-handling
   state-management
   llm-integration
   system-prompt
