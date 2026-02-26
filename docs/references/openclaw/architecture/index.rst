.. _architecture-index:

===========================
Architecture Overview
===========================

OpenClaw is a self-hosted, multi-channel AI gateway that connects messaging
platforms to large language models through a unified, privacy-first
architecture.  This section describes the high-level design, the principal
components, and the guiding principles behind the system.

.. contents:: On this page
   :local:
   :depth: 2


High-Level Architecture
=======================

The following diagram shows the major subsystems and how messages flow from
external chat channels through the gateway to the AI agent layer and back.

.. code-block:: text

   +--------------------+     +--------------------+     +--------------------+
   |  Chat Channels     |     |  Native Clients    |     |  HTTP / ACP        |
   |                    |     |                    |     |                    |
   |  Telegram          |     |  macOS App         |     |  /v1/responses     |
   |  Discord           |     |  iOS App           |     |  /v1/chat/compl.   |
   |  Slack             |     |  Android App       |     |  Agent Client      |
   |  WhatsApp          |     |  TUI (terminal)    |     |  Protocol (ACP)    |
   |  Signal            |     |  Control UI (web)  |     |                    |
   |  iMessage           |     +--------+-----------+     +--------+-----------+
   |  LINE / MS Teams   |              |                          |
   |  IRC / Matrix      |              |                          |
   |  + extension chs.  |              |                          |
   +--------+-----------+              |                          |
            |                          |                          |
            |     WebSocket / HTTP     |                          |
            +----------+---------------+--------------------------+
                       |
                       v
          +---------------------------+
          |      Gateway Server       |
          |                           |
          |  - WebSocket multiplexer  |
          |  - RPC method router      |
          |  - Auth (token/password/  |
          |    trusted-proxy/device)  |
          |  - Session management     |
          |  - Config hot-reload      |
          |  - Channel health monitor |
          |  - TLS termination        |
          |  - mDNS / Tailscale       |
          +--------+--+--+-----------+
                   |  |  |
          +--------+  |  +---------+
          |           |            |
          v           v            v
   +-----------+ +-----------+ +-----------+
   |  Agent    | |  Plugins  | |  Skills   |
   |  Runtime  | |  Runtime  | |  Loader   |
   |           | |           | |           |
   | - Pi core | | - Channel | | - Bundled |
   | - Tools   | |   plugins | |   skills  |
   | - Sandbox | | - Memory  | | - Custom  |
   | - Cron    | |   plugins | |   dirs    |
   +-----------+ +-----------+ +-----------+
          |           |
          v           v
   +-----------------------------+
   |   LLM Providers             |
   |                             |
   |  Anthropic  |  OpenAI       |
   |  Google     |  Bedrock      |
   |  Ollama     |  Local LLMs   |
   +-----------------------------+


Message Flow
============

1. **Inbound** -- A user sends a message on a chat channel (e.g., Telegram,
   Discord, WhatsApp) or through the Control UI / native app.

2. **Channel adapter** -- The channel-specific adapter normalises the message
   into an internal ``InboundMessage`` shape, resolves the session key
   (channel + account + peer), and forwards it to the gateway.

3. **Gateway routing** -- The gateway determines which agent should handle the
   message using :term:`agent bindings <AgentBinding>`, applies allowlists,
   and enqueues the message into the agent's session lane.

4. **Agent execution** -- The selected agent streams the user's message to
   an LLM provider.  The agent runtime manages tool calls, memory injection,
   skill prompts, and sandbox isolation.

5. **Outbound** -- The agent's reply is routed back through the channel
   adapter and delivered as one or more messages on the originating platform.

6. **Side-effects** -- Session state, logs, and memory updates are persisted
   under ``~/.openclaw/``.


Design Principles
=================

Self-hosted & Privacy-first
---------------------------

All data stays on the operator's machine.  There is no cloud backend, no
telemetry phone-home, and no account creation required.  Credentials and
conversation history are stored locally under ``~/.openclaw/``.

Multi-channel
-------------

OpenClaw supports **10+ messaging channels** out of the box (Telegram,
Discord, Slack, WhatsApp, Signal, iMessage, LINE, MS Teams, IRC, Google Chat)
and provides a :doc:`plugin SDK <../channels/plugin-sdk>` for adding new ones
as npm packages.

Plugin-extensible
-----------------

The plugin system allows third-party packages to:

- Add new chat channels (e.g., Matrix, Nostr, Mattermost)
- Provide custom memory backends (e.g., LanceDB)
- Expose additional gateway RPC methods and HTTP endpoints
- Ship custom agent tools

Plugins are installed via ``openclaw plugins install <name>`` and run in the
same process as the gateway.

Multi-agent
-----------

Multiple :term:`agents <Agent>` can run concurrently, each with its own
workspace, model configuration, tool set, and skill allowlist.
:term:`Agent bindings <AgentBinding>` route messages to agents based on
channel, account, peer, guild, or Discord role.

Composable CLI
--------------

The ``openclaw`` CLI is built on `Commander.js <https://github.com/tj/commander.js>`_
with lazy command registration so startup stays fast even as the command tree
grows.  Every gateway operation can be performed from the command line,
enabling headless and scripted deployments.


Component Summary
=================

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Component
     - Responsibility
     - Key paths
   * - Gateway Server
     - WebSocket + HTTP listener, auth, RPC dispatch, session management
     - ``src/gateway/``
   * - Channel Adapters
     - Normalise inbound/outbound for each messaging platform
     - ``src/telegram/``, ``src/discord/``, ``src/slack/``, ``src/web/``, etc.
   * - Agent Runtime
     - Orchestrate LLM calls, tools, sandbox, streaming
     - ``src/agents/``
   * - Plugin Runtime
     - Discover, load, and lifecycle-manage npm plugins
     - ``src/plugins/``
   * - Skill Loader
     - Load skill prompts from bundled + custom directories
     - ``src/agents/skills/``, ``skills/``
   * - Config System
     - JSON5 config loading, Zod validation, env overrides, hot-reload
     - ``src/config/``
   * - CLI Framework
     - Commander.js program, lazy command registration, sub-CLIs
     - ``src/cli/``
   * - Control UI
     - Browser-based dashboard served by the gateway
     - ``ui/``
   * - Native Apps
     - macOS, iOS, Android clients
     - ``apps/``


Sub-pages
=========

.. toctree::
   :maxdepth: 1

   project-structure
   technology-stack
   entry-points
   configuration
   data-models
