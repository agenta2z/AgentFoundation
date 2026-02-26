.. _gateway-index:

===============
Gateway Server
===============

The gateway is the central hub of OpenClaw — an Express v5 + WebSocket server
that handles all communication between channels, agents, and the Control UI.

Architecture
============

.. code-block:: text

   ┌──────────────────────────────────────────────────────────┐
   │                    Gateway Server                        │
   │                                                          │
   │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
   │  │  Express v5  │  │  WebSocket   │  │  Static Files │  │
   │  │  HTTP API    │  │  (ws)        │  │  Control UI   │  │
   │  └──────┬──────┘  └──────┬───────┘  └───────────────┘  │
   │         │                │                              │
   │  ┌──────▼────────────────▼──────────────────────────┐   │
   │  │           JSON-RPC Protocol Layer                │   │
   │  │  agent, agent.wait, sessions.*, chat.*, ...     │   │
   │  └──────────────────────┬───────────────────────────┘   │
   │                         │                               │
   │  ┌──────────────────────▼───────────────────────────┐   │
   │  │              Agent Orchestration                 │   │
   │  │  Model resolution, tool loop, session mgmt      │   │
   │  └──────────────────────────────────────────────────┘   │
   └──────────────────────────────────────────────────────────┘

Server Startup
==============

The gateway starts via ``startGatewayServer()`` in
``src/gateway/server.impl.ts``:

1. Create Express app with middleware (auth, CORS, CSP, body parsing)
2. Set up WebSocket server on the same HTTP server
3. Register JSON-RPC method handlers
4. Mount static file serving for the Control UI
5. Start channel monitors (Telegram, Discord, Slack, WhatsApp, etc.)
6. Begin listening on configured port (``gateway.port``)

Authentication
==============

The gateway uses token-based authentication:

- **Gateway auth token**: Configured via ``gateway.authToken`` in
  ``openclaw.json`` or the ``OPENCLAW_GATEWAY_AUTH_TOKEN`` env var
- All HTTP and WebSocket requests must include the token
- The Control UI authenticates via the same token mechanism

Binding Modes
=============

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``loopback``
     - Bind to ``127.0.0.1`` only (default, most secure)
   * - ``lan``
     - Bind to ``0.0.0.0`` for LAN access
   * - Custom
     - Bind to a specific interface address

Source: ``src/gateway/server.ts``, ``src/gateway/server.impl.ts``

.. toctree::
   :maxdepth: 2

   rpc-protocol
   control-ui
