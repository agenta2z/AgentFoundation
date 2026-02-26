.. _gateway-control-ui:

==========
Control UI
==========

The Control UI is a browser-based dashboard for managing OpenClaw, built with
Lit v3 web components and served by the gateway.

Technology
==========

- **Framework**: Lit v3 (Web Components)
- **Build tool**: Vite
- **Transport**: WebSocket (real-time updates)
- **Testing**: Vitest with browser mode (Playwright)

Source: ``ui/``

Features
========

The Control UI provides:

- **Chat interface**: Send messages to agents and view responses
- **Session management**: View, switch, and delete conversation sessions
- **Channel status**: Monitor connected channels and their health
- **Configuration**: View and modify OpenClaw settings
- **Agent management**: Switch between configured agents
- **Tool events**: Real-time view of tool calls during agent execution
- **Node connections**: Manage paired mobile/desktop companion apps

Authentication
==============

The Control UI authenticates using the same gateway auth token. On first
access, the user enters the token, which is stored in browser local storage
for subsequent visits.

The UI is served as static files from the gateway, so it's automatically
available at the gateway URL (default: ``http://localhost:<port>``).

WebSocket Communication
=======================

The UI maintains a persistent WebSocket connection to the gateway for:

- Streaming agent responses (partial text as it's generated)
- Tool execution events (tool name, arguments, results)
- Reasoning stream (model's internal chain-of-thought, when available)
- Session state updates
- Channel status changes

Source: ``ui/``, ``src/gateway/server.impl.ts``
