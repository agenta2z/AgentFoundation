==============================
Directory Layout & Module Map
==============================

This page documents the repository directory structure and the purpose of
each major module under ``src/``.

.. contents:: On this page
   :local:
   :depth: 2


Top-Level Directory Tree
========================

.. code-block:: text

   openclaw/
   +-- openclaw.mjs          # CLI entry shim (enables compile cache, loads dist/entry.js)
   +-- package.json           # npm metadata, scripts, dependency declarations
   +-- pnpm-workspace.yaml    # pnpm workspace config (monorepo)
   +-- tsconfig.json          # TypeScript project config
   +-- tsdown.config.ts       # Bundle / build config (tsdown)
   +-- vitest.config.ts       # Test runner config (Vitest)
   +-- Dockerfile             # Production container image
   +-- docker-compose.yml     # Local multi-container dev setup
   +-- fly.toml               # Fly.io deployment manifest
   +-- render.yaml            # Render deployment manifest
   +--
   +-- src/                   # TypeScript source (core gateway + CLI)
   +-- ui/                    # Control UI (Lit web components, Vite)
   +-- extensions/            # Channel & feature plugins (workspace packages)
   +-- apps/                  # Native client applications
   |   +-- macos/             #   macOS menubar app (SwiftUI)
   |   +-- ios/               #   iOS app (SwiftUI)
   |   +-- android/           #   Android app (Kotlin)
   |   +-- shared/            #   Shared Swift package (OpenClawKit)
   +-- skills/                # Bundled skill definitions (~60 skills)
   +-- scripts/               # Build, release, testing, and dev helper scripts
   +-- vendor/                # Vendored third-party assets (a2ui)
   +-- patches/               # pnpm patch overrides for upstream packages
   +-- docs/                  # Documentation source (Mintlify)
   +-- test/                  # Shared test fixtures
   +-- packages/              # Internal workspace packages
   +-- git-hooks/             # Git hook scripts (pre-commit, etc.)
   +-- assets/                # Static assets (images, icons)


Source Module Map (``src/``)
============================

The table below describes each top-level module under ``src/``.  Modules are
listed alphabetically.  Tests are colocated as ``*.test.ts`` files.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Module
     - Responsibility
     - Key files / sub-modules
   * - ``acp/``
     - Agent Client Protocol (ACP) server & client implementation
     - ACP SDK integration (``@agentclientprotocol/sdk``)
   * - ``agents/``
     - Agent runtime: LLM orchestration, tool invocation, Pi embedded
       runner, skills, sandbox, auth profiles, schema definitions
     - ``cli-runner/``, ``pi-embedded-runner/``, ``pi-extensions/``,
       ``sandbox/``, ``schema/``, ``skills/``, ``tools/``
   * - ``auto-reply/``
     - Outbound reply pipeline: templating, chunking, streaming coalesce,
       history tracking, reply tokens
     - ``reply/``, ``templating.ts``, ``chunk.ts``, ``tokens.ts``
   * - ``browser/``
     - Headless browser management (Playwright), CDP proxy, profiles
     - ``routes/``, browser lifecycle
   * - ``canvas-host/``
     - Canvas artifact server (A2UI), live-reload dev server
     - ``a2ui/``
   * - ``channels/``
     - Shared channel infrastructure: plugin types, allowlists, mention
       gating, session recording, ack reactions, typing callbacks, registry
     - ``plugins/``, ``allowlists/``, ``telegram/``, ``web/``
   * - ``cli/``
     - CLI framework: Commander.js program builder, lazy command
       registration, sub-CLI routers, progress bars, prompts
     - ``program/``, ``browser-cli-actions-input/``, ``cron-cli/``,
       ``daemon-cli/``, ``gateway-cli/``, ``node-cli/``, ``nodes-cli/``,
       ``update-cli/``, ``shared/``
   * - ``commands/``
     - High-level CLI command implementations: agent, channels, models,
       onboarding, gateway-status, status-all
     - ``agent/``, ``channels/``, ``models/``, ``onboarding/``
   * - ``compat/``
     - Backward-compatibility shims for renamed APIs
     - (minimal)
   * - ``config/``
     - Configuration system: JSON5 loading, Zod schema validation, path
       resolution, runtime overrides, legacy migration, session store
     - ``paths.ts``, ``zod-schema.ts``, ``types.*.ts``, ``io.ts``,
       ``validation.ts``, ``sessions/``
   * - ``cron/``
     - Cron job scheduler: job store, isolated agent runs, service
       lifecycle
     - ``service/``, ``isolated-agent/``
   * - ``daemon/``
     - Gateway daemon/service management (launchd, systemd)
     - Daemon install/uninstall, status
   * - ``discord/``
     - Discord channel adapter: bot client, voice, monitor, thread
       bindings
     - ``monitor/``, ``voice/``
   * - ``gateway/``
     - Gateway server core: WebSocket listener, HTTP endpoints, RPC
       method dispatch, protocol schemas (TypeBox/AJV), server lifecycle,
       TLS, health state, presence
     - ``protocol/``, ``server/``, ``server-methods/``
   * - ``hooks/``
     - Webhook / event hooks: bundled hook presets, hook mapping engine
     - ``bundled/``
   * - ``imessage/``
     - iMessage channel adapter: AppleScript bridge, BlueBubbles
       integration
     - ``monitor/``
   * - ``infra/``
     - Infrastructure utilities: env normalization, dotenv loading, port
       management, HTTP body parsing, TLS helpers, network SSRF guard,
       binary management, home-dir resolution, error formatting
     - ``net/``, ``outbound/``, ``tls/``, ``format-time/``
   * - ``line/``
     - LINE channel adapter: bot SDK, flex message templates, markdown
       conversion
     - ``flex-templates/``
   * - ``link-understanding/``
     - URL content extraction (Readability, PDF parsing)
     - Link fetch and summarization
   * - ``logging/``
     - Structured logging: tslog transport, file rotation, redaction,
       console capture
     - ``logger.ts``, ``redact.ts``
   * - ``markdown/``
     - Markdown processing utilities (markdown-it integration)
     - Rendering helpers
   * - ``media/``
     - Media file handling: MIME detection, file storage, temp paths
     - ``mime.ts``, ``store.ts``
   * - ``media-understanding/``
     - Image/audio/video analysis via LLM vision or transcription
     - ``providers/``
   * - ``memory/``
     - Memory subsystem: built-in vector store (sqlite-vec), QMD
       integration, citation engine, session export
     - Full-text + vector search, embedding pipeline
   * - ``node-host/``
     - Remote node host service: browser proxy, command execution
     - Node pairing, invoke protocol
   * - ``pairing/``
     - Secure device and DM pairing flows
     - Pairing approval/rejection
   * - ``plugin-sdk/``
     - Public plugin SDK: types, helpers, and re-exports for extension
       authors
     - ``index.ts`` (barrel export)
   * - ``plugins/``
     - Plugin runtime: discovery, loading, lifecycle, HTTP route
       registration, CLI integration
     - ``runtime/``
   * - ``process/``
     - Process management: child process bridge, exec helpers, supervisor
     - ``supervisor/``
   * - ``providers/``
     - LLM provider abstractions and auth helpers
     - Provider registry
   * - ``routing/``
     - Message routing: session key derivation, route resolution
     - ``session-key.ts``, ``resolve-route.ts``
   * - ``security/``
     - Security utilities: DM policy enforcement, config auditing
     - ``dm-policy-shared.ts``
   * - ``sessions/``
     - Session persistence and compaction
     - Session store I/O
   * - ``shared/``
     - Shared utilities used across modules (net helpers, text utilities)
     - ``net/``, ``text/``
   * - ``signal/``
     - Signal channel adapter: signal-cli integration, monitor
     - ``monitor/``
   * - ``slack/``
     - Slack channel adapter: Bolt framework, HTTP mode, message actions,
       monitor
     - ``http/``, ``monitor/``
   * - ``telegram/``
     - Telegram channel adapter: grammY framework, bot management,
       outbound
     - ``bot/``
   * - ``terminal/``
     - Terminal rendering: ANSI helpers, table formatter, link formatting,
       color palette
     - ``table.ts``, ``palette.ts``, ``ansi.ts``
   * - ``tts/``
     - Text-to-speech: Edge TTS, ElevenLabs integration
     - TTS provider abstraction
   * - ``tui/``
     - Terminal UI (ink-based interactive interface)
     - ``components/``, ``theme/``
   * - ``types/``
     - Shared TypeScript type declarations
     - Utility types
   * - ``utils/``
     - General-purpose utilities: E.164 normalization, JSON parsing,
       regex escaping, sleep, clamping
     - ``utils.ts``
   * - ``web/``
     - WhatsApp Web (Baileys) channel adapter: QR pairing, auto-reply,
       inbound processing
     - ``auto-reply/``, ``inbound/``
   * - ``whatsapp/``
     - WhatsApp shared utilities: target normalization, JID handling
     - ``normalize.ts``
   * - ``wizard/``
     - Interactive onboarding/setup wizard (multi-step flows)
     - Wizard prompts and state machine


Extensions Directory (``extensions/``)
======================================

Extensions are workspace packages that implement channel plugins or feature
plugins.  Each extension has its own ``package.json`` and lives under
``extensions/<name>/``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Extension
     - Description
   * - ``bluebubbles``
     - BlueBubbles iMessage bridge integration
   * - ``discord``
     - Extended Discord features (beyond core)
   * - ``feishu``
     - Feishu / Lark channel plugin
   * - ``googlechat``
     - Google Chat channel plugin
   * - ``imessage``
     - Extended iMessage features
   * - ``irc``
     - IRC channel plugin
   * - ``line``
     - LINE channel plugin
   * - ``matrix``
     - Matrix channel plugin
   * - ``mattermost``
     - Mattermost channel plugin
   * - ``msteams``
     - Microsoft Teams channel plugin
   * - ``nostr``
     - Nostr protocol channel plugin
   * - ``signal``
     - Extended Signal features
   * - ``slack``
     - Extended Slack features
   * - ``synology-chat``
     - Synology Chat channel plugin
   * - ``telegram``
     - Extended Telegram features
   * - ``tlon``
     - Tlon (Urbit) channel plugin
   * - ``twitch``
     - Twitch channel plugin
   * - ``voice-call``
     - Voice call support (WebRTC / SIP)
   * - ``whatsapp``
     - Extended WhatsApp features
   * - ``zalo`` / ``zalouser``
     - Zalo channel plugins
   * - ``memory-core``
     - Core memory plugin interfaces
   * - ``memory-lancedb``
     - LanceDB vector memory backend
   * - ``diagnostics-otel``
     - OpenTelemetry diagnostics plugin
   * - ``lobster``
     - Terminal theming / UI extension
   * - ``copilot-proxy``
     - GitHub Copilot proxy plugin
   * - ``device-pair``
     - Device pairing extension
   * - ``llm-task``
     - LLM task runner plugin
   * - ``open-prose``
     - Prose editing assistant plugin
   * - ``phone-control``
     - Phone control plugin
   * - ``talk-voice``
     - Voice/talk mode plugin
   * - ``thread-ownership``
     - Thread ownership management plugin
   * - ``nextcloud-talk``
     - Nextcloud Talk channel plugin


Skills Directory (``skills/``)
==============================

The ``skills/`` directory contains ~60 bundled skill definitions.  Each skill
is a directory with a prompt file and optional metadata.  Skills extend the
agent's capabilities without code changes.  Examples include:

- ``1password`` -- 1Password CLI integration
- ``canvas`` -- Canvas artifact creation
- ``coding-agent`` -- Code generation skill
- ``discord`` -- Discord-specific commands
- ``gh-issues`` -- GitHub issue management
- ``github`` -- GitHub CLI skill
- ``obsidian`` -- Obsidian note management
- ``slack`` -- Slack-specific commands
- ``weather`` -- Weather lookup
- ``peekaboo`` -- Screenshot tool

Skills can also be loaded from custom directories via the
:ref:`skills.load.extraDirs <config-skills>` configuration option.
