=======================
Configuration System
=======================

This page documents the OpenClaw configuration system: file locations, format,
environment variable overrides, schema validation, and path resolution.

.. contents:: On this page
   :local:
   :depth: 2


Overview
========

OpenClaw uses a single JSON5 configuration file that controls the gateway,
channels, agents, plugins, and all subsystems.  The config is validated at
load time against a Zod schema (``OpenClawSchema``) and supports hot-reload
during gateway operation.

.. code-block:: text

   +-------------------+     +------------------+     +-----------------+
   |  Config File      |     |  Environment     |     |  CLI Flags      |
   |  (JSON5)          |     |  Variables       |     |                 |
   +--------+----------+     +--------+---------+     +--------+--------+
            |                         |                         |
            v                         v                         v
   +----------------------------------------------------------+
   |                  Config Resolution Pipeline               |
   |                                                           |
   |  1. Read JSON5 file                                       |
   |  2. Resolve $include directives                           |
   |  3. Substitute ${ENV_VAR} placeholders                    |
   |  4. Validate against OpenClawSchema (Zod)                 |
   |  5. Apply environment variable overrides                  |
   |  6. Apply CLI flag overrides                              |
   |  7. Apply runtime defaults                                |
   +----------------------------------------------------------+
            |
            v
   +-------------------+
   |  OpenClawConfig   |
   |  (typed object)   |
   +-------------------+


Config File Location
====================

The default config file path is:

.. code-block:: text

   ~/.openclaw/openclaw.json

The file uses **JSON5** syntax, which supports:

- Single-line and multi-line comments (``//``, ``/* */``)
- Trailing commas
- Unquoted property names
- Single-quoted strings

Path Resolution Order
---------------------

The config path is resolved using the following precedence:

1. **``OPENCLAW_CONFIG_PATH``** environment variable (explicit override)
2. **``CLAWDBOT_CONFIG_PATH``** environment variable (legacy fallback)
3. **State directory** + ``openclaw.json``
4. **Legacy config filenames** in state directory:
   ``clawdbot.json``, ``moldbot.json``, ``moltbot.json``
5. **Legacy state directories**: ``~/.clawdbot/``, ``~/.moldbot/``,
   ``~/.moltbot/``

.. code-block:: typescript

   // From src/config/paths.ts
   const LEGACY_STATE_DIRNAMES = [".clawdbot", ".moldbot", ".moltbot"];
   const NEW_STATE_DIRNAME = ".openclaw";
   const CONFIG_FILENAME = "openclaw.json";
   const LEGACY_CONFIG_FILENAMES = ["clawdbot.json", "moldbot.json", "moltbot.json"];


State Directory
===============

The state directory stores all mutable data (sessions, logs, caches,
credentials, agent workspaces).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Contents
   * - ``~/.openclaw/``
     - Default state directory
   * - ``~/.openclaw/openclaw.json``
     - Configuration file
   * - ``~/.openclaw/sessions/``
     - Session persistence (Pi session logs)
   * - ``~/.openclaw/credentials/``
     - OAuth tokens and provider credentials
   * - ``~/.openclaw/agents/<agentId>/``
     - Per-agent workspace and session logs

The state directory can be overridden:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Effect
   * - ``OPENCLAW_STATE_DIR``
     - Override the entire state directory
   * - ``CLAWDBOT_STATE_DIR``
     - Legacy fallback for state directory
   * - ``OPENCLAW_HOME``
     - Override the home directory base

.. note::

   If the new ``~/.openclaw/`` directory does not exist but a legacy
   directory does (e.g., ``~/.clawdbot/``), the legacy directory is used
   automatically.  The ``openclaw doctor`` command can migrate legacy
   configs.


Environment Variable Precedence
================================

Environment variables can override or supplement config file values.
The resolution order from highest to lowest priority is:

1. **CLI flags** (e.g., ``--port 9999``)
2. **Environment variables** (e.g., ``OPENCLAW_GATEWAY_PORT=9999``)
3. **Config file values** (e.g., ``gateway.port``)
4. **Runtime defaults** (hardcoded in source)

Key environment variables:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Purpose
   * - ``OPENCLAW_CONFIG_PATH``
     - Override config file path
   * - ``OPENCLAW_STATE_DIR``
     - Override state directory
   * - ``OPENCLAW_GATEWAY_PORT``
     - Override gateway port (default: 18789)
   * - ``OPENCLAW_HOME``
     - Override home directory for path resolution
   * - ``OPENCLAW_OAUTH_DIR``
     - Override OAuth credentials directory
   * - ``OPENCLAW_NIX_MODE``
     - Enable Nix mode (no auto-install, read-only config)
   * - ``OPENCLAW_NO_RESPAWN``
     - Skip CLI process respawn
   * - ``OPENCLAW_SKIP_CHANNELS``
     - Skip channel startup (dev mode)
   * - ``OPENCLAW_PROFILE``
     - Select a CLI profile
   * - ``OPENCLAW_DISABLE_LAZY_SUBCOMMANDS``
     - Eagerly register all CLI commands

Legacy environment variables (``CLAWDBOT_*``) are supported as fallbacks
for backward compatibility.


.. _config-schema:

Config Schema (``OpenClawSchema``)
===================================

The config file is validated against a Zod schema defined in
``src/config/zod-schema.ts``.  The schema is strict (no unknown properties
allowed) and uses ``.optional()`` for all fields.

Top-Level Sections
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 45 33

   * - Section
     - Purpose
     - Type reference
   * - ``meta``
     - Metadata (last touched version/timestamp)
     - ``{ lastTouchedVersion?, lastTouchedAt? }``
   * - ``env``
     - Environment variables and shell env config
     - ``{ shellEnv?, vars?, [key: string] }``
   * - ``wizard``
     - Onboarding wizard state
     - ``{ lastRunAt?, lastRunVersion?, ... }``
   * - ``diagnostics``
     - Diagnostics: flags, OpenTelemetry, cache tracing
     - ``DiagnosticsConfig``
   * - ``logging``
     - Log level, file, console style, redaction
     - ``LoggingConfig``
   * - ``update``
     - Update channel (stable/beta/dev), auto-update policy
     - ``{ channel?, checkOnStart?, auto? }``
   * - ``browser``
     - Headless browser config: profiles, CDP, SSRF policy
     - ``BrowserConfig``
   * - ``ui``
     - UI customization: accent color, assistant name/avatar
     - ``{ seamColor?, assistant? }``
   * - ``auth``
     - Authentication profiles and provider ordering
     - ``AuthConfig``
   * - ``models``
     - Model provider configuration and aliases
     - ``ModelsConfig``
   * - ``nodeHost``
     - Node host config: browser proxy
     - ``{ browserProxy? }``
   * - ``agents``
     - Agent definitions: defaults + agent list
     - ``{ defaults?, list? }``
   * - ``tools``
     - Tool access policies and configuration
     - ``ToolsConfig``
   * - ``bindings``
     - Agent-to-channel routing rules
     - ``AgentBinding[]``
   * - ``broadcast``
     - Cross-agent message broadcast rules
     - ``BroadcastConfig``
   * - ``audio``
     - Audio configuration
     - ``AudioConfig``
   * - ``media``
     - Media handling (preserveFilenames)
     - ``{ preserveFilenames? }``
   * - ``messages``
     - Message formatting, streaming, human delay
     - ``MessagesConfig``
   * - ``commands``
     - CLI command aliases and customization
     - ``CommandsConfig``
   * - ``approvals``
     - Exec approval policies
     - ``ApprovalsConfig``
   * - ``session``
     - Session persistence and compaction
     - ``SessionConfig``
   * - ``cron``
     - Cron scheduler: enable, store, webhook
     - ``{ enabled?, store?, maxConcurrentRuns?, ... }``
   * - ``hooks``
     - Webhook/event hooks: mappings, Gmail, internal
     - ``{ enabled?, path?, token?, mappings?, ... }``
   * - ``web``
     - Web channel config: heartbeat, reconnect policy
     - ``{ enabled?, heartbeatSeconds?, reconnect? }``
   * - ``channels``
     - Per-channel configuration (see below)
     - ``ChannelsConfig``
   * - ``discovery``
     - mDNS + wide-area DNS-SD discovery config
     - ``{ wideArea?, mdns? }``
   * - ``canvasHost``
     - Canvas artifact server
     - ``{ enabled?, root?, port?, liveReload? }``
   * - ``talk``
     - Talk mode: voice ID, ElevenLabs API key
     - ``{ voiceId?, modelId?, apiKey?, ... }``
   * - ``gateway``
     - Gateway server: port, bind, auth, TLS, remote, HTTP endpoints
     - ``GatewayConfig``
   * - ``memory``
     - Memory backend (builtin/qmd), citations
     - ``{ backend?, citations?, qmd? }``
   * - ``skills``
     - Skill loader: allowlist, extra dirs, per-skill config
     - ``{ allowBundled?, load?, install?, entries? }``
   * - ``plugins``
     - Plugin runtime: allow/deny, load paths, per-plugin config
     - ``{ enabled?, allow?, deny?, entries? }``


.. _config-channels:

Channels Section
----------------

The ``channels`` object contains per-channel configuration.  Core channels
have typed schemas; extension channels use dynamic keys.

.. code-block:: json5

   {
     "channels": {
       "defaults": { "groupPolicy": "closed" },
       "whatsapp": { /* WhatsAppConfig */ },
       "telegram": { /* TelegramConfig */ },
       "discord":  { /* DiscordConfig  */ },
       "slack":    { /* SlackConfig    */ },
       "signal":   { /* SignalConfig   */ },
       "imessage": { /* IMessageConfig */ },
       "msteams":  { /* MSTeamsConfig  */ },
       "irc":      { /* IrcConfig      */ },
       "googlechat": { /* GoogleChatConfig */ },
       // Extension channels:
       "matrix":   { "enabled": true, /* ... */ },
       "nostr":    { "enabled": true, /* ... */ }
     }
   }


.. _config-gateway:

Gateway Section
---------------

.. code-block:: json5

   {
     "gateway": {
       "port": 18789,
       "mode": "local",           // "local" | "remote"
       "bind": "loopback",        // "auto" | "lan" | "loopback" | "tailnet" | "custom"
       "customBindHost": "",      // Used when bind = "custom"
       "auth": {
         "mode": "token",         // "none" | "token" | "password" | "trusted-proxy"
         "token": "...",
         // "password": "...",
         // "trustedProxy": { "userHeader": "x-forwarded-user" }
       },
       "controlUi": {
         "enabled": true,
         "basePath": "/",
         "allowedOrigins": []
       },
       "tls": {
         "enabled": false,
         "autoGenerate": true
       },
       "reload": {
         "mode": "hybrid",        // "off" | "restart" | "hot" | "hybrid"
         "debounceMs": 300
       },
       "http": {
         "endpoints": {
           "chatCompletions": { "enabled": false },
           "responses": { "enabled": false }
         }
       }
     }
   }


.. _config-skills:

Skills Section
--------------

.. code-block:: json5

   {
     "skills": {
       "allowBundled": ["github", "weather", "obsidian"],
       "load": {
         "extraDirs": ["~/my-skills"],
         "watch": true
       },
       "install": {
         "preferBrew": false,
         "nodeManager": "pnpm"     // "npm" | "pnpm" | "yarn" | "bun"
       },
       "entries": {
         "github": {
           "enabled": true,
           "apiKey": "ghp_..."
         }
       }
     }
   }


Config File I/O
===============

The config system is implemented in ``src/config/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Responsibility
   * - ``paths.ts``
     - Path resolution: state dir, config path, lock dir, OAuth dir,
       gateway port
   * - ``zod-schema.ts``
     - Root ``OpenClawSchema`` Zod definition (imports sub-schemas)
   * - ``zod-schema.*.ts``
     - Sub-schemas: agents, approvals, core, hooks, installs, providers,
       session, sensitive markers
   * - ``types.ts``
     - Barrel re-export of all ``types.*.ts`` modules
   * - ``types.openclaw.ts``
     - ``OpenClawConfig`` master type and ``ConfigFileSnapshot``
   * - ``types.agents.ts``
     - ``AgentConfig``, ``AgentsConfig``, ``AgentBinding``
   * - ``types.gateway.ts``
     - ``GatewayConfig``, auth, TLS, reload, HTTP endpoint types
   * - ``types.channels.ts``
     - ``ChannelsConfig``, per-channel config types
   * - ``io.ts``
     - File I/O: ``loadConfig()``, ``readConfigFileSnapshot()``,
       ``writeConfigFile()``, ``parseConfigJson5()``
   * - ``validation.ts``
     - ``validateConfigObject()`` -- runs Zod parse + plugin schema merge
   * - ``legacy-migrate.ts``
     - ``migrateLegacyConfig()`` -- handles renames and removals
   * - ``runtime-overrides.ts``
     - Applies env var and runtime defaults after file load
   * - ``sessions/``
     - Session key derivation and session store persistence


Config Hot-Reload
=================

The gateway supports config hot-reload via the ``gateway.reload`` setting:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Mode
     - Behavior
   * - ``off``
     - No automatic reload; requires gateway restart
   * - ``restart``
     - Full gateway restart on config change
   * - ``hot``
     - Apply changes in-place without restarting connections
   * - ``hybrid``
     - Hot-reload what is safe; restart for breaking changes (default)

Config file changes are detected via ``chokidar`` file watching.  The
``debounceMs`` setting (default: 300ms) prevents rapid re-reads during
editor saves.


Sensitive Value Handling
========================

Certain config fields are marked with ``.register(sensitive)`` in the Zod
schema (e.g., ``gateway.auth.token``, ``hooks.token``, skill API keys).
The logging system can redact these values when ``logging.redactSensitive``
is set to ``"tools"``.

.. warning::

   Sensitive values in the config file are stored in plaintext.  Protect the
   config file with appropriate filesystem permissions.  Consider using
   environment variables for production secrets.


Nix Mode
========

When ``OPENCLAW_NIX_MODE=1``:

- No auto-install flows are attempted
- Missing dependencies produce Nix-specific error messages
- Config is treated as externally managed (read-only perspective)

This mode is detected in ``src/config/paths.ts`` via ``resolveIsNixMode()``.
