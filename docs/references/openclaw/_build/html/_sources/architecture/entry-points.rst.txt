=======================================
Entry Points & Bootstrap Chain
=======================================

This page documents how the ``openclaw`` CLI boots, how the Commander.js
program is built, and how the Gateway server starts.

.. contents:: On this page
   :local:
   :depth: 2


Bootstrap Overview
==================

The following diagram shows the full bootstrap chain from the shell command
``openclaw`` to a running CLI program or Gateway server.

.. code-block:: text

   Shell: $ openclaw <command> [options]
          |
          v
   openclaw.mjs                     (1) Node.js entry shim
          |
          +-- module.enableCompileCache()
          +-- installProcessWarningFilter()
          +-- import("./dist/entry.js")
          |
          v
   src/entry.ts                     (2) Process bootstrap
          |
          +-- process.title = "openclaw"
          +-- installProcessWarningFilter()
          +-- normalizeEnv()
          +-- normalizeWindowsArgv()
          +-- ensureExperimentalWarningSuppressed()
          |     (may respawn with --disable-warning=ExperimentalWarning)
          +-- parseCliProfileArgs()
          +-- applyCliProfileEnv()
          +-- import("./cli/run-main.js").runCli()
          |
          v
   src/cli/run-main.ts              (3) CLI initialization
          |
          +-- loadDotEnv()
          +-- normalizeEnv()
          +-- ensureOpenClawCliOnPath()
          +-- assertSupportedRuntime()
          +-- tryRouteCli()           (fast-path for special routes)
          +-- enableConsoleCapture()
          +-- buildProgram()
          +-- installUnhandledRejectionHandler()
          +-- registerCoreCliByName() (lazy primary command)
          +-- registerSubCliByName()  (lazy sub-CLI)
          +-- registerPluginCliCommands() (if needed)
          +-- program.parseAsync()
          |
          v
   Commander.js dispatches to the matched command handler


Step 1: ``openclaw.mjs`` -- Node.js Entry Shim
===============================================

The file ``openclaw.mjs`` is the npm ``bin`` entry point declared in
``package.json``:

.. code-block:: json

   {
     "bin": {
       "openclaw": "openclaw.mjs"
     }
   }

Its responsibilities are minimal:

1. **Enable compile cache** -- Calls ``module.enableCompileCache()`` (Node 22+)
   to speed up subsequent starts.

2. **Install warning filter** -- Suppresses noisy ``ExperimentalWarning``
   messages from the Node.js runtime.

3. **Load the build output** -- Tries ``./dist/entry.js``, then
   ``./dist/entry.mjs``.  If neither exists, it throws an error indicating
   the project has not been built.

.. code-block:: typescript

   // openclaw.mjs (simplified)
   import module from "node:module";

   if (module.enableCompileCache) {
     module.enableCompileCache();
   }

   await installProcessWarningFilter();

   if (await tryImport("./dist/entry.js")) {
     // OK
   } else if (await tryImport("./dist/entry.mjs")) {
     // OK
   } else {
     throw new Error("openclaw: missing dist/entry.(m)js");
   }


Step 2: ``src/entry.ts`` -- Process Bootstrap
==============================================

``entry.ts`` is the compiled TypeScript entry point.  It runs only when
executed as the main module (guarded by ``isMainModule()``).

Key operations:

1. **Set process title** to ``"openclaw"`` (visible in ``ps``).

2. **Suppress experimental warnings** -- If the
   ``--disable-warning=ExperimentalWarning`` flag is not present and
   respawning is allowed, it re-invokes ``process.execPath`` with the flag
   prepended.  The ``OPENCLAW_NODE_OPTIONS_READY`` env var prevents infinite
   recursion.

3. **Parse CLI profile** -- The ``--profile <name>`` flag selects a config
   profile, which applies environment variable overrides before any other
   code runs.

4. **Delegate to ``runCli``** -- Dynamically imports ``./cli/run-main.js``
   and calls ``runCli(process.argv)``.

.. warning::

   The ``isMainModule`` guard prevents ``entry.ts`` from running its
   side-effects when imported as a dependency (e.g., when ``dist/index.js``
   is the actual entry point for library consumers).

Environment variables that influence entry behavior:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Effect
   * - ``OPENCLAW_NO_RESPAWN``
     - Skip the child-process respawn for warning suppression
   * - ``OPENCLAW_NODE_OPTIONS_READY``
     - Guard against respawn recursion (set automatically)
   * - ``NO_COLOR`` / ``FORCE_COLOR``
     - Disable/force ANSI color output


Step 3: ``src/cli/run-main.ts`` -- CLI Initialization
=====================================================

``runCli()`` is the main CLI startup function.  It performs:

1. **Environment setup** -- Loads ``.env`` files, normalizes environment
   variables, ensures the ``openclaw`` binary is on ``$PATH``.

2. **Runtime guard** -- Asserts the Node.js version meets the minimum
   requirement (>=22.12.0).

3. **Fast-path routing** -- ``tryRouteCli()`` handles special argument
   patterns before the full Commander.js program is built.

4. **Console capture** -- Redirects ``console.log/warn/error`` into the
   structured logging system.

5. **Build program** -- Calls ``buildProgram()`` which constructs the
   Commander.js ``Command`` tree.

6. **Error handlers** -- Installs global handlers for uncaught exceptions
   and unhandled promise rejections.

7. **Command registration** -- Lazily registers only the primary command
   (identified from ``argv``) to keep startup fast.  If the primary command
   is not a built-in, plugin CLI commands are registered too.

8. **Parse** -- ``program.parseAsync(parseArgv)`` delegates to Commander.js.

.. code-block:: typescript

   // src/cli/run-main.ts (simplified)
   export async function runCli(argv: string[]) {
     loadDotEnv({ quiet: true });
     normalizeEnv();
     ensureOpenClawCliOnPath();
     assertSupportedRuntime();

     if (await tryRouteCli(normalizedArgv)) return;

     enableConsoleCapture();
     const program = buildProgram();
     installUnhandledRejectionHandler();

     await program.parseAsync(parseArgv);
   }


Program Construction
====================

``buildProgram()`` in ``src/cli/program/build-program.ts``:

.. code-block:: typescript

   export function buildProgram() {
     const program = new Command();
     const ctx = createProgramContext();

     setProgramContext(program, ctx);
     configureProgramHelp(program, ctx);
     registerPreActionHooks(program, ctx.programVersion);
     registerProgramCommands(program, ctx, argv);

     return program;
   }

``registerProgramCommands`` calls two registrars:

- ``registerCoreCliCommands()`` -- Core commands (see table below)
- ``registerSubCliCommands()`` -- Sub-CLI commands (see table below)

Both use **lazy registration**: a lightweight placeholder command is created
with ``allowUnknownOption(true)`` and ``allowExcessArguments(true)``.  When
the placeholder's action fires, it dynamically imports the real registrar,
replaces itself, and re-parses the arguments.


CLI Command Registry
====================

Core Commands
-------------

These commands are registered by ``src/cli/program/command-registry.ts``:

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Command
     - Description
     - Subcommands?
   * - ``setup``
     - Initialize local config and agent workspace
     - No
   * - ``onboard``
     - Interactive onboarding wizard for gateway, workspace, and skills
     - No
   * - ``configure``
     - Interactive setup wizard for credentials, channels, gateway, and
       agent defaults
     - No
   * - ``config``
     - Non-interactive config helpers (get/set/unset)
     - Yes
   * - ``doctor``
     - Health checks + quick fixes for the gateway and channels
     - No
   * - ``dashboard``
     - Open the Control UI with your current token
     - No
   * - ``reset``
     - Reset local config/state (keeps the CLI installed)
     - No
   * - ``uninstall``
     - Uninstall the gateway service + local data (CLI remains)
     - No
   * - ``message``
     - Send, read, and manage messages
     - Yes
   * - ``memory``
     - Search and reindex memory files
     - Yes
   * - ``agent``
     - Run one agent turn via the Gateway
     - No
   * - ``agents``
     - Manage isolated agents (workspaces, auth, routing)
     - Yes
   * - ``status``
     - Show channel health and recent session recipients
     - No
   * - ``health``
     - Fetch health from the running gateway
     - No
   * - ``sessions``
     - List stored conversation sessions
     - No
   * - ``browser``
     - Manage OpenClaw's dedicated browser (Chrome/Chromium)
     - Yes


Sub-CLI Commands
----------------

These commands are registered by ``src/cli/program/register.subclis.ts``:

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Command
     - Description
     - Subcommands?
   * - ``acp``
     - Agent Control Protocol tools
     - Yes
   * - ``gateway``
     - Run, inspect, and query the WebSocket Gateway
     - Yes
   * - ``daemon``
     - Gateway service (legacy alias)
     - Yes
   * - ``logs``
     - Tail gateway file logs via RPC
     - No
   * - ``system``
     - System events, heartbeat, and presence
     - Yes
   * - ``models``
     - Discover, scan, and configure models
     - Yes
   * - ``approvals``
     - Manage exec approvals (gateway or node host)
     - Yes
   * - ``nodes``
     - Manage gateway-owned node pairing and node commands
     - Yes
   * - ``devices``
     - Device pairing + token management
     - Yes
   * - ``node``
     - Run and manage the headless node host service
     - Yes
   * - ``sandbox``
     - Manage sandbox containers for agent isolation
     - Yes
   * - ``tui``
     - Open a terminal UI connected to the Gateway
     - No
   * - ``cron``
     - Manage cron jobs via the Gateway scheduler
     - Yes
   * - ``dns``
     - DNS helpers for wide-area discovery (Tailscale + CoreDNS)
     - Yes
   * - ``docs``
     - Search the live OpenClaw docs
     - No
   * - ``hooks``
     - Manage internal agent hooks
     - Yes
   * - ``webhooks``
     - Webhook helpers and integrations
     - Yes
   * - ``qr``
     - Generate iOS pairing QR/setup code
     - No
   * - ``clawbot``
     - Legacy clawbot command aliases
     - Yes
   * - ``pairing``
     - Secure DM pairing (approve inbound requests)
     - Yes
   * - ``plugins``
     - Manage OpenClaw plugins and extensions
     - Yes
   * - ``channels``
     - Manage connected chat channels (Telegram, Discord, etc.)
     - Yes
   * - ``directory``
     - Lookup contact and group IDs for supported chat channels
     - Yes
   * - ``security``
     - Security tools and local config audits
     - Yes
   * - ``skills``
     - List and inspect available skills
     - Yes
   * - ``update``
     - Update OpenClaw and inspect update channel status
     - Yes
   * - ``completion``
     - Generate shell completion script
     - No


Gateway Server Startup
======================

The gateway starts when the user runs ``openclaw gateway run``.  The startup
sequence is:

.. code-block:: text

   openclaw gateway run [--port N] [--bind MODE]
          |
          v
   src/cli/gateway-cli.ts
          |
          +-- loadConfig()
          +-- resolveGatewayPort(config)
          +-- resolveBindAddress(config)
          |
          v
   Gateway server lifecycle
          |
          +-- Create Express app
          +-- Create HTTP(S) server (with optional TLS)
          +-- Create WebSocket server (ws)
          +-- Register RPC method handlers
          +-- Register HTTP routes (Control UI, /v1/*, hooks, plugins)
          +-- Start channel adapters (Telegram, Discord, Slack, ...)
          +-- Start cron scheduler (if enabled)
          +-- Start mDNS broadcast (if enabled)
          +-- Start Tailscale serve/funnel (if configured)
          +-- Begin accepting connections
          +-- Acquire lock file (prevents duplicate instances)

The gateway listens on a single multiplexed port (default ``18789``) that
serves both WebSocket and HTTP traffic.  The bind address is controlled by
the ``gateway.bind`` config option:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Mode
     - Bind Address
     - Use Case
   * - ``loopback``
     - ``127.0.0.1``
     - Local-only access (default)
   * - ``lan``
     - ``0.0.0.0``
     - Access from any network interface
   * - ``auto``
     - ``127.0.0.1`` or ``0.0.0.0``
     - Loopback if available, else all interfaces
   * - ``tailnet``
     - Tailnet IPv4 (``100.64.0.0/10``)
     - Tailscale network access
   * - ``custom``
     - User-specified IP
     - Custom bind address (``gateway.customBindHost``)

.. note::

   The gateway lock file is stored at ``$TMPDIR/openclaw-<uid>/`` to prevent
   multiple gateway instances from running on the same port.


Library Entry Point (``src/index.ts``)
======================================

``src/index.ts`` (compiled to ``dist/index.js``) serves as the library entry
point for programmatic use.  It:

1. Loads ``.env`` files and normalizes the environment.
2. Ensures the ``openclaw`` CLI is on ``$PATH``.
3. Enables console capture for structured logging.
4. Asserts the runtime version.
5. Builds the Commander.js program.
6. Exports public API functions for library consumers.

The ``exports`` field in ``package.json`` exposes two entry points:

.. code-block:: json

   {
     "exports": {
       ".": "./dist/index.js",
       "./plugin-sdk": {
         "types": "./dist/plugin-sdk/index.d.ts",
         "default": "./dist/plugin-sdk/index.js"
       }
     }
   }

The ``./plugin-sdk`` export provides the public plugin SDK for extension
authors (see :doc:`data-models`).
