.. _infrastructure-logging:

=======
Logging
=======

OpenClaw uses a structured logging subsystem based on tslog, with support for
log levels, subsystem tagging, sensitive data redaction, and file output.

Architecture
============

.. code-block:: text

   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │  Subsystem   │ ──▶ │   Logger     │ ──▶ │  Transport   │
   │  Logger      │     │   Core       │     │  (console/   │
   │              │     │              │     │   file)       │
   └──────────────┘     └──────────────┘     └──────────────┘
         │                     │
         ▼                     ▼
   ┌──────────────┐     ┌──────────────┐
   │  Redaction   │     │  Console     │
   │  Pipeline    │     │  Capture     │
   └──────────────┘     └──────────────┘

Key Files
=========

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Purpose
   * - ``src/logging/config.ts``
     - Logger configuration
   * - ``src/logging/levels.ts``
     - Log level definitions
   * - ``src/logging/subsystem.ts``
     - Subsystem-scoped loggers
   * - ``src/logging/redact.ts``
     - Sensitive data redaction
   * - ``src/logging/console.ts``
     - Console output capture and formatting
   * - ``src/logging/env-log-level.ts``
     - Environment-based log level resolution
   * - ``src/logging/state.ts``
     - Logger state management
   * - ``src/logger.ts``
     - Main logger instance and convenience exports

Subsystem Loggers
=================

Components create scoped loggers via ``createSubsystemLogger()``:

.. code-block:: typescript

   import { createSubsystemLogger } from "../logging/subsystem.js";

   const log = createSubsystemLogger("skills");
   log.debug("Loading skills from workspace");
   log.warn("Skill file too large", { path, size });

Common subsystems: ``skills``, ``memory``, ``gateway``, ``agent``,
``telegram``, ``discord``, ``slack``, ``config``, ``auth``, ``browser``.

Log Levels
==========

Levels from most to least verbose:

1. ``silly`` — Extremely verbose debug output
2. ``trace`` — Detailed trace information
3. ``debug`` — Debug information
4. ``info`` — Informational messages
5. ``warn`` — Warning conditions
6. ``error`` — Error conditions
7. ``fatal`` — Fatal errors

Set via environment: ``LOG_LEVEL=debug`` or ``OPENCLAW_LOG_LEVEL=debug``.

Sensitive Data Redaction
========================

The redaction pipeline (``src/logging/redact.ts``) scrubs sensitive values
from log output:

- API keys and tokens
- Phone numbers
- Personal identifiers

Configured via ``logging.redactSensitive`` in ``openclaw.json`` (default:
``"tools"`` — redacts tool call arguments).

Console Capture
===============

``src/logging/console.ts`` provides console output capture for intercepting
``console.log``, ``console.warn``, etc. from third-party libraries. This
routes all console output through the structured logger with proper
formatting and level mapping.

File Size Capping
=================

Log files are automatically capped to prevent disk exhaustion. The capping
mechanism is tested in ``src/logging/log-file-size-cap.test.ts``.

Source: ``src/logging/``
