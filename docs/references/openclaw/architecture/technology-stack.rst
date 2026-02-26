================================
Technology Stack & Dependencies
================================

This page documents the languages, frameworks, and libraries that make up the
OpenClaw technology stack, based on the declarations in ``package.json``.

.. contents:: On this page
   :local:
   :depth: 2


Languages & Runtimes
====================

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Layer
     - Language / Runtime
     - Notes
   * - Core gateway + CLI
     - TypeScript (ESM)
     - Strict typing, no ``any`` policy.  Compiled via ``tsdown``.
   * - Control UI
     - TypeScript + Lit
     - Web components, bundled with Vite.
   * - macOS app
     - Swift (SwiftUI)
     - Menubar app with gateway embedding.
   * - iOS app
     - Swift (SwiftUI)
     - ``Observation`` framework preferred.
   * - Android app
     - Kotlin
     - Standard Android project (Gradle).
   * - Build scripts
     - TypeScript + Bash
     - ``tsx`` for TS script execution, Bash for CI/packaging.
   * - Runtime
     - Node.js >= 22.12.0
     - ESM modules.  Bun also supported for dev/test.


Server & Networking
===================

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Library
     - Version
     - Purpose
   * - ``express``
     - ^5.2.1
     - HTTP server for Gateway REST endpoints and Control UI serving
   * - ``ws``
     - ^8.19.0
     - WebSocket server for the Gateway RPC protocol
   * - ``undici``
     - ^7.22.0
     - HTTP client for outbound API calls (LLM providers, webhooks)
   * - ``https-proxy-agent``
     - ^7.0.6
     - HTTPS proxy support for outbound connections
   * - ``ipaddr.js``
     - ^2.3.0
     - IP address parsing and validation (SSRF guard, bind modes)
   * - ``@homebridge/ciao``
     - ^1.3.5
     - mDNS / Bonjour service discovery


CLI Framework
=============

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Library
     - Version
     - Purpose
   * - ``commander``
     - ^14.0.3
     - CLI program builder, command parsing, help generation
   * - ``chalk``
     - ^5.6.2
     - Terminal color output
   * - ``@clack/prompts``
     - ^1.0.1
     - Interactive CLI prompts (select, confirm, text input)
   * - ``osc-progress``
     - ^0.3.0
     - Terminal progress bars (OSC-based)
   * - ``cli-highlight``
     - ^2.1.11
     - Syntax highlighting for CLI output
   * - ``qrcode-terminal``
     - ^0.12.0
     - QR code rendering in terminal (WhatsApp/iOS pairing)


AI & LLM Integration
=====================

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Library
     - Version
     - Purpose
   * - ``@mariozechner/pi-agent-core``
     - 0.54.1
     - Pi agent orchestration core (agent runtime)
   * - ``@mariozechner/pi-ai``
     - 0.54.1
     - Pi AI abstractions (LLM provider interface)
   * - ``@mariozechner/pi-coding-agent``
     - 0.54.1
     - Coding agent capabilities
   * - ``@mariozechner/pi-tui``
     - 0.54.1
     - Pi terminal UI integration
   * - ``@aws-sdk/client-bedrock``
     - ^3.995.0
     - AWS Bedrock LLM provider
   * - ``@agentclientprotocol/sdk``
     - 0.14.1
     - Agent Client Protocol (ACP) implementation


Chat Channel SDKs
=================

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Library
     - Version
     - Channel
   * - ``grammy``
     - ^1.40.0
     - Telegram Bot API framework
   * - ``@grammyjs/runner``
     - ^2.0.3
     - Telegram long-polling runner
   * - ``@grammyjs/transformer-throttler``
     - ^1.2.1
     - Telegram rate-limit throttler
   * - ``@buape/carbon``
     - 0.0.0-beta-*
     - Discord framework
   * - ``@discordjs/voice``
     - ^0.19.0
     - Discord voice connections
   * - ``@slack/bolt``
     - ^4.6.0
     - Slack Bot framework (Socket Mode + HTTP)
   * - ``@slack/web-api``
     - ^7.14.1
     - Slack Web API client
   * - ``@whiskeysockets/baileys``
     - 7.0.0-rc.9
     - WhatsApp Web multi-device client
   * - ``@line/bot-sdk``
     - ^10.6.0
     - LINE Messaging API SDK
   * - ``@larksuiteoapi/node-sdk``
     - ^1.59.0
     - Feishu / Lark API SDK


Schema Validation & Serialization
==================================

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Library
     - Version
     - Purpose
   * - ``zod``
     - ^4.3.6
     - Config schema validation (``OpenClawSchema``)
   * - ``@sinclair/typebox``
     - 0.34.48
     - Gateway protocol schema definitions (JSON Schema compatible)
   * - ``ajv``
     - ^8.18.0
     - JSON Schema validation for protocol frames at runtime
   * - ``json5``
     - ^2.2.3
     - JSON5 parsing for config files (comments, trailing commas)
   * - ``yaml``
     - ^2.8.2
     - YAML parsing for skill and hook definitions


Data Storage
============

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Library
     - Version
     - Purpose
   * - ``sqlite-vec``
     - 0.1.7-alpha.2
     - SQLite-based vector store for built-in memory/RAG
   * - ``jszip``
     - ^3.10.1
     - ZIP archive handling (session export, media bundles)
   * - ``tar``
     - 7.5.9
     - Tar archive handling (plugin installs, updates)


Media Processing
================

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Library
     - Version
     - Purpose
   * - ``sharp``
     - ^0.34.5
     - Image processing (resize, format conversion)
   * - ``pdfjs-dist``
     - ^5.4.624
     - PDF text extraction and rendering
   * - ``file-type``
     - ^21.3.0
     - File MIME type detection from buffer
   * - ``@mozilla/readability``
     - ^0.6.0
     - HTML article content extraction (link understanding)
   * - ``linkedom``
     - ^0.18.12
     - Server-side DOM for HTML processing
   * - ``playwright-core``
     - 1.58.2
     - Headless browser automation (browsing tool, screenshots)


Text-to-Speech
==============

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Library
     - Version
     - Purpose
   * - ``node-edge-tts``
     - ^1.2.10
     - Microsoft Edge TTS engine
   * - ``opusscript``
     - ^0.0.8
     - Opus audio codec for Discord voice
   * - ``@discordjs/opus``
     - ^0.10.0
     - Native Opus bindings (optional)


Infrastructure & Utilities
===========================

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Library
     - Version
     - Purpose
   * - ``dotenv``
     - ^17.3.1
     - ``.env`` file loading
   * - ``chokidar``
     - ^5.0.0
     - File system watching (config hot-reload, skill watching)
   * - ``croner``
     - ^10.0.1
     - Cron expression parser and scheduler
   * - ``tslog``
     - ^4.10.2
     - Structured logging framework
   * - ``markdown-it``
     - ^14.1.1
     - Markdown parsing and rendering
   * - ``long``
     - ^5.3.2
     - 64-bit integer support (protobuf, Baileys)
   * - ``jiti``
     - ^2.6.1
     - Runtime TypeScript/ESM module loader (plugin resolution)
   * - ``@lydell/node-pty``
     - 1.2.0-beta.3
     - Pseudo-terminal (sandbox, TUI)


Development Dependencies
=========================

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Library
     - Version
     - Purpose
   * - ``typescript``
     - ^5.9.3
     - TypeScript compiler
   * - ``@typescript/native-preview``
     - 7.0.0-dev.*
     - Native TS type checker (``tsgo``)
   * - ``tsdown``
     - ^0.20.3
     - Build / bundle tool
   * - ``tsx``
     - ^4.21.0
     - TypeScript script execution (dev, scripts)
   * - ``vitest``
     - ^4.0.18
     - Test runner (unit, e2e, live, gateway tests)
   * - ``@vitest/coverage-v8``
     - ^4.0.18
     - V8-based code coverage
   * - ``oxlint``
     - ^1.49.0
     - Linter (Rust-based, type-aware)
   * - ``oxfmt``
     - 0.34.0
     - Code formatter (Rust-based)
   * - ``lit``
     - ^3.3.2
     - Web component library (Control UI)
   * - ``signal-utils``
     - 0.21.1
     - Signal-based state management (Control UI)


Peer / Optional Dependencies
=============================

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Library
     - Version
     - Purpose
   * - ``@napi-rs/canvas``
     - ^0.1.89
     - Native canvas for image generation (peer, optional)
   * - ``node-llama-cpp``
     - 3.15.1
     - Local LLM inference via llama.cpp (peer, optional)


Package Manager
===============

OpenClaw uses **pnpm 10.23.0** as its package manager.  The monorepo is
configured via ``pnpm-workspace.yaml`` with workspace packages under
``extensions/``, ``packages/``, and ``ui/``.

.. note::

   Dependencies listed in ``pnpm.patchedDependencies`` must use exact
   versions (no ``^`` or ``~`` ranges) to ensure patches apply correctly.

.. note::

   Bun is supported as an alternative runtime for development and testing
   (``bun <file.ts>``, ``bunx <tool>``).  Node.js remains the primary
   production runtime.
