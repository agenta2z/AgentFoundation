.. _plugins-index:

=============
Plugin System
=============

OpenClaw supports extending its functionality through plugins (extensions).
Plugins can add new channels, tools, and capabilities without modifying core
code.

Plugin Architecture
===================

Plugins live under ``extensions/`` as separate packages, each with its own
``package.json`` and an ``openclaw.plugin.json`` manifest.

.. code-block:: text

   extensions/
   ├── discord/
   │   ├── openclaw.plugin.json    # Plugin manifest
   │   ├── package.json            # Dependencies
   │   ├── src/                    # Source code
   │   └── dist/                   # Built output
   ├── msteams/
   ├── matrix/
   ├── zalo/
   ├── zalouser/
   ├── tlon/
   ├── open-prose/
   └── voice-call/

Plugin Manifest
===============

Each plugin requires an ``openclaw.plugin.json`` manifest:

.. code-block:: json

   {
     "name": "my-plugin",
     "version": "1.0.0",
     "description": "A custom OpenClaw plugin",
     "main": "dist/index.js",
     "type": "channel",
     "channel": {
       "id": "my-channel",
       "displayName": "My Channel"
     }
   }

The manifest specifies the plugin type, entry point, and type-specific
configuration. The gateway loads plugins at startup by scanning the
``extensions/`` directory and importing each plugin's main module.

Plugin Types
============

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - ``channel``
     - Adds a new messaging channel integration
   * - ``tool``
     - Adds new agent tools
   * - ``skill``
     - Provides additional skills (via skill directories)
   * - ``workflow``
     - Adds workflow/prose capabilities (e.g., open-prose)

Plugin SDK
==========

Plugins use the SDK exported from ``openclaw/plugin-sdk``:

.. code-block:: typescript

   import { definePlugin } from "openclaw/plugin-sdk";

   export default definePlugin({
     name: "my-plugin",
     setup(api) {
       // Register tools
       api.registerTool({
         name: "my_tool",
         description: "Does something useful",
         parameters: { /* TypeBox schema */ },
         execute: async (toolCallId, args) => { /* ... */ }
       });

       // Register channel adapter
       api.registerChannel({
         id: "my-channel",
         messaging: myMessagingAdapter,
         config: myConfigAdapter,
         pairing: myPairingAdapter,
       });
     }
   });

Source: ``src/plugin-sdk/index.ts``, ``src/plugins/``

.. toctree::
   :maxdepth: 2

   extension-development
