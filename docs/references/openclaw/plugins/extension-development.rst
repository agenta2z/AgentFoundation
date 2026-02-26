.. _plugins-extension-development:

=====================
Extension Development
=====================

This guide covers creating a new OpenClaw extension plugin.

Project Setup
=============

1. Create a directory under ``extensions/``:

.. code-block:: bash

   mkdir -p extensions/my-plugin/src

2. Initialize ``package.json``:

.. code-block:: json

   {
     "name": "@openclaw/plugin-my-plugin",
     "version": "1.0.0",
     "type": "module",
     "main": "dist/index.js",
     "scripts": {
       "build": "tsdown src/index.ts --outDir dist"
     },
     "dependencies": {},
     "devDependencies": {
       "openclaw": "workspace:*"
     },
     "peerDependencies": {
       "openclaw": "*"
     }
   }

.. note::

   Use ``workspace:*`` in ``devDependencies`` only. Do not use ``workspace:*``
   in ``dependencies`` as it breaks ``npm install`` in production. The runtime
   resolves ``openclaw/plugin-sdk`` via the jiti alias.

3. Create ``openclaw.plugin.json``:

.. code-block:: json

   {
     "name": "my-plugin",
     "version": "1.0.0",
     "main": "dist/index.js"
   }

Tool Registration
=================

Plugins register tools via ``api.registerTool()``:

.. code-block:: typescript

   import { Type } from "@sinclair/typebox";

   api.registerTool({
     name: "my_tool",
     description: "Performs a useful action",
     parameters: Type.Object({
       input: Type.String({ description: "The input to process" }),
       verbose: Type.Optional(Type.Boolean()),
     }),
     async execute(toolCallId, args) {
       const result = await doSomething(args.input);
       return { content: [{ type: "text", text: result }] };
     },
   });

Tool Policy
-----------

Plugin tools can be:

- **Opt-in**: Only available if explicitly enabled in the agent's tool config
- **Default**: Available unless explicitly denied

The tool policy pipeline (``applyToolPolicyPipeline()``) applies these
restrictions in order:

1. Owner-only restrictions
2. Tool profile allowlists
3. Provider-specific allow/deny
4. Subagent tool policies
5. Group chat tool policies
6. Sandbox restrictions
7. Plugin tool opt-in requirements

Channel Plugin Development
==========================

Channel plugins implement the adapter interfaces from the Plugin SDK:

.. code-block:: typescript

   api.registerChannel({
     id: "my-channel",
     messaging: {
       send: async (params) => { /* send message */ },
       onMessage: (handler) => { /* register handler */ },
       capabilities: () => ({
         threading: true,
         reactions: true,
         media: ["image", "audio"],
         formatting: "markdown",
       }),
     },
     config: {
       configFields: () => [
         { key: "apiToken", label: "API Token", type: "password" },
       ],
       validate: (config) => ({ valid: true }),
     },
   });

See :doc:`../channels/plugin-sdk` for the full adapter interface
specifications.

Testing
=======

Plugin tests use Vitest with the extensions test config:

.. code-block:: bash

   pnpm test:extensions

Test files follow the ``*.test.ts`` colocated pattern within the extension's
``src/`` directory.
