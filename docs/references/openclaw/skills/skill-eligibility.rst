.. _skill-eligibility:

==================
Skill Eligibility
==================

Not all skills are appropriate for every environment. The eligibility system
gates skills based on the runtime platform, available binaries, environment
variables, and configuration state.

Evaluation Function
===================

The ``shouldIncludeSkill()`` function in ``src/agents/skills/config.ts``
evaluates whether a skill should be included:

.. code-block:: typescript

   function shouldIncludeSkill(params: {
     entry: SkillEntry;
     config?: OpenClawConfig;
     eligibility?: SkillEligibilityContext;
   }): boolean

The evaluation proceeds through these checks in order:

1. **Config disable**: If ``skills.entries.<key>.enabled === false``, exclude
2. **Bundled allowlist**: If ``skills.allowBundled`` is set and the skill is
   bundled, it must appear in the list
3. **Runtime eligibility**: Evaluated by ``evaluateRuntimeEligibility()``

Runtime Eligibility Checks
==========================

These checks run via ``evaluateRuntimeEligibility()`` from
``src/shared/config-eval.ts``:

OS Platform Check
-----------------

If the skill specifies ``metadata.openclaw.os``, the current platform
(``process.platform``) must match one of the listed values:

.. code-block:: yaml

   metadata:
     openclaw:
       os:
         - darwin        # macOS only
         - linux         # Linux only
         - win32         # Windows only

For remote agents (e.g., macOS node connected to a Linux gateway), the
remote platform is checked via ``eligibility.remote.platforms``.

Always-Include Override
-----------------------

If ``metadata.openclaw.always`` is ``true``, the skill is included regardless
of all other runtime checks (but config disable and bundled allowlist still
apply).

Required Binaries (all)
-----------------------

All listed binaries must be found on the system:

.. code-block:: yaml

   metadata:
     openclaw:
       requires:
         bins:
           - ffmpeg
           - ffprobe

Each binary is checked via ``hasBinary()`` which runs ``which <name>`` (or
checks ``eligibility.remote.hasBin`` for remote platforms).

Required Binaries (any)
-----------------------

At least one of the listed binaries must be found:

.. code-block:: yaml

   metadata:
     openclaw:
       requires:
         anyBins:
           - brave-browser
           - google-chrome
           - chromium

Environment Variables
---------------------

All listed environment variables must be set (non-empty):

.. code-block:: yaml

   metadata:
     openclaw:
       requires:
         env:
           - OPENAI_API_KEY

The check considers three sources:

1. ``process.env[envName]`` — system environment
2. ``skillConfig.env[envName]`` — per-skill config env block
3. ``skillConfig.apiKey`` combined with ``metadata.openclaw.primaryEnv`` — if
   the skill has a ``primaryEnv`` and an ``apiKey`` is configured, the env
   requirement is satisfied

This third source enables configuring API keys per-skill in ``openclaw.json``:

.. code-block:: json5

   {
     "skills": {
       "entries": {
         "openai-image-gen": {
           "apiKey": "sk-..."
         }
       }
     }
   }

Config Path Checks
------------------

All listed config paths must resolve to truthy values:

.. code-block:: yaml

   metadata:
     openclaw:
       requires:
         config:
           - browser.enabled

The ``isConfigPathTruthy()`` function walks the config object using dot-
separated paths and checks truthiness. Some paths have default values (e.g.,
``browser.enabled`` defaults to ``true``).

Remote Platform Support
=======================

When an agent runs on a remote node (e.g., macOS companion app connected to a
Linux server), eligibility checks are performed against the remote platform:

.. code-block:: typescript

   type SkillEligibilityContext = {
     remote?: {
       platforms: string[];
       hasBin: (bin: string) => boolean;
       hasAnyBin: (bins: string[]) => boolean;
       note?: string;
     };
   };

This enables macOS-specific skills (like ``apple-notes`` or ``things-mac``)
to be available when a macOS node is connected, even if the gateway runs on
Linux.

Source: ``src/agents/skills/config.ts:70-102``
