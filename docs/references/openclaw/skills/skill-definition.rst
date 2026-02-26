.. _skill-definition:

================
Skill Definition
================

Each skill is a directory containing a ``SKILL.md`` file. The file uses YAML
frontmatter (delimited by ``---``) followed by the prompt content that gets
injected into the agent's system prompt.

SKILL.md File Format
====================

.. code-block:: markdown

   ---
   name: my-skill
   description: A brief description of what this skill does
   user-invocable: true
   command-dispatch: tool
   command-tool: exec
   command-arg-mode: raw
   metadata:
     openclaw:
       primaryEnv: MY_API_KEY
       emoji: "🔧"
       homepage: https://example.com
       os:
         - darwin
         - linux
       requires:
         bins:
           - my-binary
         anyBins:
           - alt-binary-a
           - alt-binary-b
         env:
           - MY_API_KEY
         config:
           - browser.enabled
       install:
         - kind: brew
           formula: my-tool
           bins:
             - my-binary
         - kind: node
           package: my-npm-package
           bins:
             - my-binary
   ---

   # My Skill

   Instructions for the agent on how to use this skill...


Type Specifications
===================

SkillEntry
----------

The core type representing a loaded skill:

.. code-block:: typescript

   type SkillEntry = {
     skill: Skill;                        // from @mariozechner/pi-coding-agent
     frontmatter: ParsedSkillFrontmatter; // YAML key-value pairs
     metadata?: OpenClawSkillMetadata;    // OpenClaw-specific metadata
     invocation?: SkillInvocationPolicy;  // user-invocable, disable-model-invocation
   };

Source: ``src/agents/skills/types.ts``

OpenClawSkillMetadata
---------------------

.. code-block:: typescript

   type OpenClawSkillMetadata = {
     always?: boolean;       // Always include regardless of eligibility
     skillKey?: string;      // Override key for config lookup
     primaryEnv?: string;    // Primary environment variable for API key
     emoji?: string;         // Display emoji
     homepage?: string;      // Skill homepage URL
     os?: string[];          // Supported OS platforms (darwin, linux, win32)
     requires?: {
       bins?: string[];      // ALL listed binaries must exist
       anyBins?: string[];   // At least ONE listed binary must exist
       env?: string[];       // Environment variables that must be set
       config?: string[];    // Config paths that must be truthy
     };
     install?: SkillInstallSpec[];  // Auto-installation specifications
   };

SkillInstallSpec
----------------

Defines how a skill's dependencies can be auto-installed:

.. code-block:: typescript

   type SkillInstallSpec = {
     id?: string;                    // Installation identifier
     kind: "brew" | "node" | "go" | "uv" | "download";
     label?: string;                 // Display label
     bins?: string[];                // Binaries provided by this install
     os?: string[];                  // OS restriction for this install step
     formula?: string;               // Homebrew formula name (kind: brew)
     package?: string;               // npm/Go/uv package name
     module?: string;                // Go module path (kind: go)
     url?: string;                   // Download URL (kind: download)
     archive?: string;               // Archive type
     extract?: boolean;              // Whether to extract archive
     stripComponents?: number;       // tar --strip-components value
     targetDir?: string;             // Installation target directory
   };

SkillInvocationPolicy
---------------------

Controls how the skill can be invoked:

.. code-block:: typescript

   type SkillInvocationPolicy = {
     userInvocable: boolean;          // Can users invoke via /command?
     disableModelInvocation: boolean; // Exclude from model's system prompt?
   };

- ``user-invocable: true`` — Registers the skill as a slash command
- ``disable-model-invocation: true`` — Prevents the model from seeing the
  skill in its system prompt; only invocable via explicit user commands

SkillCommandDispatchSpec
------------------------

Enables deterministic tool dispatch for slash commands (bypassing the LLM):

.. code-block:: typescript

   type SkillCommandDispatchSpec = {
     kind: "tool";
     toolName: string;      // Name of the tool to invoke
     argMode?: "raw";       // Forward raw args string without parsing
   };

When a skill has ``command-dispatch: tool``, invoking the slash command
directly calls the specified tool instead of going through an LLM turn. This
is used for commands like ``/exec`` or ``/web_search`` where the user's input
maps directly to tool parameters.

SkillSnapshot
-------------

A frozen copy of resolved skills for a session:

.. code-block:: typescript

   type SkillSnapshot = {
     prompt: string;         // Formatted skills prompt text
     skills: Array<{
       name: string;
       primaryEnv?: string;
       requiredEnv?: string[];
     }>;
     skillFilter?: string[];     // Agent-level filter used
     resolvedSkills?: Skill[];   // Full resolved skill objects
     version?: number;           // Snapshot version
   };

Source: ``src/agents/skills/types.ts``

Example: Weather Skill
======================

A minimal real-world skill from the bundled collection:

.. code-block:: text

   skills/weather/
   └── SKILL.md

The ``SKILL.md`` contains instructions that teach the agent how to check
weather using available tools (e.g., web search or a weather API), what
formats to use for responses, and how to handle location ambiguity.
