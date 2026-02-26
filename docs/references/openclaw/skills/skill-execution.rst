.. _skill-execution:

===============
Skill Execution
===============

Skills are executed through two mechanisms: implicit model invocation (the LLM
uses skills included in its system prompt) and explicit user commands (slash
commands).

Model Invocation
================

The primary execution path. When skills pass eligibility checks and are not
marked ``disable-model-invocation: true``, they are formatted into the agent's
system prompt:

1. ``filterSkillEntries()`` applies all filtering rules
2. ``formatSkillsForPrompt()`` (from ``@mariozechner/pi-coding-agent``)
   assembles the skills text
3. ``compactSkillPaths()`` replaces home directory prefixes with ``~`` to save
   tokens (~400-600 tokens saved across all skills)
4. The formatted text is injected into the system prompt

The LLM then naturally uses the skill's instructions during conversation. For
example, if a user asks "What's the weather?", the agent sees the weather
skill's instructions in its prompt and follows them to call the appropriate
tools.

Slash Commands
==============

Skills with ``user-invocable: true`` register as slash commands that users can
invoke directly:

.. code-block:: text

   User: /spotify play Bohemian Rhapsody
   User: /weather London
   User: /github list issues

Resolution is handled by ``resolveSkillCommandInvocation()`` in
``src/auto-reply/skill-commands.ts``:

1. Extract the command name from the message (e.g., ``/spotify`` → ``spotify``)
2. Look up the skill by sanitized command name (lowercase, alphanumeric + underscore)
3. If found and user-invocable, invoke the skill

Command names are sanitized via ``sanitizeSkillCommandName()``:

- Lowercased
- Non-alphanumeric characters replaced with ``_``
- Max length: 32 characters
- Fallback name: ``skill`` (if normalization produces empty string)

Unique names are guaranteed by ``resolveUniqueSkillCommandName()`` which
appends ``_2``, ``_3``, etc. for conflicts.

Deterministic Dispatch
======================

Some skills bypass the LLM entirely by using ``command-dispatch: tool``:

.. code-block:: yaml

   user-invocable: true
   command-dispatch: tool
   command-tool: exec
   command-arg-mode: raw

When a user types ``/exec ls -la``:

1. The system recognizes ``/exec`` as a skill command with dispatch
2. Instead of sending the message through an LLM turn, it directly calls the
   ``exec`` tool with ``ls -la`` as arguments
3. The tool result is returned without LLM processing

This is useful for commands where the user's intent maps directly to a tool
call (e.g., ``/exec``, ``/web_search``, ``/web_fetch``).

``SkillCommandSpec`` captures the dispatch configuration:

.. code-block:: typescript

   type SkillCommandSpec = {
     name: string;                // Command name (e.g., "exec")
     skillName: string;           // Skill name (e.g., "coding-agent")
     description: string;         // Command description (max 100 chars)
     dispatch?: {
       kind: "tool";
       toolName: string;          // Tool to invoke
       argMode?: "raw";           // Forward raw user args
     };
   };

Source: ``src/agents/skills/types.ts:41-57``

Skill Command Listing
=====================

The ``/skill`` meta-command (or ``/skills``) lists all available skill
commands for the current agent. Each entry shows the command name,
description, and whether it uses deterministic dispatch.

Channel-specific formatting adapts the command list:

- **Discord**: Commands are registered as Discord slash commands (description
  max 100 characters)
- **Other channels**: Commands appear as ``/command-name`` in help text
