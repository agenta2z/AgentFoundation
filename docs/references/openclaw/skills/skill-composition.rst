.. _skill-composition:

=================
Skill Composition
=================

Multiple skills are composed additively into the agent's system prompt. The
composition pipeline filters, orders, and formats skills within token budget
constraints.

Filtering Pipeline
==================

The ``filterSkillEntries()`` function applies these filters in order:

1. **Eligibility filter**: ``shouldIncludeSkill()`` removes skills that fail
   OS, binary, env, or config checks (see :doc:`skill-eligibility`)
2. **Agent skill filter**: If the agent config has a ``skillFilter`` list,
   only skills whose names appear in the list are kept

.. code-block:: typescript

   function filterSkillEntries(
     entries: SkillEntry[],
     config?: OpenClawConfig,
     skillFilter?: string[],
     eligibility?: SkillEligibilityContext,
   ): SkillEntry[]

The ``skillFilter`` is normalized via ``normalizeSkillFilter()`` from
``src/agents/skills/filter.ts``, which handles case-insensitive matching and
deduplication.

Source: ``src/agents/skills/workspace.ts:67-88``

Prompt Formatting
=================

After filtering, eligible skills are formatted for the system prompt:

1. Extract ``Skill`` objects from ``SkillEntry[]``
2. Call ``formatSkillsForPrompt()`` (from ``@mariozechner/pi-coding-agent``)
   which concatenates skill names, file paths, and prompt content
3. Apply ``compactSkillPaths()`` to replace home directory prefixes with ``~``

Path Compaction
---------------

.. code-block:: text

   Before: /Users/alice/.bun/install/global/node_modules/openclaw/skills/github/SKILL.md
   After:  ~/.bun/install/global/node_modules/openclaw/skills/github/SKILL.md

This saves ~5-6 tokens per skill path. Across 50+ skills, that's approximately
400-600 tokens saved. The LLM understands ``~`` expansion, and the ``read``
tool resolves ``~`` to the home directory at runtime.

Source: ``src/agents/skills/workspace.ts:45-53``

Token Budgeting
===============

The system uses a binary search approach to find the largest set of skills
that fits within the character budget:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Parameter
     - Default
     - Description
   * - ``DEFAULT_MAX_SKILLS_IN_PROMPT``
     - 150
     - Max number of skills in prompt
   * - ``DEFAULT_MAX_SKILLS_PROMPT_CHARS``
     - 30,000
     - Max total characters for skills text

If the total formatted prompt exceeds ``maxSkillsPromptChars``, skills are
trimmed from the end until the budget is met. Priority is determined by source
order (higher-precedence sources appear first).

Workspace Synchronization
=========================

For sandboxed agent execution, skills need to be available inside the sandbox.
The ``syncSkillsToWorkspace()`` function copies all resolved skill directories
to a target workspace path:

.. code-block:: text

   Source: ~/.openclaw/skills/weather/SKILL.md
   Target: <sandbox-workspace>/skills/weather/SKILL.md

This ensures sandboxed agents have access to the same skills as the host
agent, without requiring filesystem access outside the sandbox.

Snapshot System
===============

When a session begins, a ``SkillSnapshot`` is created and stored
with the session entry. This captures:

- The formatted prompt text
- List of skill names with their ``primaryEnv`` and ``requiredEnv``
- The ``skillFilter`` applied
- A version number for cache invalidation

Snapshots ensure consistent skill availability across a conversation session,
even if skills are modified on disk mid-conversation.

.. code-block:: typescript

   type SkillSnapshot = {
     prompt: string;
     skills: Array<{
       name: string;
       primaryEnv?: string;
       requiredEnv?: string[];
     }>;
     skillFilter?: string[];
     resolvedSkills?: Skill[];
     version?: number;
   };

Source: ``src/agents/skills/types.ts:82-89``

Composition Example
===================

Given an agent with ``skillFilter: ["weather", "github", "exec"]``:

.. code-block:: text

   1. Load from 6 sources:
      - Bundled: weather, github, exec, spotify, discord, ... (50+)
      - Workspace: (none)

   2. Filter by eligibility:
      - spotify: excluded (requires spotify_player binary)
      - discord: included

   3. Filter by skillFilter:
      - Only weather, github, exec kept

   4. Format for prompt:
      - weather SKILL.md content (~500 chars)
      - github SKILL.md content (~2000 chars)
      - exec SKILL.md content (~800 chars)
      - Total: ~3300 chars (well within 30,000 budget)

   5. Compact paths:
      - /Users/alice/.bun/.../weather/SKILL.md → ~/.bun/.../weather/SKILL.md

   6. Inject into system prompt
