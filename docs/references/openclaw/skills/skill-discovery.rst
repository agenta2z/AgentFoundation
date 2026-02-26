.. _skill-discovery:

================
Skill Discovery
================

Skills are loaded from multiple directory sources with ascending precedence.
This layered approach allows bundled skills to be overridden by managed,
personal, project, or workspace skills.

Six-Layer Precedence
====================

Skills are loaded from these directories, listed from lowest to highest
precedence:

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Priority
     - Source
     - Location
   * - 1 (lowest)
     - Extra dirs
     - From ``skills.load.extraDirs`` config + plugin skill directories
   * - 2
     - Bundled
     - Shipped with the ``openclaw`` npm package (resolved via ``resolveBundledSkillsDir()``)
   * - 3
     - Managed
     - ``~/.openclaw/skills/`` (installed from ClawHub)
   * - 4
     - Personal agents
     - ``~/.agents/skills/``
   * - 5
     - Project agents
     - ``<workspace>/.agents/skills/``
   * - 6 (highest)
     - Workspace
     - ``<workspace>/skills/``

**Override semantics**: When multiple sources provide a skill with the same
name, the higher-precedence source wins. This lets users customize bundled
skills by placing a modified version in their workspace.

Source: ``src/agents/skills/workspace.ts``

Discovery Limits
================

To prevent pathological scans and excessive prompt sizes, discovery is bounded
by several limits:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Constant
     - Default
     - Description
   * - ``DEFAULT_MAX_CANDIDATES_PER_ROOT``
     - 300
     - Max directories scanned per source root
   * - ``DEFAULT_MAX_SKILLS_LOADED_PER_SOURCE``
     - 200
     - Max skills loaded from a single source
   * - ``DEFAULT_MAX_SKILLS_IN_PROMPT``
     - 150
     - Max skills included in the system prompt
   * - ``DEFAULT_MAX_SKILLS_PROMPT_CHARS``
     - 30,000
     - Max total characters for all skill prompts
   * - ``DEFAULT_MAX_SKILL_FILE_BYTES``
     - 256,000
     - Max file size for a single SKILL.md (256 KB)

Source: ``src/agents/skills/workspace.ts:95-99``

Loading Process
===============

The loading process for each source directory:

1. **Scan**: List subdirectories up to ``maxCandidatesPerRoot``
2. **Load**: For each subdirectory, look for ``SKILL.md`` (case-insensitive)
3. **Parse**: Read the file, parse YAML frontmatter via ``parseFrontmatter()``
4. **Resolve metadata**: Extract ``OpenClawSkillMetadata`` from the frontmatter
   ``metadata.openclaw`` field
5. **Resolve invocation policy**: Extract ``user-invocable`` and
   ``disable-model-invocation`` flags
6. **Create SkillEntry**: Combine the ``Skill`` object (from
   ``@mariozechner/pi-coding-agent``), parsed frontmatter, metadata, and
   invocation policy

.. code-block:: text

   Source Directory
       │
       ├── skill-a/
       │   └── SKILL.md    → parse → SkillEntry { skill, frontmatter, metadata, invocation }
       ├── skill-b/
       │   └── SKILL.md    → parse → SkillEntry { ... }
       └── ...

Symlinks are explicitly skipped. Only ``.md`` files named ``SKILL.md`` are
recognized.

Skill Key Resolution
====================

Each skill has a **skill key** used for config lookup (``skills.entries.<key>``):

1. If ``metadata.openclaw.skillKey`` is set, use that
2. Otherwise, use the skill's directory name (lowercased)

This allows config-based enable/disable per skill:

.. code-block:: json5

   {
     "skills": {
       "entries": {
         "weather": { "enabled": false },
         "spotify-player": { "apiKey": "sk-..." }
       }
     }
   }
