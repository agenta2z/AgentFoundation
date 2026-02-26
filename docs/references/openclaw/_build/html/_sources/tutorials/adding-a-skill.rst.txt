.. _tutorial-adding-a-skill:

===============
Adding a Skill
===============

This tutorial walks through creating a custom skill from scratch, testing it,
and deploying it for your agents to use.

What We're Building
===================

A skill called ``summarize-url`` that teaches the agent how to fetch a URL
and produce a structured summary. When a user says "summarize this article:
https://example.com/post", the agent will know exactly how to handle it.

Step 1: Choose a Location
===========================

Skills can be placed at any of the 6 discovery layers. For a personal skill,
use your workspace's skills directory:

.. code-block:: bash

   mkdir -p <workspace>/skills/summarize-url/

Alternatively, for a skill shared across all agents:

.. code-block:: bash

   mkdir -p ~/.agents/skills/summarize-url/

Step 2: Create SKILL.md
=========================

Create ``SKILL.md`` in the skill directory. This is the only required file.

.. code-block:: bash

   touch <workspace>/skills/summarize-url/SKILL.md

The file has two parts: YAML frontmatter and the prompt body.

Step 3: Write the Frontmatter
==============================

The YAML frontmatter (between ``---`` delimiters) controls metadata,
eligibility, and invocation behavior:

.. code-block:: markdown

   ---
   name: summarize-url
   description: Fetch a URL and produce a structured summary with key points
   user-invocable: /summarize
   ---

**Key frontmatter fields:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``name``
     - Unique skill identifier (used for override resolution)
   * - ``description``
     - Brief description shown in skill listings
   * - ``user-invocable``
     - Slash command trigger (e.g., ``/summarize``)
   * - ``os``
     - Platform restriction (``darwin``, ``linux``, ``win32``)
   * - ``all-bins``
     - Required binaries (all must exist)
   * - ``any-bins``
     - Required binaries (at least one must exist)
   * - ``env``
     - Required environment variables
   * - ``always``
     - If ``true``, always include in system prompt
   * - ``disable-model-invocation``
     - If ``true``, only invocable via slash command

Step 4: Write the Prompt Body
==============================

Below the frontmatter, write the instructions the agent receives when this
skill is active:

.. code-block:: markdown

   ---
   name: summarize-url
   description: Fetch a URL and produce a structured summary with key points
   user-invocable: /summarize
   ---

   # URL Summarizer

   When asked to summarize a URL or when the `/summarize` command is used:

   1. Use the `web_fetch` tool to retrieve the page content
   2. Extract the main article text, ignoring navigation, ads, and boilerplate
   3. Produce a structured summary with:
      - **Title**: The article's title
      - **Source**: The domain name
      - **Date**: Publication date if available
      - **Summary**: 2-3 sentence overview
      - **Key Points**: 3-5 bullet points capturing the main ideas
      - **Notable Quotes**: Any standout quotes (if present)

   ## Format

   Use this exact output format:

   ```
   📰 **{Title}**
   🔗 {Source} | 📅 {Date}

   **Summary:** {overview}

   **Key Points:**
   - {point 1}
   - {point 2}
   - {point 3}

   **Notable Quotes:**
   > "{quote}" — {attribution}
   ```

   ## Edge Cases

   - If the URL is behind a paywall, report that the content is not accessible
   - If the page has no article content (e.g., a homepage), describe what the
     page contains instead
   - For very long articles, focus on the most important points

The prompt body is injected into the agent's system prompt, so write it as
direct instructions to the agent.

Step 5: Add Eligibility Gates (Optional)
==========================================

For skills that require specific tools or environments, add gating frontmatter:

.. code-block:: yaml

   ---
   name: summarize-url
   description: Fetch a URL and produce a structured summary
   user-invocable: /summarize
   all-bins: [curl]
   ---

The ``shouldIncludeSkill()`` function evaluates gates in order:

1. **OS check**: Does the current platform match ``os``?
2. **Binary check**: Are required binaries available?
3. **Environment check**: Are required env vars set?
4. **Config check**: Do required config paths exist?

If any gate fails, the skill is excluded from the agent's system prompt.

Source: ``src/agents/skills/config.ts``

Step 6: Test Your Skill
=========================

Start a conversation with your agent and verify the skill is loaded:

.. code-block:: text

   You: /summarize https://example.com/article

The agent should:

1. Recognize the ``/summarize`` slash command
2. Match it to your ``summarize-url`` skill
3. Follow the instructions in the prompt body

You can also test model invocation by asking naturally:

.. code-block:: text

   You: Can you summarize this article for me? https://example.com/article

If the skill is in the system prompt, the agent will recognize the pattern
and apply the skill's instructions.

Step 7: Add Advanced Features
==============================

Deterministic Dispatch
----------------------

For skills that should directly invoke a tool without LLM reasoning:

.. code-block:: yaml

   ---
   name: quick-search
   description: Quick web search
   user-invocable: /search
   command-dispatch: web_search
   ---

When the user types ``/search Tokyo weather``, the system bypasses the LLM
and directly calls the ``web_search`` tool with the arguments.

Source: ``src/auto-reply/skill-commands.ts``

Install Specifications
-----------------------

If your skill requires software that might not be installed:

.. code-block:: yaml

   ---
   name: python-analysis
   description: Data analysis with Python
   all-bins: [python3, pip]
   install:
     - kind: brew
       formula: python@3.12
     - kind: uv
       package: pandas
   ---

Supported install kinds: ``brew``, ``node``, ``go``, ``uv``, ``download``.

Always-On Skills
-----------------

For skills that should always be in the system prompt:

.. code-block:: yaml

   ---
   name: coding-standards
   description: Team coding standards
   always: true
   ---

These skills bypass model invocation — they're always present as context.

Skill Discovery Precedence
===========================

Your skill's location determines its priority. Later sources override earlier
ones with the same name:

.. code-block:: text

   1. Extra dirs (lowest priority)
   2. Bundled skills (shipped with OpenClaw)
   3. Managed skills (~/.openclaw/skills/)
   4. Personal agents (~/.agents/skills/)
   5. Project agents (<workspace>/.agents/skills/)
   6. Workspace (<workspace>/skills/)   ← highest priority

A workspace skill named ``summarize-url`` overrides a bundled skill with the
same name.

Source: ``src/agents/skills/workspace.ts``

Token Budget Considerations
============================

Skills compete for space in the system prompt. The total skills prompt is
capped at ``maxSkillsPromptChars = 30,000`` characters. If your skills
exceed this limit, a binary search algorithm finds the largest subset that
fits.

Keep skill prompts concise — a few hundred characters is ideal. Very large
prompts waste context window space that could be used for conversation
history.

Source: ``src/agents/skills/workspace.ts``

Debugging
==========

If your skill isn't appearing:

1. **Check location**: Is the ``SKILL.md`` file in a recognized discovery
   directory?
2. **Check eligibility**: Do all gates pass? (OS, binaries, env vars)
3. **Check name conflicts**: Is another skill with the same name overriding
   yours?
4. **Check limits**: Are you hitting the ``maxCandidatesPerRoot=300`` or
   ``maxSkillsLoadedPerSource=200`` limits?
5. **Check agent filter**: Does the agent's ``skills`` config filter out
   your skill?

Summary
========

Creating a skill requires just one file (``SKILL.md``) with:

1. YAML frontmatter for metadata and gating
2. A prompt body with agent instructions
3. Placement in a recognized discovery directory

The skill system handles discovery, eligibility evaluation, filtering,
prompt formatting, and token budgeting automatically.
