.. _skills-index:

=============
Skills System
=============

Skills are **markdown-defined agent capability modules** that extend what an
OpenClaw agent can do. Each skill is a directory containing a ``SKILL.md`` file
with YAML frontmatter that specifies metadata, requirements, and prompt content
injected into the agent's system prompt.

What Skills Do
==============

When an agent processes a message, its system prompt includes the content of all
eligible skills. This gives the LLM knowledge about available capabilities
(e.g., how to use Spotify, manage Trello boards, or generate images) without
requiring changes to the core codebase.

Skills can:

- Teach the agent how to use external tools and services
- Provide domain-specific instructions and workflows
- Define slash commands for user-invocable actions
- Gate themselves behind OS, binary, or environment requirements
- Auto-install their dependencies (brew, npm, go, uv, download)

How Skills Work
===============

.. code-block:: text

   ┌─────────────────────────────────────────────────┐
   │                  Skill Loading                   │
   │                                                  │
   │  6 Source Layers (ascending precedence)          │
   │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
   │  │ Extra    │→ │ Bundled  │→ │ Managed  │→ ... │
   │  └──────────┘  └──────────┘  └──────────┘      │
   │                      │                           │
   │              ┌───────▼────────┐                  │
   │              │ Merge by Name  │                  │
   │              └───────┬────────┘                  │
   │                      │                           │
   │              ┌───────▼────────┐                  │
   │              │  Filter &      │                  │
   │              │  Eligibility   │                  │
   │              └───────┬────────┘                  │
   │                      │                           │
   │              ┌───────▼────────┐                  │
   │              │ Format for     │                  │
   │              │ System Prompt  │                  │
   │              └────────────────┘                  │
   └─────────────────────────────────────────────────┘

1. **Discovery**: Skills are loaded from 6 directory layers with ascending
   precedence (see :doc:`skill-discovery`)
2. **Merging**: Later sources override earlier ones by skill name
3. **Filtering**: Skills are filtered by config, bundled allowlists, and
   runtime eligibility (see :doc:`skill-eligibility`)
4. **Formatting**: Eligible skills are formatted into the system prompt with
   path compaction (see :doc:`skill-composition`)

Bundled Skills Catalog
======================

OpenClaw ships with 50+ bundled skills in the ``skills/`` directory:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Category
     - Skills
     - Description
   * - Productivity
     - apple-notes, apple-reminders, bear-notes, notion, obsidian, things-mac, trello
     - Note-taking, task management, and knowledge tools
   * - Communication
     - discord, slack, imsg, himalaya
     - Messaging and email integrations
   * - Development
     - coding-agent, gh-issues, github, clawhub
     - Code assistance and repository management
   * - Media
     - openai-image-gen, video-frames, camsnap, peekaboo, gifgrep
     - Image generation, video processing, screenshots
   * - Audio/Voice
     - sherpa-onnx-tts, openai-whisper, openai-whisper-api, voice-call, songsee
     - Text-to-speech, transcription, voice calls
   * - Music
     - spotify-player, sonoscli
     - Music playback and control
   * - AI Models
     - gemini, oracle
     - Direct model integrations
   * - Web/Search
     - xurl, blogwatcher, summarize
     - Web content extraction and monitoring
   * - Smart Home
     - openhue
     - Smart lighting control
   * - System
     - tmux, healthcheck, session-logs, model-usage, skill-creator
     - System management and diagnostics
   * - Utility
     - weather, goplaces, nano-pdf, canvas, mcporter
     - Various utility skills

.. toctree::
   :maxdepth: 2

   skill-definition
   skill-discovery
   skill-execution
   skill-eligibility
   skill-composition
