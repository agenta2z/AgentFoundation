============================
How Knowledge is Created
============================

OpenClaw's knowledge base is populated from two sources: agent-written memory
files and session transcript indexing.  This page covers both mechanisms and
the memory-flush system that bridges session context into persistent storage.

.. contents:: On this page
   :depth: 3
   :local:


--------------------------------------------
Agent-Written Memory Files
--------------------------------------------

The agent writes knowledge into Markdown files within the workspace directory.
Two file patterns are recognized:

Evergreen Memory (``MEMORY.md``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Located at the workspace root: ``MEMORY.md`` or ``memory.md``.
- Contains durable facts, preferences, and long-lived context.
- The agent updates this file directly via tool calls.
- **Not subject to temporal decay** -- evergreen files are treated as always
  current during retrieval scoring.

Episodic Memory (``memory/YYYY-MM-DD.md``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Located in the ``memory/`` subdirectory, one file per date.
- Contains daily observations, decisions, and events.
- Created by the agent during memory flush or on explicit instruction.
- **Subject to temporal decay** -- the date in the filename is parsed and
  used to calculate recency scores.  The regex for extraction:

  .. code-block:: text

     /(?:^|\/)memory\/(\d{4})-(\d{2})-(\d{2})\.md$/

- Files outside this naming pattern in ``memory/`` (e.g. ``memory/topics.md``)
  are treated as evergreen and do not decay.

Additional Paths
^^^^^^^^^^^^^^^^

The ``extraPaths`` configuration allows indexing Markdown files from
directories outside the workspace (e.g. shared knowledge bases).  These
paths are resolved relative to the workspace and normalized in
``normalizeExtraMemoryPaths()``.


--------------------------------------------
Memory Flush
--------------------------------------------

The memory flush is a pre-compaction mechanism that gives the agent an
opportunity to persist important context before the session's conversation
history is compressed.

Source: ``src/auto-reply/reply/memory-flush.ts``

When It Triggers
^^^^^^^^^^^^^^^^

The function ``shouldRunMemoryFlush()`` returns ``true`` when all conditions
are met:

1. Memory flush is enabled (default: ``true``).
2. The session's ``totalTokens`` (or ``totalTokensFresh``) exceeds a
   threshold computed as:

   .. code-block:: text

      threshold = contextWindowTokens - reserveTokensFloor - softThresholdTokens

3. The flush has not already run at the current ``compactionCount``.

**Default values:**

.. list-table::
   :header-rows: 1
   :widths: 40 20

   * - Parameter
     - Default
   * - ``softThresholdTokens``
     - ``4000``
   * - ``reserveTokensFloor``
     - From ``DEFAULT_PI_COMPACTION_RESERVE_TOKENS_FLOOR``
   * - ``enabled``
     - ``true``

The Prompt
^^^^^^^^^^

When a flush is triggered, the agent receives a system message and a user
message:

**System prompt** (``DEFAULT_MEMORY_FLUSH_SYSTEM_PROMPT``):

.. code-block:: text

   Pre-compaction memory flush turn.
   The session is near auto-compaction; capture durable memories to disk.
   You may reply, but usually <|silent|> is correct.

**User prompt** (``DEFAULT_MEMORY_FLUSH_PROMPT``):

.. code-block:: text

   Pre-compaction memory flush.
   Store durable memories now (use memory/YYYY-MM-DD.md; create memory/ if needed).
   IMPORTANT: If the file already exists, APPEND new content only and do
   not overwrite existing entries.
   If nothing to store, reply with <|silent|>.

The ``YYYY-MM-DD`` placeholder is replaced with the current date in the
user's timezone (from ``resolveCronStyleNow``).

Configuration
^^^^^^^^^^^^^

Memory flush is configurable under
``agents.defaults.compaction.memoryFlush``:

.. code-block:: yaml

   agents:
     defaults:
       compaction:
         memoryFlush:
           enabled: true
           softThresholdTokens: 4000
           prompt: "..."          # custom prompt (optional)
           systemPrompt: "..."    # custom system prompt (optional)


--------------------------------------------
Session Transcript Indexing
--------------------------------------------

When the ``sources`` configuration includes ``"sessions"``, OpenClaw indexes
session transcript files (``~/.openclaw/agents/<agentId>/sessions/*.jsonl``)
alongside memory files.

Source: ``src/memory/session-files.ts``

Building Session Entries
^^^^^^^^^^^^^^^^^^^^^^^^

The ``buildSessionEntry()`` function transforms a JSONL transcript into an
indexable text document:

1. Read the ``.jsonl`` file line by line.
2. Filter for lines with ``type: "message"`` and ``role: "user"`` or
   ``role: "assistant"``.
3. Extract text content via ``extractSessionText()`` -- handles both plain
   string content and structured ``[{ type: "text", text: "..." }]`` arrays.
4. Apply ``redactSensitiveText()`` to strip credentials and secrets.
5. Format each message as ``"User: <text>"`` or ``"Assistant: <text>"``.
6. Build a ``lineMap`` that maps each output line back to its 1-indexed
   JSONL source line (used for accurate citation line numbers).

.. code-block:: typescript

   export type SessionFileEntry = {
     path: string;       // "sessions/<filename>"
     absPath: string;    // absolute filesystem path
     mtimeMs: number;
     size: number;
     hash: string;       // hash of content + lineMap
     content: string;    // "User: ...\nAssistant: ..." text
     lineMap: number[];  // content line -> JSONL line number mapping
   };

Incremental Session Sync
^^^^^^^^^^^^^^^^^^^^^^^^^

Session files grow continuously as conversations progress.  The sync system
uses a delta-tracking mechanism to avoid re-indexing on every append:

1. ``onSessionTranscriptUpdate`` emits an event when a transcript file is
   written to.
2. The ``scheduleSessionDirty()`` method debounces (5 seconds) and batches
   pending file notifications.
3. ``processSessionDeltaBatch()`` computes how many new bytes and how many
   new messages (newlines) have been appended since the last sync.
4. A sync is triggered only when either ``deltaBytes`` or ``deltaMessages``
   thresholds are exceeded.
5. After indexing, ``resetSessionDelta()`` resets the counters.

This ensures that a rapid series of messages does not cause excessive
re-indexing while still keeping the index reasonably current.
