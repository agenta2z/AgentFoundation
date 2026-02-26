.. _knowledge-index:

============================
Knowledge System Overview
============================

How OpenClaw remembers and recalls -- a deep-focus reference for the memory
subsystem that gives agents persistent, searchable knowledge across sessions.

.. contents:: On this page
   :depth: 3
   :local:

-----------
Big Picture
-----------

OpenClaw's knowledge system answers a single question: *how does an agent
remember things it learned yesterday and find them tomorrow?*

The answer involves two kinds of memory files, a SQLite index that turns those
files into searchable chunks, and a retrieval pipeline that combines vector
similarity with keyword matching to surface the most relevant snippets at
query time.

.. code-block:: text

   +------------------+     +------------------+     +------------------+
   |   MEMORY.md      |     | memory/          |     | session          |
   |  (evergreen)     |     | YYYY-MM-DD.md    |     | transcripts      |
   |                  |     | (episodic)       |     | (.jsonl)         |
   +--------+---------+     +--------+---------+     +--------+---------+
            |                        |                        |
            +----------+-------------+------------+-----------+
                       |                          |
                       v                          v
              +--------+---------+       +--------+---------+
              | Builtin SQLite   |       | QMD External     |
              | MemoryIndexMgr   |       | QmdMemoryManager |
              +--------+---------+       +--------+---------+
                       |                          |
                       +----------+---------------+
                                  |
                          +-------v--------+
                          | FallbackMemory |
                          | Manager        |
                          +-------+--------+
                                  |
                          +-------v--------+
                          | memory_search  |
                          | memory_get     |
                          | (agent tools)  |
                          +----------------+


----------------------------
Knowledge Lifecycle
----------------------------

Every piece of knowledge follows a five-stage lifecycle:

1. **Create** -- The agent writes Markdown files (``MEMORY.md``,
   ``memory/YYYY-MM-DD.md``) or session transcripts accumulate as ``.jsonl``
   files.  A *memory flush* may trigger just before context compaction to
   capture durable facts.

2. **Store** -- Files are chunked (default 400 tokens, 80-token overlap),
   embedded, and persisted into a SQLite database with five+one tables
   (``meta``, ``files``, ``chunks``, ``chunks_fts``, ``chunks_vec``,
   ``embedding_cache``).

3. **Index** -- Chunks are inserted into both an FTS5 full-text index and a
   ``vec0`` virtual table for vector search.  Chokidar file watchers and
   session-transcript listeners trigger incremental re-indexing with a
   configurable debounce.

4. **Retrieve** -- When the agent calls ``memory_search``, the query flows
   through a multi-stage pipeline: query expansion, parallel vector + keyword
   search, hybrid merge (0.7/0.3 weighting), optional temporal decay, optional
   MMR diversity re-ranking, score threshold, and top-K selection.

5. **Use** -- The agent receives ``MemorySearchResult[]`` with paths, line
   ranges, scores, and snippets.  It can drill into specific lines with
   ``memory_get``.  Citations are optionally appended depending on the chat
   context (direct vs. group).


------------------------------
Dual-Backend Architecture
------------------------------

OpenClaw ships two memory backends, selectable via ``memory.backend`` in
configuration:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - **Builtin** (default)
     - **QMD** (external)
   * - Backend ID
     - ``"builtin"``
     - ``"qmd"``
   * - Implementation
     - ``MemoryIndexManager``
     - ``QmdMemoryManager``
   * - Storage
     - Node SQLite (``node:sqlite``) with sqlite-vec
     - QMD CLI + its own index.sqlite
   * - Embedding
     - Local GGUF, OpenAI, Gemini, Voyage, Mistral
     - QMD's built-in models
   * - Search
     - Hybrid (vector + FTS5 BM25)
     - ``search`` / ``vsearch`` / ``query`` modes
   * - MCP Bridge
     - N/A
     - ``mcporter`` for MCP-based tool calls

The ``getMemorySearchManager()`` factory in ``src/memory/search-manager.ts``
resolves the backend.  When QMD is selected, a ``FallbackMemoryManager``
wraps the QMD primary with a lazy builtin fallback -- if QMD fails at
runtime, the system transparently degrades to the builtin index.


---------------------------------
Graceful Degradation Chain
---------------------------------

The knowledge system is designed to never hard-fail.  It degrades through four
levels:

.. code-block:: text

   Level 1:  Full Hybrid        vector search + FTS5 keyword search
                  |              (vectorWeight=0.7, textWeight=0.3)
                  v
   Level 2:  Vector-Only        FTS disabled or unavailable
                  |              (pure cosine similarity ranking)
                  v
   Level 3:  FTS-Only           No embedding provider available
                  |              (query expansion + BM25 keyword search)
                  v
   Level 4:  Disabled           Neither provider nor FTS available
                                 (memory_search returns empty results)

**How degradation is decided:**

- If no API key is found for any embedding provider and local embeddings
  cannot be loaded, ``createEmbeddingProvider()`` returns
  ``provider: null``.  The manager enters FTS-only mode.
- If the FTS5 SQLite extension fails to load, keyword search is skipped
  and only vector search is used.
- If both are unavailable, ``search()`` returns ``[]``.

Within the QMD backend, the ``FallbackMemoryManager`` adds another layer:
if the QMD CLI process fails, the system falls back to the builtin index.


-------------------------------
Class Hierarchy
-------------------------------

The builtin backend uses a three-level inheritance chain:

.. code-block:: text

   MemoryManagerSyncOps          (abstract)
       |                         - File watching (chokidar)
       |                         - Session transcript listeners
       |                         - Sync orchestration (runSync, runSafeReindex)
       |                         - Database open/schema/meta
       |                         - Interval sync, watch debounce
       v
   MemoryManagerEmbeddingOps     (abstract, extends SyncOps)
       |                         - Batch embedding (buildEmbeddingBatches)
       |                         - Embedding cache (load/upsert/prune)
       |                         - Retry with exponential backoff
       |                         - Batch failure tracking (auto-disable at 2)
       |                         - Provider-specific batch runners (OpenAI, Gemini, Voyage)
       |                         - File indexing (indexFile)
       v
   MemoryIndexManager            (concrete, extends EmbeddingOps)
                                 - Implements MemorySearchManager interface
                                 - search(), readFile(), status(), sync(), close()
                                 - Vector + keyword search orchestration
                                 - Hybrid merge, temporal decay, MMR
                                 - Instance cache (INDEX_CACHE)
                                 - Factory: MemoryIndexManager.get()

Each class is defined in its own file:

- ``src/memory/manager-sync-ops.ts`` -- ``MemoryManagerSyncOps``
- ``src/memory/manager-embedding-ops.ts`` -- ``MemoryManagerEmbeddingOps``
- ``src/memory/manager.ts`` -- ``MemoryIndexManager``


---------------------------------------------
Memory v2 Research Direction
---------------------------------------------

The file ``docs/experiments/research/memory.md`` outlines a research direction
called **Workspace Memory v2** (offline-first).  Key concepts:

**Retain / Recall / Reflect**

- **Retain** -- Normalize daily logs into narrative, self-contained facts.
  Each fact has a type prefix (``W`` world, ``B`` experience, ``O`` opinion,
  ``S`` observation) and entity mentions (``@Peter``, ``@warelay``).

- **Recall** -- Query the derived index for lexical, entity-centric, temporal,
  and opinion-aware retrieval.  Results carry ``kind``, ``timestamp``,
  ``entities``, ``content``, and ``source`` (file + line).

- **Reflect** -- A scheduled job that updates entity summary pages
  (``bank/entities/*.md``), evolves opinion confidence, and proposes edits to
  core memory files.

**Proposed layout** (not yet implemented):

.. code-block:: text

   ~/.openclaw/workspace/
     memory.md                    # durable facts + preferences (core)
     memory/
       YYYY-MM-DD.md              # daily log (append; narrative)
     bank/                        # typed memory pages (stable)
       world.md                   # objective facts
       experience.md              # first-person experience
       opinions.md                # subjective prefs + confidence
       entities/
         Peter.md
         ...

This is exploratory and not yet part of the production codebase.


.. toctree::
   :maxdepth: 2
   :caption: Knowledge System

   memory-architecture
   knowledge-creation
   knowledge-storage
   knowledge-retrieval
   embeddings
   qmd-backend
   memory-tools
