.. _knowledge-storage:

============================
SQLite Storage Schema
============================

The builtin memory backend stores all indexed knowledge in a single SQLite
database.  This page documents the schema, database lifecycle, and sync
mechanisms.

.. contents:: On this page
   :depth: 3
   :local:


--------------------------------------------
Database Location
--------------------------------------------

The database is stored at a path resolved from configuration, defaulting to::

   ~/.openclaw/state/memory/<agentId>.sqlite

The path is resolved via ``resolveUserPath(this.settings.store.path)`` in
``MemoryManagerSyncOps.openDatabase()``.  The parent directory is created
automatically via ``ensureDir()``.

The database is opened with ``node:sqlite``'s ``DatabaseSync`` in synchronous
mode, with ``allowExtension: true`` when vector search is enabled (required
for loading the sqlite-vec extension).


--------------------------------------------
Schema: 5+1 Tables
--------------------------------------------

The schema is applied by ``ensureMemoryIndexSchema()`` in
``src/memory/memory-schema.ts``.

meta
^^^^

Key-value store for index metadata (provider, model, chunk settings).

.. code-block:: sql

   CREATE TABLE IF NOT EXISTS meta (
     key   TEXT PRIMARY KEY,
     value TEXT NOT NULL
   );

The single active key is ``memory_index_meta_v1``, storing a JSON object:

.. code-block:: typescript

   type MemoryIndexMeta = {
     model: string;           // embedding model name
     provider: string;        // "openai", "local", "gemini", etc.
     providerKey?: string;    // hash of provider config (for cache keying)
     sources?: MemorySource[];// ["memory"] or ["memory","sessions"]
     chunkTokens: number;     // e.g. 400
     chunkOverlap: number;    // e.g. 80
     vectorDims?: number;     // e.g. 256, 1536
   };

When any of these fields change (e.g. switching embedding providers), a full
reindex is triggered.

files
^^^^^

Tracks each indexed file and its hash for incremental sync.

.. code-block:: sql

   CREATE TABLE IF NOT EXISTS files (
     path   TEXT PRIMARY KEY,
     source TEXT NOT NULL DEFAULT 'memory',
     hash   TEXT NOT NULL,
     mtime  INTEGER NOT NULL,
     size   INTEGER NOT NULL
   );

``source`` is ``"memory"`` or ``"sessions"``.  On sync, files whose hash
matches the stored hash are skipped.

chunks
^^^^^^

Stores every text chunk with its embedding vector.

.. code-block:: sql

   CREATE TABLE IF NOT EXISTS chunks (
     id         TEXT PRIMARY KEY,
     path       TEXT NOT NULL,
     source     TEXT NOT NULL DEFAULT 'memory',
     start_line INTEGER NOT NULL,
     end_line   INTEGER NOT NULL,
     hash       TEXT NOT NULL,
     model      TEXT NOT NULL,
     text       TEXT NOT NULL,
     embedding  TEXT NOT NULL,
     updated_at INTEGER NOT NULL
   );

   CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
   CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);

- ``id`` is a SHA-256 hash of ``source:path:startLine:endLine:chunkHash:model``.
- ``embedding`` is a JSON array of floats (``JSON.stringify(embedding)``).
- ``model`` identifies the embedding model for multi-model coexistence.

chunks_fts (FTS5 Virtual Table)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Full-text search index over chunk text for keyword retrieval.

.. code-block:: sql

   CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
     text,
     id UNINDEXED,
     path UNINDEXED,
     source UNINDEXED,
     model UNINDEXED,
     start_line UNINDEXED,
     end_line UNINDEXED
   );

Only the ``text`` column is indexed for FTS; all other columns are stored
but ``UNINDEXED`` for efficient retrieval without inflating the index.

If FTS5 is not available in the SQLite build, the table creation is caught
and ``fts.available`` is set to ``false``.

chunks_vec (vec0 Virtual Table)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Vector similarity search index, created dynamically when the first embedding
is stored:

.. code-block:: sql

   CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
     id TEXT PRIMARY KEY,
     embedding FLOAT[<dimensions>]
   );

- Dimensions are determined at runtime from the embedding provider (e.g. 256
  for local GemmaEmbedding, 1536 for OpenAI text-embedding-3-small).
- If the dimensions change (e.g. provider switch), the table is dropped and
  recreated.
- Embeddings are stored as ``Float32Array`` buffers for native ``vec0``
  compatibility.

embedding_cache
^^^^^^^^^^^^^^^

Caches embeddings across reindexes to avoid redundant API calls.

.. code-block:: sql

   CREATE TABLE IF NOT EXISTS embedding_cache (
     provider     TEXT NOT NULL,
     model        TEXT NOT NULL,
     provider_key TEXT NOT NULL,
     hash         TEXT NOT NULL,
     embedding    TEXT NOT NULL,
     dims         INTEGER,
     updated_at   INTEGER NOT NULL,
     PRIMARY KEY (provider, model, provider_key, hash)
   );

   CREATE INDEX IF NOT EXISTS idx_embedding_cache_updated_at
     ON embedding_cache(updated_at);

- ``provider_key`` is a hash of the provider configuration (base URL, headers,
  model) to distinguish embeddings from different API endpoints.
- Cache is pruned by ``pruneEmbeddingCacheIfNeeded()`` when the entry count
  exceeds ``maxEntries``.  Oldest entries (by ``updated_at``) are evicted
  first.


--------------------------------------------
Safe Reindex (Temp DB + Atomic Swap)
--------------------------------------------

When a full reindex is required (model change, chunk settings change, source
set change, forced sync), the builtin backend uses a safe reindex strategy
to avoid corrupting the live index:

.. code-block:: text

   1. Create a temporary database at <dbPath>.tmp-<UUID>
   2. Apply schema to temp DB
   3. Seed embedding cache from the original DB
   4. Swap this.db to point to temp DB
   5. Run full sync (memory files + session files)
   6. Write new meta record
   7. Prune embedding cache
   8. Close both databases
   9. Atomic swap:
      a. Rename original -> <dbPath>.backup-<UUID>
      b. Rename temp -> <dbPath>
      c. If (b) fails, restore backup
      d. Delete backup
   10. Reopen database at <dbPath>
   11. Reset vector state (will re-probe on next search)

This ensures that a crash during reindex leaves the original database intact.
For test environments (``OPENCLAW_TEST_FAST=1``), an unsafe in-place reindex
is used to reduce filesystem churn.


--------------------------------------------
File Watching
--------------------------------------------

The ``ensureWatcher()`` method sets up a Chokidar file watcher for
incremental re-indexing:

**Watched paths:**

.. code-block:: text

   <workspaceDir>/MEMORY.md
   <workspaceDir>/memory.md
   <workspaceDir>/memory/**/*.md
   <extraPaths>/**/*.md           (for each configured extra path)

**Configuration:**

- ``ignoreInitial: true`` -- do not trigger on existing files at startup.
- ``awaitWriteFinish.stabilityThreshold`` -- configurable via
  ``sync.watchDebounceMs`` (default 1500ms).
- ``awaitWriteFinish.pollInterval`` -- 100ms.
- Ignored directories: ``.git``, ``node_modules``, ``.pnpm-store``,
  ``.venv``, ``venv``, ``.tox``, ``__pycache__``.

**Events:** ``add``, ``change``, ``unlink`` all set ``this.dirty = true``
and schedule a debounced sync via ``scheduleWatchSync()``.

**Session listener:** The ``ensureSessionListener()`` method subscribes to
``onSessionTranscriptUpdate`` events.  When a transcript file for the
current agent is modified, a 5-second debounced delta check determines
whether to trigger a session sync.

**Interval sync:** If ``sync.intervalMinutes`` is configured, a periodic
``setInterval`` triggers full syncs regardless of file changes.
