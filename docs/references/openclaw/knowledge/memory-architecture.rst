==============================
Architecture Deep-Dive
==============================

This page documents the interfaces, types, and structural decisions behind
OpenClaw's memory subsystem.  All references point to source files under
``src/memory/``.

.. contents:: On this page
   :depth: 3
   :local:


-----------------------------------------
MemorySearchManager Interface
-----------------------------------------

Defined in ``src/memory/types.ts``, this is the contract every memory backend
must satisfy:

.. code-block:: typescript

   export interface MemorySearchManager {
     search(
       query: string,
       opts?: {
         maxResults?: number;
         minScore?: number;
         sessionKey?: string;
       },
     ): Promise<MemorySearchResult[]>;

     readFile(params: {
       relPath: string;
       from?: number;
       lines?: number;
     }): Promise<{ text: string; path: string }>;

     status(): MemoryProviderStatus;

     sync?(params?: {
       reason?: string;
       force?: boolean;
       progress?: (update: MemorySyncProgressUpdate) => void;
     }): Promise<void>;

     probeEmbeddingAvailability(): Promise<MemoryEmbeddingProbeResult>;

     probeVectorAvailability(): Promise<boolean>;

     close?(): Promise<void>;
   }

**Method reference:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Purpose
   * - ``search``
     - Semantic + keyword search over indexed memory; returns ranked snippets
       with path, line range, score, and source attribution.
   * - ``readFile``
     - Safe snippet read from memory files; supports ``from``/``lines``
       pagination.  Only ``.md`` files within the workspace or extra paths
       are allowed.
   * - ``status``
     - Returns the full ``MemoryProviderStatus`` snapshot for diagnostics
       (``openclaw status --deep``).
   * - ``sync``
     - Trigger an index sync (optional).  Reasons: ``"search"``,
       ``"watch"``, ``"session-start"``, ``"session-delta"``,
       ``"interval"``, ``"manual"``.
   * - ``probeEmbeddingAvailability``
     - Smoke-test embedding by embedding the string ``"ping"``.
   * - ``probeVectorAvailability``
     - Check if sqlite-vec is loaded and vector search is functional.
   * - ``close``
     - Shut down watchers, timers, pending syncs, and release the database.


-----------------------------------------
MemorySearchResult Type
-----------------------------------------

.. code-block:: typescript

   export type MemorySearchResult = {
     path: string;       // relative path within the workspace
     startLine: number;  // 1-indexed first line of the matched chunk
     endLine: number;    // 1-indexed last line of the matched chunk
     score: number;      // combined relevance score (0..1)
     snippet: string;    // text content of the matched chunk (max 700 chars)
     source: MemorySource;  // "memory" | "sessions"
     citation?: string;  // optional citation string (path#Lstart-Lend)
   };

``MemorySource`` is a union of ``"memory"`` (MEMORY.md, memory/\*.md) and
``"sessions"`` (session transcript .jsonl files).


-----------------------------------------
MemoryProviderStatus Type
-----------------------------------------

Returned by ``status()`` and used by the CLI ``openclaw status --deep`` output:

.. code-block:: typescript

   export type MemoryProviderStatus = {
     backend: "builtin" | "qmd";
     provider: string;          // e.g. "openai", "local", "gemini", "none"
     model?: string;            // e.g. "text-embedding-3-small"
     requestedProvider?: string; // what the user asked for ("auto", "openai", etc.)
     files?: number;            // number of indexed files
     chunks?: number;           // number of indexed chunks
     dirty?: boolean;           // true if a sync is needed
     workspaceDir?: string;     // resolved workspace directory
     dbPath?: string;           // path to the SQLite database
     extraPaths?: string[];     // additional paths being indexed
     sources?: MemorySource[];  // active sources ("memory", "sessions")
     sourceCounts?: Array<{
       source: MemorySource;
       files: number;
       chunks: number;
     }>;
     cache?: {
       enabled: boolean;
       entries?: number;
       maxEntries?: number;
     };
     fts?: {
       enabled: boolean;
       available: boolean;
       error?: string;
     };
     fallback?: {
       from: string;       // original provider that failed
       reason?: string;     // why it failed
     };
     vector?: {
       enabled: boolean;
       available?: boolean;
       extensionPath?: string;
       loadError?: string;
       dims?: number;       // embedding dimensions (e.g. 256, 1536)
     };
     batch?: {
       enabled: boolean;
       failures: number;    // current failure count
       limit: number;       // auto-disable threshold (2)
       wait: boolean;
       concurrency: number;
       pollIntervalMs: number;
       timeoutMs: number;
       lastError?: string;
       lastProvider?: string;
     };
     custom?: Record<string, unknown>;  // searchMode, providerUnavailableReason
   };


--------------------------------------------
FallbackMemoryManager
--------------------------------------------

Defined in ``src/memory/search-manager.ts``, this class wraps a QMD primary
manager with a lazy builtin fallback:

.. code-block:: text

   getMemorySearchManager()
       |
       +-- backend == "qmd"?
       |       |
       |       +-- QmdMemoryManager.create()
       |       |       |
       |       |       +-- success: wrap in FallbackMemoryManager
       |       |       |             primary = QmdMemoryManager
       |       |       |             fallback = lazy MemoryIndexManager
       |       |       |
       |       |       +-- failure: log warning, fall through
       |       |
       |       v
       +-- MemoryIndexManager.get()
               |
               +-- success: return manager
               +-- failure: return { manager: null, error }

**FallbackMemoryManager behavior:**

- All interface methods delegate to the primary (QMD) first.
- On the first ``search()`` error, ``primaryFailed`` is set to ``true``,
  the primary is closed, the cache entry is evicted, and all subsequent calls
  go to the lazily-constructed builtin fallback.
- The ``status()`` method merges the fallback status with a ``fallback``
  field indicating the original QMD failure reason.
- The cache key is evicted on failure so the next request can retry QMD
  with a fresh manager instance.


--------------------------------------------
Instance Caching
--------------------------------------------

Both backends use module-level caches to avoid re-creating managers:

- **Builtin:** ``INDEX_CACHE`` (Map keyed by ``agentId:workspaceDir:settings``)
  in ``src/memory/manager.ts``.
- **QMD:** ``QMD_MANAGER_CACHE`` (Map keyed by ``agentId:stableSerialize(config)``)
  in ``src/memory/search-manager.ts``.

Managers are evicted from their cache on ``close()`` or on primary failure
in the ``FallbackMemoryManager``.


--------------------------------------------
Configuration Resolution
--------------------------------------------

Backend selection is resolved in ``src/memory/backend-config.ts``:

.. code-block:: typescript

   export function resolveMemoryBackendConfig(params: {
     cfg: OpenClawConfig;
     agentId: string;
   }): ResolvedMemoryBackendConfig;

Returns ``{ backend, citations, qmd? }``.  The ``backend`` field is either
``"builtin"`` (default) or ``"qmd"`` when ``memory.backend`` is set to
``"qmd"`` in config.

The ``citations`` field controls how source citations are attached to search
results: ``"auto"`` (default -- on for direct chats, off for groups),
``"on"`` (always), or ``"off"`` (never).
