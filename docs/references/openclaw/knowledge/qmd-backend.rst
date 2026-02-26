============================
External QMD Backend
============================

QMD is an external CLI tool that provides an alternative memory backend with
its own embedding models, vector index, and search capabilities.  When
configured, it replaces the builtin SQLite-based index while retaining the
same ``MemorySearchManager`` interface.

.. contents:: On this page
   :depth: 3
   :local:


--------------------------------------------
Overview
--------------------------------------------

Source: ``src/memory/qmd-manager.ts``, ``src/memory/backend-config.ts``

The QMD backend is activated by setting ``memory.backend: "qmd"`` in
configuration.  It delegates indexing and search to the ``qmd`` CLI binary,
which maintains its own SQLite index with built-in ML models for embeddings
and retrieval.

.. code-block:: text

   OpenClaw Agent
       |
       v
   QmdMemoryManager
       |
       +-- qmd update          (index files)
       +-- qmd embed           (compute embeddings)
       +-- qmd search/vsearch/query  (search)
       +-- qmd collection add  (manage collections)
       |
       v
   QMD CLI Binary
       |
       v
   $XDG_CACHE_HOME/qmd/index.sqlite


--------------------------------------------
QMD CLI Commands
--------------------------------------------

``QmdMemoryManager`` shells out to the QMD binary via Node's ``spawn()``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Purpose
   * - ``qmd collection add <path> --name <name> --mask <pattern>``
     - Register a file collection for indexing
   * - ``qmd collection remove <name>``
     - Remove a collection
   * - ``qmd collection list --json``
     - List all registered collections with paths and patterns
   * - ``qmd update``
     - Scan collections and update the index
   * - ``qmd embed``
     - Compute embeddings for unembedded documents
   * - ``qmd search <query> --json -n <limit>``
     - BM25 keyword search
   * - ``qmd vsearch <query> --json -n <limit>``
     - Vector similarity search
   * - ``qmd query <query> --json -n <limit>``
     - Full query with expansion + reranking (slow on CPU)

**Timeouts:**

.. list-table::
   :header-rows: 1
   :widths: 30 20

   * - Operation
     - Default Timeout
   * - Command (collection add/remove/list)
     - 30 seconds
   * - Update
     - 120 seconds
   * - Embed
     - 120 seconds
   * - Search
     - 4 seconds


--------------------------------------------
Collection Storage
--------------------------------------------

QMD organizes indexed files into *collections*.  Each collection maps to a
filesystem path with a glob pattern:

.. code-block:: typescript

   type ResolvedQmdCollection = {
     name: string;    // e.g. "memory-root-myagent"
     path: string;    // absolute filesystem path
     pattern: string; // glob pattern (e.g. "MEMORY.md", "**/*.md")
     kind: "memory" | "custom" | "sessions";
   };

**Default collections** (when ``includeDefaultMemory`` is true):

.. list-table::
   :header-rows: 1
   :widths: 30 30 20 20

   * - Name Pattern
     - Path
     - Pattern
     - Kind
   * - ``memory-root-<agentId>``
     - ``<workspaceDir>``
     - ``MEMORY.md``
     - memory
   * - ``memory-alt-<agentId>``
     - ``<workspaceDir>``
     - ``memory.md``
     - memory
   * - ``memory-dir-<agentId>``
     - ``<workspaceDir>/memory``
     - ``**/*.md``
     - memory

**Custom collections** can be added via ``memory.qmd.paths``:

.. code-block:: yaml

   memory:
     backend: qmd
     qmd:
       paths:
         - path: ~/shared-knowledge
           pattern: "**/*.md"
           name: shared-kb

Collection names are scoped to the agent ID to prevent collisions when
multiple agents share the same QMD instance.  Names are sanitized to
lowercase alphanumeric with hyphens.


--------------------------------------------
State Isolation
--------------------------------------------

Each agent gets its own QMD state directory:

.. code-block:: text

   ~/.openclaw/state/agents/<agentId>/qmd/
     xdg-config/        # QMD config (XDG_CONFIG_HOME override)
     xdg-cache/          # QMD cache (XDG_CACHE_HOME override)
       qmd/
         index.sqlite    # the QMD index database
         models/ -> ~/.cache/qmd/models/  (symlink)
     sessions/           # exported session transcripts (if enabled)

**Model sharing:** To avoid downloading ML models per agent, the manager
symlinks ``~/.cache/qmd/models/`` into the per-agent cache directory.  On
Windows, a directory junction is used if symlinks require elevated privileges.


--------------------------------------------
Update Lifecycle
--------------------------------------------

.. code-block:: text

   runUpdate(reason, force?)
       |
       +-- Export sessions (if enabled)
       |       |
       |       +-- buildSessionEntry() for each .jsonl
       |       +-- renderSessionMarkdown() -> .md file
       |       +-- Write to sessions/ directory
       |       +-- Remove stale .md files
       |
       +-- qmd update (with retry for boot runs)
       |       |
       |       +-- Up to 3 attempts for boot
       |       +-- 500ms base delay with exponential backoff
       |       +-- Null-byte collection repair on ENOTDIR
       |
       +-- qmd embed (if searchMode != "search")
       |       |
       |       +-- Runs under a global lock (qmdEmbedQueueTail)
       |       +-- Exponential backoff on failure (60s base, 1h max)
       |       +-- Failure count tracking
       |
       +-- Clear doc path cache
       +-- Record lastUpdateAt

**Update scheduling:**

- **On boot:** ``update.onBoot`` (default: ``true``).  Optionally wait for
  completion (``waitForBootSync``).
- **Periodic:** ``update.intervalMs`` (default: 5 minutes).
- **Debounce:** ``update.debounceMs`` (default: 15 seconds) prevents
  rapid-fire updates.
- **Forced:** ``sync()`` with ``force: true`` bypasses debounce.

**Queued forced updates:** When ``runUpdate`` is called with ``force`` while
another update is in progress, the request is queued.  A drain loop processes
queued updates sequentially after the current update completes.


--------------------------------------------
Search Modes
--------------------------------------------

The ``searchMode`` configuration controls which QMD command is used:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Mode
     - Command
     - Description
   * - ``search``
     - ``qmd search``
     - BM25 keyword search.  Fast, no embedding required.
       **Default** (recommended for CPU-only systems).
   * - ``vsearch``
     - ``qmd vsearch``
     - Vector similarity search.  Requires ``qmd embed``.
   * - ``query``
     - ``qmd query``
     - Full query with expansion and reranking.  Slowest but highest recall.

For Han-script (CJK) queries, ``normalizeHanBm25Query()`` extracts keywords
(up to 12) to improve BM25 matching.

**Multi-collection search:** When multiple collections are configured, each
is searched separately and results are merged by document ID, keeping the
highest score for duplicates.


--------------------------------------------
mcporter MCP Bridge
--------------------------------------------

Source: ``src/memory/qmd-manager.ts``

When ``memory.qmd.mcporter.enabled`` is ``true``, search calls are routed
through ``mcporter`` -- an MCP (Model Context Protocol) bridge that wraps
QMD as a persistent daemon:

.. code-block:: text

   QmdMemoryManager
       |
       v
   mcporter call <server>.<tool> --args <json> --output json --timeout <ms>
       |
       v
   QMD MCP Server (persistent daemon)

**Configuration:**

.. code-block:: typescript

   type ResolvedQmdMcporterConfig = {
     enabled: boolean;     // default: false
     serverName: string;   // default: "qmd"
     startDaemon: boolean; // default: true (when enabled)
   };

**Tool mapping:**

.. list-table::
   :header-rows: 1
   :widths: 20 30

   * - Search Mode
     - MCP Tool
   * - ``search``
     - ``qmd.search``
   * - ``vsearch``
     - ``qmd.vector_search``
   * - ``query``
     - ``qmd.deep_search``

**Daemon lifecycle:** The daemon is started lazily on the first search call
(``mcporter daemon start``).  A global flag prevents multiple starts.  If
the start fails, the flag is cleared so the next search retries.


--------------------------------------------
Session Export
--------------------------------------------

When ``memory.qmd.sessions.enabled`` is ``true``, session transcripts are
exported as Markdown files into a dedicated collection:

1. ``listSessionFilesForAgent()`` finds all ``.jsonl`` transcripts.
2. ``buildSessionEntry()`` extracts user/assistant messages.
3. ``renderSessionMarkdown()`` formats them as a Markdown document with a
   header: ``# Session <session-id>``.
4. Files are written to the sessions export directory.
5. Stale Markdown files (no matching .jsonl) are removed.
6. Exported files are tracked via ``exportedSessionState`` to avoid
   re-rendering unchanged sessions.

**Retention:** If ``sessions.retentionDays`` is set, sessions older than
the cutoff are skipped during export.

The session collection is automatically added to the QMD collection list
with the name ``sessions-<agentId>`` and pattern ``**/*.md``.


--------------------------------------------
Scope Control
--------------------------------------------

QMD search can be restricted by session context via the ``scope``
configuration:

.. code-block:: typescript

   // Default scope: allow direct chats only, deny groups/channels
   const DEFAULT_QMD_SCOPE = {
     default: "deny",
     rules: [
       { action: "allow", match: { chatType: "direct" } },
     ],
   };

The ``isScopeAllowed()`` method extracts channel and chat type from the
session key and evaluates them against the scope rules.  Denied searches
return ``[]`` with a warning log.


--------------------------------------------
Limits
--------------------------------------------

.. code-block:: typescript

   const DEFAULT_QMD_LIMITS = {
     maxResults: 6,
     maxSnippetChars: 700,
     maxInjectedChars: 4_000,
     timeoutMs: 4_000,
   };

- ``maxResults`` caps the number of returned results.
- ``maxSnippetChars`` truncates individual snippet text.
- ``maxInjectedChars`` is a total budget -- results are clamped so the sum
  of snippet lengths does not exceed this value.
- Results are also diversified by source (interleaving memory and session
  results) via ``diversifyResultsBySource()``.
