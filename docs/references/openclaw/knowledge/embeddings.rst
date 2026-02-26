====================================
Embedding Providers & Operations
====================================

This page documents the embedding providers available to OpenClaw, the
auto-selection cascade, and the embedding operations that transform text
chunks into vectors for semantic search.

.. contents:: On this page
   :depth: 3
   :local:


--------------------------------------------
Provider Table
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 12 12 30 20 26

   * - Provider ID
     - Type
     - Default Model
     - Max Input Tokens
     - Notes
   * - ``local``
     - Local
     - ``hf:ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf``
     - N/A (model-dependent)
     - Uses ``node-llama-cpp``; requires Node 22+.  Model is downloaded on
       first use from Hugging Face.
   * - ``openai``
     - Remote
     - ``text-embedding-3-small``
     - 8192
     - Standard OpenAI embeddings API.  Also supports ``text-embedding-3-large``
       and ``text-embedding-ada-002``.
   * - ``gemini``
     - Remote
     - ``gemini-embedding-001``
     - 2048 (``text-embedding-004``)
     - Google AI Studio / Vertex AI.  Supports API key rotation.
   * - ``voyage``
     - Remote
     - ``voyage-4-large``
     - 32000 (``voyage-3``, ``voyage-code-3``)
     - Voyage AI embeddings with batch API support.
   * - ``mistral``
     - Remote
     - ``mistral-embed``
     - N/A
     - Mistral AI embeddings endpoint.


--------------------------------------------
Auto-Selection Cascade
--------------------------------------------

When ``provider`` is set to ``"auto"`` (the default), OpenClaw tries each
provider in sequence until one succeeds:

.. code-block:: text

   1. local    -- only if a local model file already exists on disk
   2. openai   -- if OPENAI_API_KEY is available
   3. gemini   -- if GEMINI_API_KEY or Vertex credentials are available
   4. voyage   -- if VOYAGE_API_KEY is available
   5. mistral  -- if MISTRAL_API_KEY is available

**Rules:**

- **Local is tried first** only if ``canAutoSelectLocal()`` returns ``true``
  (the configured ``modelPath`` points to an existing file on disk, not a
  ``hf:`` or ``https:`` URL).
- Each remote provider is tried by calling ``createProvider()``.  If it
  throws a "missing API key" error, the cascade continues to the next.
- If a non-auth error occurs (e.g. network failure), it is thrown immediately
  (no further cascade).
- **If all providers fail** with missing-key errors, the system enters
  **FTS-only mode** (``provider: null``).  Memory search still works via
  keyword matching.

**Fallback configuration:**

A ``fallback`` setting allows specifying a backup provider.  If the primary
fails:

1. Try the primary provider.
2. If it fails, try the ``fallback`` provider.
3. If both fail with auth errors, enter FTS-only mode.
4. If a non-auth error occurs on fallback, throw.

At runtime, if embedding calls fail with errors matching
``/embedding|embeddings|batch/i``, the sync layer can dynamically activate the
fallback provider and trigger a full reindex.


--------------------------------------------
L2 Normalization
--------------------------------------------

Source: ``src/memory/embeddings.ts``

All embedding vectors (from any provider) are L2-normalized before storage:

.. code-block:: typescript

   function sanitizeAndNormalizeEmbedding(vec: number[]): number[] {
     const sanitized = vec.map(v => Number.isFinite(v) ? v : 0);
     const magnitude = Math.sqrt(
       sanitized.reduce((sum, v) => sum + v * v, 0)
     );
     if (magnitude < 1e-10) return sanitized;
     return sanitized.map(v => v / magnitude);
   }

Normalization ensures that cosine similarity is equivalent to dot product,
and that the ``vec_distance_cosine`` function in sqlite-vec produces
consistent results regardless of provider-specific output scaling.


--------------------------------------------
Chunking
--------------------------------------------

Source: ``src/memory/internal.ts`` (``chunkMarkdown``)

Before embedding, Markdown files are split into chunks:

**Default parameters:**

- **Tokens:** 400 (``DEFAULT_CHUNK_TOKENS``)
- **Overlap:** 80 (``DEFAULT_CHUNK_OVERLAP``)

**Algorithm:**

1. Split the file into lines.
2. Accumulate lines into a current chunk until ``maxChars`` (tokens * 4)
   is exceeded.
3. Flush the chunk, recording ``startLine`` and ``endLine``.
4. Carry over the last ``overlapChars`` (overlap * 4) characters worth of
   lines into the next chunk for context continuity.
5. Each chunk gets a SHA-256 ``hash`` of its text for change detection.

For session files, ``remapChunkLines()`` adjusts the chunk line numbers
using the ``lineMap`` to point back to the original JSONL line numbers.


--------------------------------------------
Batch Embedding
--------------------------------------------

Source: ``src/memory/manager-embedding-ops.ts`` and 14 ``batch-*.ts`` files.

**Batch files in ``src/memory/``:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Purpose
   * - ``batch-runner.ts``
     - Generic batch orchestration (polling, concurrency, timeouts)
   * - ``batch-openai.ts``
     - OpenAI batch embedding API integration
   * - ``batch-gemini.ts``
     - Gemini batch embedding (``asyncBatchEmbedContent``)
   * - ``batch-voyage.ts``
     - Voyage batch embedding API
   * - ``batch-http.ts``
     - HTTP utilities for batch file upload/download
   * - ``batch-upload.ts``
     - JSONL file creation and upload for OpenAI batch API
   * - ``batch-output.ts``
     - Output parsing for completed batch jobs
   * - ``batch-utils.ts``
     - Shared utilities (ID generation, status polling)
   * - ``batch-provider-common.ts``
     - Provider-agnostic batch logic
   * - ``batch-error-utils.ts``
     - Error classification (retryable vs. fatal)

**How batch embedding works:**

1. Chunks are grouped into batches by estimated byte size
   (``EMBEDDING_BATCH_MAX_TOKENS = 8000``).
2. Cached embeddings are loaded first; only uncached chunks are sent.
3. For OpenAI: JSONL batch files are uploaded to the batch API endpoint.
4. For Gemini: ``asyncBatchEmbedContent`` is called with content parts.
5. For Voyage: Standard batch embedding with polling.
6. Results are polled with configurable concurrency, interval, and timeout.
7. Completed embeddings are written to the embedding cache.


--------------------------------------------
Batch Failure Tracking
--------------------------------------------

The system automatically disables batch embedding after repeated failures
to prevent cascading errors:

.. code-block:: text

   BATCH_FAILURE_LIMIT = 2

- Each batch failure increments ``batchFailureCount``.
- When ``batchFailureCount >= BATCH_FAILURE_LIMIT`` (2), batch mode is
  disabled for the remainder of the session.
- The system falls back to sequential non-batch embedding.
- On a successful batch, the failure count resets to 0.
- Special case: ``asyncBatchEmbedContent not available`` immediately
  disables batch (``forceDisable = true``).
- A serialized lock (``batchFailureLock``) prevents race conditions when
  multiple batch operations complete concurrently.

**Status reporting:**

The ``batch`` field in ``MemoryProviderStatus`` reports:

.. code-block:: typescript

   batch: {
     enabled: boolean;     // current state
     failures: number;     // current failure count
     limit: number;        // 2 (the disable threshold)
     wait: boolean;        // wait for batch completion before search?
     concurrency: number;  // max concurrent batch polls
     pollIntervalMs: number;
     timeoutMs: number;
     lastError?: string;
     lastProvider?: string;
   }


--------------------------------------------
Embedding Cache
--------------------------------------------

The embedding cache (``embedding_cache`` table) stores computed embeddings
keyed by ``(provider, model, provider_key, hash)``:

- On index, the cache is checked first.  Cache hits skip the embedding API
  call entirely.
- On reindex, the cache is seeded from the old database into the new temp
  database via ``seedEmbeddingCache()``.
- Cache pruning occurs after reindex when the entry count exceeds
  ``maxEntries`` -- oldest entries (by ``updated_at``) are deleted.

This is particularly valuable for:

- **Provider switches:** When switching from OpenAI to Gemini, only the
  uncached chunks need re-embedding.
- **Full reindex:** The safe reindex (temp DB swap) preserves the cache
  so unchanged chunks are not re-embedded.
- **Workspace restores:** If the database is deleted but files are unchanged,
  the first sync re-embeds everything but the cache is rebuilt for future use.


--------------------------------------------
Retry and Timeout
--------------------------------------------

Source: ``src/memory/manager-embedding-ops.ts``

**Retry logic** (``embedBatchWithRetry``):

- Max attempts: 3 (``EMBEDDING_RETRY_MAX_ATTEMPTS``)
- Base delay: 500ms with exponential backoff and 20% jitter
- Max delay cap: 8000ms
- Retryable errors: ``rate_limit``, ``too many requests``, ``429``,
  ``resource has been exhausted``, ``5xx``, ``cloudflare``

**Timeout values:**

.. list-table::
   :header-rows: 1
   :widths: 40 20 20

   * - Operation
     - Local
     - Remote
   * - Query (single embedding)
     - 5 minutes
     - 60 seconds
   * - Batch (multiple embeddings)
     - 10 minutes
     - 2 minutes
