============================
Retrieval Pipeline
============================

This page walks through the full retrieval pipeline, from the moment the
agent calls ``memory_search`` to the final ranked results.  Each stage is
explained with its algorithm, configuration, and source file reference.

.. contents:: On this page
   :depth: 3
   :local:


--------------------------------------------
Pipeline Overview
--------------------------------------------

.. code-block:: text

   User Query
       |
       v
   [1] Query Expansion     -- extract keywords for FTS-only mode
       |
       v
   [2] Parallel Search     -- vector search + keyword search (concurrent)
       |           |
       v           v
   [3] Hybrid Merge        -- weighted combination (0.7 vector + 0.3 text)
       |
       v
   [4] Temporal Decay      -- recency scoring (30-day half-life, optional)
       |
       v
   [5] MMR Re-ranking      -- diversity via Jaccard similarity (optional)
       |
       v
   [6] Score Threshold     -- filter by minScore (default 0.35)
       |
       v
   [7] Top-K Selection     -- return maxResults (default 6)
       |
       v
   MemorySearchResult[]

**Default configuration values:**

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Parameter
     - Default
     - Source constant
   * - Chunk tokens
     - 400
     - ``DEFAULT_CHUNK_TOKENS``
   * - Chunk overlap
     - 80
     - ``DEFAULT_CHUNK_OVERLAP``
   * - Max results (top-K)
     - 6
     - ``DEFAULT_MAX_RESULTS``
   * - Min score threshold
     - 0.35
     - ``DEFAULT_MIN_SCORE``
   * - Hybrid enabled
     - ``true``
     - ``DEFAULT_HYBRID_ENABLED``
   * - Vector weight
     - 0.7
     - ``DEFAULT_HYBRID_VECTOR_WEIGHT``
   * - Text weight
     - 0.3
     - ``DEFAULT_HYBRID_TEXT_WEIGHT``
   * - Candidate multiplier
     - 4
     - ``DEFAULT_HYBRID_CANDIDATE_MULTIPLIER``
   * - MMR enabled
     - ``false``
     - ``DEFAULT_MMR_ENABLED``
   * - MMR lambda
     - 0.7
     - ``DEFAULT_MMR_LAMBDA``
   * - Temporal decay enabled
     - ``false``
     - ``DEFAULT_TEMPORAL_DECAY_ENABLED``
   * - Temporal decay half-life
     - 30 days
     - ``DEFAULT_TEMPORAL_DECAY_HALF_LIFE_DAYS``
   * - Snippet max chars
     - 700
     - ``SNIPPET_MAX_CHARS``


--------------------------------------------
Stage 1: Query Expansion
--------------------------------------------

Source: ``src/memory/query-expansion.ts``

Query expansion is primarily used in **FTS-only mode** (no embedding provider
available).  The ``extractKeywords()`` function strips stop words and extracts
meaningful terms from conversational queries.

**Supported languages:** English, Spanish, Portuguese, Arabic, Chinese,
Korean, Japanese.

**Algorithm:**

1. Tokenize the query by splitting on whitespace and punctuation.
2. For CJK text, extract character unigrams and bigrams.
3. For Korean, strip trailing particles (e.g. ``API`` from ``API``).
4. Filter out stop words across all supported languages.
5. Filter out tokens that are too short (<3 chars for English), numeric-only,
   or punctuation-only.
6. Deduplicate.

**Example:**

.. code-block:: text

   Input:  "that thing we discussed about the API"
   Output: ["discussed", "API"]

In FTS-only mode, each extracted keyword is searched separately and results
are merged, keeping the highest score for each chunk ID.

In hybrid mode, query expansion is not applied -- the raw query is used for
both vector embedding and FTS.


--------------------------------------------
Stage 2: Vector Search
--------------------------------------------

Source: ``src/memory/manager-search.ts`` (``searchVector``)

**With sqlite-vec available:**

The query is embedded into a vector, then searched against ``chunks_vec``
using native cosine distance:

.. code-block:: sql

   SELECT c.id, c.path, c.start_line, c.end_line, c.text, c.source,
          vec_distance_cosine(v.embedding, ?) AS dist
     FROM chunks_vec v
     JOIN chunks c ON c.id = v.id
    WHERE c.model = ?
    ORDER BY dist ASC
    LIMIT ?

The score is computed as ``1 - dist`` (cosine similarity from cosine
distance).

**Without sqlite-vec (JavaScript fallback):**

If the sqlite-vec extension cannot be loaded, all chunks are loaded from the
``chunks`` table, their JSON-encoded embeddings are parsed, and cosine
similarity is computed in JavaScript:

.. code-block:: typescript

   function cosineSimilarity(a: number[], b: number[]): number {
     let dot = 0, normA = 0, normB = 0;
     for (let i = 0; i < a.length; i++) {
       dot += a[i] * b[i];
       normA += a[i] * a[i];
       normB += b[i] * b[i];
     }
     return dot / (Math.sqrt(normA) * Math.sqrt(normB));
   }

This is significantly slower but correct.  It works because chunk counts in
personal memory workspaces are typically in the hundreds, not millions.


--------------------------------------------
Stage 2: Keyword Search (FTS5 BM25)
--------------------------------------------

Source: ``src/memory/manager-search.ts`` (``searchKeyword``),
``src/memory/hybrid.ts``

The raw query is transformed into an FTS5 query by ``buildFtsQuery()``:

.. code-block:: typescript

   // "hello world" -> '"hello" AND "world"'
   function buildFtsQuery(raw: string): string | null {
     const tokens = raw.match(/[\p{L}\p{N}_]+/gu)
       ?.map(t => t.trim()).filter(Boolean) ?? [];
     if (tokens.length === 0) return null;
     return tokens.map(t => `"${t}"`).join(" AND ");
   }

The FTS5 ``bm25()`` function returns a rank (lower is better).  This is
converted to a score in [0, 1] by ``bm25RankToScore()``:

.. code-block:: typescript

   function bm25RankToScore(rank: number): number {
     const normalized = Number.isFinite(rank) ? Math.max(0, rank) : 999;
     return 1 / (1 + normalized);
   }

In FTS-only mode (no provider), the model filter is removed so chunks from
any model are searchable.


--------------------------------------------
Stage 3: Hybrid Merge
--------------------------------------------

Source: ``src/memory/hybrid.ts`` (``mergeHybridResults``)

Vector and keyword results are merged by chunk ID.  Each chunk gets a
combined score:

.. code-block:: text

   score = vectorWeight * vectorScore + textWeight * textScore

**Default weights:** ``vectorWeight = 0.7``, ``textWeight = 0.3``.

Weights are normalized so they always sum to 1.0.  If a chunk appears in
only one result set, the missing score is 0.

The ``candidateMultiplier`` (default 4) controls how many candidates each
search arm fetches: ``Math.min(200, maxResults * candidateMultiplier)``.


--------------------------------------------
Stage 4: Temporal Decay
--------------------------------------------

Source: ``src/memory/temporal-decay.ts``

When enabled (``temporalDecay.enabled = true``), scores are multiplied by an
exponential decay factor based on the age of the source file:

.. code-block:: text

   decayedScore = score * exp(-lambda * ageInDays)

   where lambda = ln(2) / halfLifeDays

**Default half-life:** 30 days (score halves every 30 days).

**Timestamp extraction priority:**

1. **Date in filename** -- ``memory/2025-11-27.md`` -> Nov 27, 2025.
2. **Evergreen check** -- ``MEMORY.md``, ``memory.md``, or non-dated files
   in ``memory/`` are immune to decay (return ``null`` timestamp).
3. **File mtime** -- If no date is in the path and the file is not evergreen,
   use the filesystem modification time.
4. **No timestamp** -- If no timestamp can be extracted, the chunk is not
   decayed (score unchanged).

This ensures that evergreen knowledge (``MEMORY.md``) always has full weight,
while episodic daily notes naturally fade unless they remain relevant via
high vector/keyword scores.


--------------------------------------------
Stage 5: MMR Re-ranking
--------------------------------------------

Source: ``src/memory/mmr.ts``

Maximal Marginal Relevance (MMR) is an optional re-ranking step that balances
relevance with diversity.  When enabled (``mmr.enabled = true``):

**Algorithm** (Carbonell & Goldstein, 1998):

1. Normalize all scores to [0, 1].
2. Start with the highest-scoring item.
3. For each remaining slot, select the candidate that maximizes:

   .. code-block:: text

      MMR = lambda * relevance - (1 - lambda) * max_similarity_to_selected

4. Similarity is computed via **Jaccard similarity** on lowercased
   alphanumeric token sets:

   .. code-block:: typescript

      function jaccardSimilarity(setA: Set<string>, setB: Set<string>): number {
        const intersectionSize = [...smaller].filter(t => larger.has(t)).length;
        const unionSize = setA.size + setB.size - intersectionSize;
        return unionSize === 0 ? 0 : intersectionSize / unionSize;
      }

**Default lambda:** 0.7 (biased toward relevance; set lower for more
diversity).

Token sets are pre-computed and cached per item for efficiency.


--------------------------------------------
Stage 6: Score Threshold
--------------------------------------------

After all scoring adjustments, results with ``score < minScore`` are
filtered out.

**Default:** ``minScore = 0.35``.

This can be overridden per-call via the ``minScore`` parameter on
``memory_search``.


--------------------------------------------
Stage 7: Top-K Selection
--------------------------------------------

The final results are sliced to ``maxResults`` (default 6).  This can be
overridden per-call via the ``maxResults`` parameter.


--------------------------------------------
FTS-Only Mode
--------------------------------------------

When no embedding provider is available (all API keys missing, local
embeddings not installed), the search pipeline simplifies to:

.. code-block:: text

   Query
     |
     v
   extractKeywords() -- stop-word removal + tokenization
     |
     v
   searchKeyword() per keyword -- FTS5 BM25 matching
     |
     v
   Merge results -- deduplicate by chunk ID, keep highest score
     |
     v
   Filter by minScore
     |
     v
   Slice to maxResults
     |
     v
   MemorySearchResult[]

This mode ensures the knowledge system remains functional even without any
embedding infrastructure -- just SQLite with FTS5 is sufficient for basic
keyword recall.


--------------------------------------------
Complete Search Flow (Code Walkthrough)
--------------------------------------------

The ``search()`` method in ``MemoryIndexManager`` (``src/memory/manager.ts``)
orchestrates the full pipeline:

.. code-block:: typescript

   async search(query, opts?) {
     // Warm session (trigger sync on first search in a session)
     void this.warmSession(opts?.sessionKey);

     // Sync if dirty
     if (this.settings.sync.onSearch && (this.dirty || this.sessionsDirty)) {
       void this.sync({ reason: "search" });
     }

     const minScore = opts?.minScore ?? this.settings.query.minScore;
     const maxResults = opts?.maxResults ?? this.settings.query.maxResults;
     const candidates = Math.min(200, maxResults * candidateMultiplier);

     // FTS-only mode
     if (!this.provider) {
       const keywords = extractKeywords(query);
       // Search each keyword, merge, sort, filter, slice
       return merged.filter(e => e.score >= minScore).slice(0, maxResults);
     }

     // Hybrid mode
     const keywordResults = hybrid.enabled
       ? await this.searchKeyword(query, candidates)
       : [];
     const queryVec = await this.embedQueryWithTimeout(query);
     const vectorResults = queryVec.some(v => v !== 0)
       ? await this.searchVector(queryVec, candidates)
       : [];

     if (!hybrid.enabled) {
       return vectorResults.filter(e => e.score >= minScore).slice(0, maxResults);
     }

     const merged = await this.mergeHybridResults({
       vector: vectorResults,
       keyword: keywordResults,
       vectorWeight, textWeight,
       mmr, temporalDecay,
     });

     return merged.filter(e => e.score >= minScore).slice(0, maxResults);
   }
