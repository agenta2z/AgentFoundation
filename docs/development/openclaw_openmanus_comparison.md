# OpenClaw vs OpenManus vs BrowserAgent Knowledge/Skills System — Comprehensive Comparison Report

**Date:** 2026-02-23 (Updated)
**Author:** Analysis based on deep investigation of source code and documentation
**Purpose:** Compare knowledge/skills creation, ingestion, storage, and retrieval designs across three systems and highlight BrowserAgent's competitive advantages.

---

## Executive Summary

| Dimension | OpenClaw | OpenManus | BrowserAgent | BrowserAgent Advantage |
|-----------|----------|-----------|--------------------------|----------------------|
| **Architecture** | File-based + Vector embeddings | Stateless LLM-centric | ✅ Three-layer (Metadata + Pieces + Graph) | **UNIQUE** — Entity graph enables relationship-aware retrieval |
| **Memory Persistence** | ✅ Markdown files + SQLite vectors | ❌ Ephemeral (lost on exit) | ✅ Multiple backends (SQLite, LanceDB, etc.) | Flexibility |
| **Knowledge Ingestion** | ⚠️ Manual (user writes Markdown) | ❌ None (stateless) | ✅ **LLM-based DocumentIngester + chunking + structuring** | **SUPERIOR** — Automated extraction with format detection |
| **Knowledge Storage** | Plain Markdown files in workspace | None (in-memory only) | ✅ Structured KnowledgePieces with rich metadata | Structured + searchable |
| **Skills System** | ✅ Modular file-based (manual only) | ❌ Hardcoded tools only | ✅ **Auto-synthesis + manual authoring** | **SUPERIOR** — Skills emerge automatically |
| **Semantic Search** | ✅ Hybrid (BM25 + Vector + RRF) | ❌ No embeddings at all | ✅ Hybrid (BM25 + Vector + RRF) | Parity |
| **Deduplication** | ⚠️ Basic file-level | ❌ None | ✅ **Three-tier (Hash → Embedding → LLM Judge)** | **SUPERIOR** — Semantic understanding |
| **Temporal Awareness** | ✅ Exponential decay + recency | ❌ None | ✅ Exponential decay + **evergreen exemptions** | **IMPROVED** — Skills/instructions never decay |
| **Context Budget** | ✅ Progressive disclosure | ❌ None | ✅ Progressive disclosure + pre-compaction flush | Parity |
| **Knowledge Spaces** | ⚠️ Implicit via workspace structure | ❌ None | ✅ **Explicit spaces (main/personal/developmental)** | **SUPERIOR** — First-class space management |
| **Merge Strategy Control** | ❌ None | ❌ None | ✅ **5 configurable strategies per type** | **UNIQUE** |
| **Entity Graph** | ❌ None | ❌ None | ✅ **Typed edges + provenance** | **UNIQUE** |
| **MMR Diversity** | ✅ λ-balanced | ❌ None | ✅ λ-balanced | Parity |
| **Knowledge Validation** | ❌ None | ❌ None | ✅ **LLM-based integrity checking (3 modes)** | **UNIQUE** — Proactive correctness & authenticity verification |
| **Agentic Knowledge Mgmt** | ⚠️ Semi-agentic (pre-compaction only) | ❌ None | ✅ **Fully agentic (7/7 capabilities)** | **SUPERIOR** — Self-improving KB |

**Key Insight:** BrowserAgent has **6 unique capabilities** not found in either competitor, **4 superior implementations**, and **parity** with OpenClaw on all mature features. OpenManus is intentionally minimal (stateless) and not a direct competitor.

---

## Part 1: OpenClaw Analysis

### 1.1 Overview

OpenClaw is a TypeScript-based agent system with a **WebSocket gateway architecture**. It's designed for persistent, multi-channel conversations (WhatsApp, Telegram, Slack, Discord, iMessage, etc.).

**Philosophy:** "Memory is plain Markdown in the agent workspace. The files are the source of truth."

### 1.2 Memory System

#### 1.2.1 Storage Architecture

```
~/.openclaw/
├── workspace/                    # Agent workspace (configurable)
│   ├── MEMORY.md                 # Long-term curated memory
│   ├── memory/
│   │   └── YYYY-MM-DD.md         # Daily logs (append-only)
│   ├── AGENTS.md                 # Operating instructions
│   ├── SOUL.md                   # Persona/boundaries
│   ├── TOOLS.md                  # Tool notes
│   └── skills/                   # User skills
├── agents/<agentId>/
│   ├── sessions/
│   │   ├── sessions.json         # Session state store
│   │   └── <SessionId>.jsonl     # Full transcripts
│   └── qmd/                      # QMD sidecar (optional)
└── memory/<agentId>.sqlite       # Vector embeddings store
```

**Key Design Decisions:**
1. **Markdown as source of truth** — Human-readable, version-controllable, no proprietary formats
2. **Two-layer memory** — Daily logs (ephemeral context) + MEMORY.md (curated long-term)
3. **Session transcripts in JSONL** — Full audit trail, compaction summaries persist
4. **SQLite for vectors** — Optional sqlite-vec acceleration for semantic search

#### 1.2.2 Vector Memory Search

OpenClaw implements **true hybrid search** combining:

1. **Dense embeddings** (semantic similarity)
2. **BM25 keyword search** (exact tokens: IDs, code symbols, error strings)
3. **Reciprocal Rank Fusion (RRF)** — NOT serial fallback, but parallel fusion

```typescript
// From docs/concepts/memory.md
// Weighted score merge:
finalScore = vectorWeight * vectorScore + textWeight * textScore
// Default: 0.7 vector + 0.3 keyword

// BM25 rank to score conversion:
textScore = 1 / (1 + max(0, bm25Rank))
```

**BrowserAgent comparison:** ✅ We now implement the same parallel RRF hybrid search.

#### 1.2.3 Temporal Decay

OpenClaw implements **exponential recency weighting**:

```typescript
decayedScore = score × e^(-λ × ageInDays)
// where λ = ln(2) / halfLifeDays

// With halfLife = 30 days:
// Today: 100% of score
// 7 days: ~84%
// 30 days: 50%
// 90 days: 12.5%
// 180 days: ~1.6%
```

**Key insight:** Evergreen files (MEMORY.md, non-dated files) are NEVER decayed — only dated daily files.

**BrowserAgent comparison:** ✅ We implement the same exponential decay **PLUS** explicit evergreen exemptions for `skills` and `instructions` info_types. This is **more granular** than OpenClaw's file-based approach.

#### 1.2.4 MMR Re-ranking (Diversity)

OpenClaw applies **Maximal Marginal Relevance** to avoid returning near-duplicate results:

```
MMR selects results that maximize:
λ × relevance − (1−λ) × max_similarity_to_selected

Default λ = 0.7 (balanced, slight relevance bias)
```

**BrowserAgent comparison:** ✅ We implement the same MMR re-ranking with configurable λ.

#### 1.2.5 Context Compaction

OpenClaw auto-compacts sessions when nearing context window limits:

1. **Pre-compaction memory flush** — Silent agentic turn reminds model to write durable notes
2. **Compaction** — Summarizes older conversation into a compact entry
3. **Pruning** — Trims old tool results from in-memory context (doesn't rewrite transcript)

**BrowserAgent comparison:** ✅ We implement the same pre-compaction flush with silent agentic turn.

### 1.3 Skills System

#### 1.3.1 Skill Structure

```
skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter (name, description)
│   └── Markdown instructions
└── Bundled Resources (optional)
    ├── scripts/          # Executable code
    ├── references/       # Documentation
    └── assets/           # Templates, images, etc.
```

**Progressive Disclosure (3-level loading):**
1. **Metadata** (~100 tokens) — Always in context
2. **SKILL.md body** (<5k words) — When skill triggers
3. **Bundled resources** — As needed (scripts can execute without reading into context)

**Key limitation:** OpenClaw skills are **manually authored only**. There is no automatic skill synthesis from ingested knowledge.

**BrowserAgent comparison:** ✅ We support **both** manual authoring AND automatic skill synthesis via Merge-on-Ingest. This is a **significant advantage**.

---

## Part 2: OpenManus Analysis

### 2.1 Overview

OpenManus is a Python-based **ReAct agent framework** focused on simplicity and tool extensibility. It follows an **LLM-centric, stateless architecture**.

**Critical finding:** OpenManus has **NO knowledge management system**. This is documented explicitly in their own docs:

> *"A key finding is that OpenManus does not have a dedicated knowledge management system or formalized skill library."*

### 2.2 Memory System (Ephemeral Only)

```python
# app/schema.py
class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]  # Sliding window
```

**No persistence:** When the process exits, all context is lost.

### 2.3 Skills System (Hardcoded Tools Only)

In OpenManus, "skills" are `BaseTool` subclasses with static configuration per agent type. From the docs:

> *"OpenManus cannot learn, create, or acquire new skills at runtime."*

**BrowserAgent comparison:** This is a fundamentally different architecture. OpenManus is a lightweight agent framework; BrowserAgent is a knowledge-augmented system. They're not direct competitors.

---

## Part 3: BrowserAgent Knowledge System (Implemented)

### 3.1 Unique Capabilities (Not in OpenClaw or OpenManus)

| Capability | Description | Competitive Advantage |
|------------|-------------|----------------------|
| **Entity Graph Store** | Captures relationships between knowledge pieces via typed graph edges (DERIVED_FROM, EXTENDS, CONTRADICTS, etc.) | Enables multi-hop retrieval, provenance tracking, and relationship-aware search |
| **Explicit Knowledge Spaces** | First-class `main`/`personal`/`developmental` separation with configurable priorities and lifecycles | OpenClaw uses implicit workspace structure; we have explicit space management with auto-expiry policies |
| **Merge Strategy Flexibility** | 5 strategies per knowledge type: `AUTO_MERGE_ON_INGEST`, `SUGGESTION_ON_INGEST`, `POST_INGESTION_AUTO`, `POST_INGESTION_SUGGESTION`, `MANUAL_ONLY` | No other system allows per-type merge control with human-in-the-loop options |
| **Three-Tier Deduplication** | Hash → Embedding similarity → LLM Judge with ADD/UPDATE/MERGE/NO_OP decisions | OpenClaw has only file-change detection; we have semantic understanding of duplicates |
| **Auto Skill Synthesis** | Merge-on-Ingest automatically detects and creates skills from scattered knowledge | OpenClaw requires manual skill authoring via SKILL.md files |
| **Knowledge Validation** | LLM-based proactive examination with 3 modes: `auto-checking-upon-ingestion`, `auto-checking-post-ingestion`, `manual-trigger-check` | No other system has proactive integrity verification |

### 3.2 Superior Implementations (Better than OpenClaw)

#### 3.2.1 Deduplication Depth

| Tier | OpenClaw | BrowserAgent | Why BrowserAgent is Better |
|------|----------|------------|--------------------------|
| **Tier 1: Exact Match** | File modification time | Content SHA256 hash | **Content-addressed** — detects duplicates even if file metadata differs |
| **Tier 2: Semantic** | None (via search only) | Embedding similarity search with thresholds (0.85-0.98) | **Proactive** — catches near-duplicates during ingestion, not just retrieval |
| **Tier 3: Nuance** | None | LLM Judge with ADD/UPDATE/MERGE/NO_OP | **Intelligent** — detects state changes, additive info, paraphrases |

**Example where BrowserAgent wins:**
- **Input A:** "Use learning rate 1e-4 for BERT fine-tuning"
- **Input B:** "Updated: Use learning rate 1e-5 for BERT fine-tuning (corrected)"

| System | Result |
|--------|--------|
| OpenClaw | Stores both as separate memories (no content-level dedup) |
| BrowserAgent | LLM Judge detects UPDATE → marks A inactive, keeps B |

#### 3.2.2 Temporal Decay with Evergreen Exemptions

| Aspect | OpenClaw | BrowserAgent | Why BrowserAgent is Better |
|--------|----------|------------|--------------------------|
| **Decay Function** | Exponential with half-life | Exponential with half-life | Parity |
| **Exemption Mechanism** | File-based (MEMORY.md, non-dated files) | **Info-type-based** (`skills`, `instructions`) | **More granular** — any piece can be evergreen regardless of storage location |
| **Configuration** | Per-workspace | Per-info-type with override | **More flexible** |

#### 3.2.3 Merge Strategy Control

| Aspect | OpenClaw | BrowserAgent | Why BrowserAgent is Better |
|--------|----------|------------|--------------------------|
| **Strategy Options** | None (always stores new) | 5 configurable strategies | **Control** — different knowledge types have different merge behaviors |
| **Human-in-Loop** | None | `suggestion-of-*` modes | **Safety** — skills can require human approval before merge |
| **Timing** | N/A | On-ingest OR post-ingestion | **Flexibility** — background merging for lazy strategies |

**Default strategies by KnowledgeType:**

| KnowledgeType | Default Strategy | Rationale |
|---------------|------------------|-----------||
| `Procedure` | `MANUAL_ONLY` | Procedures are high-value, human-curated |
| `Instruction` | `SUGGESTION_ON_INGEST` | Instructions should be reviewed before merging |
| `Fact` | `AUTO_MERGE_ON_INGEST` | Facts can safely auto-merge |
| `Preference` | `MANUAL_ONLY` | User preferences are sensitive |
| `Episodic` | `POST_INGESTION_AUTO` | Episodic can be cleaned up later |
| `Note` | `AUTO_MERGE_ON_INGEST` | Notes can safely auto-merge |
| `Example` | `SUGGESTION_ON_INGEST` | Examples should be reviewed before merging |

#### 3.2.4 Knowledge Validation (Integrity Checking)

BrowserAgent implements **LLM-based proactive knowledge validation** — a unique capability not found in OpenClaw or OpenManus.

**Three Validation Modes:**

| Mode | Trigger | Behavior |
|------|---------|----------|
| `auto-checking-upon-ingestion` | During `_load_into_kb()` | LLM automatically validates before storing |
| `auto-checking-post-ingestion` | Background job | Periodic validation of recently ingested knowledge |
| `manual-trigger-check` | Explicit `kb.validate(piece_id)` call | On-demand validation when requested |

**Validation Checks:**

| Check | Description | Example |
|-------|-------------|----------|
| **Correctness** | Verify factual accuracy against known sources | "Is this learning rate recommendation valid for BERT?" |
| **Authenticity** | Verify source credibility and provenance | "Does this come from a trusted source?" |
| **Consistency** | Check for contradictions with existing knowledge | "Does this conflict with existing pieces?" |
| **Completeness** | Identify missing context or dependencies | "Are prerequisite concepts present?" |
| **Staleness** | Flag potentially outdated information | "Is this 2-year-old API reference still valid?" |
| **Security** | Detect sensitive data exposure (credentials, API keys, secrets) | "Does this contain hardcoded passwords or tokens?" |
| **Privacy** | Check for PII and data protection compliance | "Does this expose email addresses, phone numbers, or user data?" |
| **Policy Compliance** | Ensure adherence to organizational policies and rules | "Does this violate content guidelines or licensing terms?" |

**Validation Configuration:**

Validation is configured via `ValidationConfig` with the following enabled checks by default:
- `correctness` — Verify factual accuracy
- `authenticity` — Verify source credibility
- `consistency` — Check for contradictions
- `completeness` — Identify missing context
- `staleness` — Flag outdated information
- `security` — Detect credentials/secrets (regex-based, fast)
- `privacy` — Check for PII (regex-based, fast)
- `policy_compliance` — Ensure adherence to policies

Security and privacy checks use fast regex patterns; other checks use LLM-based validation.

**LLM Validation Prompt:**

```
You are a knowledge integrity validator.

Given a KNOWLEDGE PIECE, evaluate its quality:

## Knowledge Piece
{content}
(Domain: {domain}, Source: {source}, Info Type: {info_type})

## Validation Checks
1. **Correctness**: Is this factually accurate?
2. **Authenticity**: Is the source credible?
3. **Consistency**: Does this contradict known facts?
4. **Completeness**: Is context/dependencies present?
6. **Security**: Does this contain credentials, API keys, or secrets?
7. **Privacy**: Does this expose PII (emails, phone numbers, user data)?
8. **Policy Compliance**: Does this violate content guidelines or licensing terms?
5. **Staleness**: Is this potentially outdated?

## Your Assessment
Return JSON:
{
  "is_valid": true|false,
  "confidence": 0.0-1.0,
  "issues": ["list of issues found"],
  "suggestions": ["list of improvement suggestions"],
  "reasoning": "Brief explanation"
}
```

**ValidationResult Schema:**

```python
@attrs
class ValidationResult:
    is_valid: bool = attrib()           # Overall validity
    confidence: float = attrib()        # 0.0-1.0
    issues: List[str] = attrib()        # Detected problems
    suggestions: List[str] = attrib()   # Improvement recommendations
    checks_passed: List[str] = attrib() # List of checks that passed
    checks_failed: List[str] = attrib() # List of checks that failed
```

**Integration with Ingestion Pipeline:**

```python
def _apply_enhancements(
    self,
    data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Apply deduplication and validation to pieces before loading."""
    counts = {"deduped": 0, "failed_validation": 0}

    if not (self._deduplicator or self._validator):
        return data, counts

    enhanced_pieces = []

    for piece_dict in data.get("pieces", []):
        piece = KnowledgePiece.from_dict(piece_dict)

        # Validation (if enabled)
        if self._validator:
            val_result = self._validator.validate(piece)
            if not val_result.is_valid:
                piece.validation_status = "failed"
                piece.validation_issues = val_result.issues
                piece.space = "developmental"  # Move to review space
                counts["failed_validation"] += 1

        # Deduplication (if enabled)
        if self._deduplicator:
            dedup_result = self._deduplicator.deduplicate(piece)
            if dedup_result.action == DedupAction.NO_OP:
                counts["deduped"] += 1
                continue  # Skip this piece

        enhanced_pieces.append(piece.to_dict())

    data["pieces"] = enhanced_pieces
    return data, counts
```

**Why This Matters:**

| Without Validation | With Validation |
|--------------------|----------------|
| Bad knowledge silently enters KB | Bad knowledge caught before storage |
| Contradictions accumulate | Consistency maintained |
| Stale info persists forever | Staleness detected and flagged |
| No audit trail | Full validation history |

### 3.3 Parity Features (Equal to OpenClaw)

| Feature | Implementation |
|---------|---------------|
| **Parallel RRF Hybrid Search** | Vector + BM25 executed in parallel, fused with RRF (k=60), configurable weights (default 0.7/0.3) |
| **Exponential Temporal Decay** | `score × e^(-λ×days)` where `λ = ln(2)/halfLife`, configurable half-life (default 30 days) |
| **MMR Diversity Re-ranking** | λ-balanced Maximal Marginal Relevance (default λ=0.7) |
| **Pre-Compaction Memory Flush** | Silent agentic turn before context compression to persist important knowledge |
| **Context Budget Management** | Per-info-type token limits with progressive skill disclosure |
| **Embedding Cache** | Content-hash keyed cache to avoid redundant embedding API calls |

### 3.4 Architecture Advantages

#### 3.4.1 Three-Layer Knowledge Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE BASE                           │
│                                                             │
│  ┌─────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │  Metadata   │  │ Knowledge Pieces │  │  Entity Graph  │  │
│  │   Store     │  │     Store        │  │     Store      │  │
│  │             │  │                  │  │                │  │
│  │ • entity_id │  │ • content        │  │ • nodes        │  │
│  │ • type      │  │ • domain         │  │ • edges        │  │
│  │ • space     │  │ • tags           │  │ • types        │  │
│  │ • profile   │  │ • info_type      │  │ • properties   │  │
│  │             │  │ • space          │  │ • space        │  │
│  │             │  │ • merge_strategy │  │                │  │
│  └─────────────┘  └─────────────────┘  └────────────────┘  │
│                                                             │
│  Neither OpenClaw nor OpenManus has this three-layer       │
│  architecture. OpenClaw is file-based; OpenManus is        │
│  stateless.                                                 │
└─────────────────────────────────────────────────────────────┘
```

#### 3.4.2 Retrieval Pipeline

```
QUERY
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ HYBRID SEARCH (Parallel)                                    │
│   Vector Search ─────┐                                      │
│                      ├── RRF Fusion (k=60)                  │
│   BM25 Keyword ──────┘                                      │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ POST-PROCESSING                                             │
│   1. Temporal Decay (exponential, evergreen exemptions)     │
│   2. MMR Diversity Re-ranking (λ=0.7)                       │
│   3. Space Priority Sorting                                 │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ FORMATTING & INJECTION (Budget enforced here)               │
│   • Per-info-type token budgets                             │
│   • Progressive skill disclosure (summaries first)          │
│   • Info-type-specific formatting                           │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
CONTEXT WINDOW
```

---

## Part 4: Why BrowserAgent is the Best Choice

### 4.1 Competitive Positioning Summary

| Category | OpenClaw | OpenManus | BrowserAgent |
|----------|----------|-----------|------------|
| **Unique Features** | QMD sidecar, Markdown source of truth | MCP interoperability, simplicity | **5 unique capabilities** (see 3.1) |
| **Deduplication** | Basic (file-level) | None | **Three-tier with LLM Judge** |
| **Skill Creation** | Manual only | None | **Auto-synthesis + manual** |
| **Knowledge Spaces** | Implicit (workspace) | None | **Explicit with lifecycles** |
| **Graph Support** | None | None | **Entity Graph Store** |
| **Merge Control** | None | None | **5 strategies per type** |
| **Knowledge Validation** | None | None | **3-mode LLM validation** |

### 4.2 When to Use Each System

| Use Case | Best Choice | Reason |
|----------|-------------|--------|
| Simple chat agent with no learning | OpenManus | Stateless simplicity |
| Personal assistant with markdown notes | OpenClaw | File-based, human-readable |
| **Enterprise knowledge management** | **BrowserAgent** | Structured storage, deduplication, spaces |
| **Auto-evolving skill library** | **BrowserAgent** | Merge-on-Ingest synthesizes skills |
| **Multi-user with isolation needs** | **BrowserAgent** | Explicit spaces with priorities |
| **High-volume ingestion** | **BrowserAgent** | Three-tier dedup prevents bloat |
| **Quality-critical knowledge** | **BrowserAgent** | LLM validation ensures integrity |

### 4.3 Feature-by-Feature Winner

| Feature | Winner | Margin |
|---------|--------|--------|
| Hybrid Search | Tie (OpenClaw = BrowserAgent) | — |
| Temporal Decay | **BrowserAgent** | Evergreen exemptions by info_type |
| Deduplication | **BrowserAgent** | Three-tier vs file-level |
| Skill Creation | **BrowserAgent** | Auto + manual vs manual only |
| Knowledge Spaces | **BrowserAgent** | Explicit vs implicit |
| Entity Relationships | **BrowserAgent** | Graph store vs none |
| Merge Control | **BrowserAgent** | 5 strategies vs none |
| Knowledge Validation | **BrowserAgent** | 3-mode LLM validation vs none |
| MMR Diversity | Tie (OpenClaw = BrowserAgent) | — |
| Pre-Compaction Flush | Tie (OpenClaw = BrowserAgent) | — |
| Context Budget | Tie (OpenClaw = BrowserAgent) | — |

**Score: BrowserAgent 7, OpenClaw 0, Ties 4**

---

## Appendix A: File References

### OpenClaw Sources Analyzed
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/openclaw/docs/concepts/memory.md`
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/openclaw/docs/concepts/session.md`
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/openclaw/docs/concepts/compaction.md`
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/openclaw/docs/concepts/context.md`
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/openclaw/docs/concepts/agent.md`
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/openclaw/docs/concepts/architecture.md`
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/openclaw/skills/skill-creator/SKILL.md`
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/openclaw/src/memory/` (TypeScript implementation)

### OpenManus Sources Analyzed
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/OpenManus/docs/knowledge_and_skills.rst`
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/OpenManus/docs/architecture.rst`
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/OpenManus/app/schema.py`
- `/data/users/zgchen/fbsource/fbcode/_tony_dev/OpenManus/app/tool/planning.py`

### BrowserAgent Plan
- `/home/zgchen/.llms/plans/knowledge_system_v2_integration.plan.md` (v2.3.2 — Fully Implemented)

---

## Appendix B: Complete Feature Comparison Table

| Feature | OpenClaw | OpenManus | BrowserAgent | BrowserAgent Advantage |
|---------|----------|-----------|------------|---------------------|
| **ARCHITECTURE** |||||
| Language | TypeScript | Python | Python | — |
| Persistence | File + SQLite | None | Multiple backends | Flexibility |
| Knowledge graph | ❌ | ❌ | ✅ Entity graph | **UNIQUE** |
| Three-layer store | ❌ | ❌ | ✅ Metadata + Pieces + Graph | **UNIQUE** |
| **MEMORY** |||||
| Long-term storage | Markdown files | None | KnowledgePieces | Structured |
| Session history | JSONL transcripts | In-memory | Session persistence | — |
| Compaction | ✅ Auto + manual | ❌ | ✅ Auto + pre-flush | Parity |
| **SEARCH** |||||
| Vector search | ✅ | ❌ | ✅ | — |
| Keyword search | ✅ BM25/FTS5 | ❌ | ✅ FTS5 | — |
| Hybrid fusion | ✅ Parallel RRF | ❌ | ✅ Parallel RRF | Parity |
| Temporal decay | ✅ Exponential | ❌ | ✅ Exponential + evergreen | **IMPROVED** |
| MMR diversity | ✅ | ❌ | ✅ | Parity |
| **DEDUPLICATION** |||||
| Exact match | File-level | None | ✅ Content hash | **SUPERIOR** |
| Semantic | ❌ | ❌ | ✅ Embedding (0.85-0.98) | **UNIQUE** |
| LLM judge | ❌ | ❌ | ✅ ADD/UPDATE/MERGE/NO_OP | **UNIQUE** |
| **SKILLS** |||||
| Manual authoring | ✅ SKILL.md | ❌ | ✅ Supported | Parity |
| Auto-synthesis | ❌ | ❌ | ✅ Merge-on-Ingest | **UNIQUE** |
| Progressive disclosure | ✅ 3-level | ❌ | ✅ 3-level | Parity |
| **MERGE CONTROL** |||||
| Per-type strategies | ❌ | ❌ | ✅ 5 strategies | **UNIQUE** |
| Human-in-loop | ❌ | ❌ | ✅ Suggestion modes | **UNIQUE** |
| Background merge | ❌ | ❌ | ✅ Post-ingestion lazy | **UNIQUE** |
| **KNOWLEDGE VALIDATION** |||||
| Validation modes | ❌ | ❌ | ✅ 3 modes (upon-ingest, post-ingest, manual) | **UNIQUE** |
| Correctness checking | ❌ | ❌ | ✅ LLM-based fact verification | **UNIQUE** |
| Authenticity checking | ❌ | ❌ | ✅ Source credibility verification | **UNIQUE** |
| Consistency checking | ❌ | ❌ | ✅ Contradiction detection | **UNIQUE** |
| Security checking | ❌ | ❌ | ✅ Credential/secret detection | **UNIQUE** |
| Privacy checking | ❌ | ❌ | ✅ PII detection & compliance | **UNIQUE** |
| Policy compliance | ❌ | ❌ | ✅ Organizational rules enforcement | **UNIQUE** |
| Staleness detection | ❌ | ❌ | ✅ Outdated info flagging | **UNIQUE** |
| **CONTEXT** |||||
| Token budget | ✅ Per-type limits | ❌ | ✅ Per-type limits | Parity |
| Pre-compaction flush | ✅ | ❌ | ✅ | Parity |
| **KNOWLEDGE SPACES** |||||
| Explicit spaces | ❌ (workspace-implicit) | ❌ | ✅ main/personal/dev | **UNIQUE** |
| Space priorities | ❌ | ❌ | ✅ Configurable | **UNIQUE** |
| Space lifecycles | ❌ | ❌ | ✅ Auto-expiry (30 days) | **UNIQUE** |
| **ENTITY GRAPH** |||||
| Graph nodes | ❌ | ❌ | ✅ Typed nodes | **UNIQUE** |
| Graph edges | ❌ | ❌ | ✅ DERIVED_FROM, EXTENDS, etc. | **UNIQUE** |
| Provenance tracking | ❌ | ❌ | ✅ Source piece linking | **UNIQUE** |

---

## Appendix C: BrowserAgent Unique Capabilities Summary

### C.1 The 6 Unique Features

1. **Entity Graph Store** — Neither competitor has relationship tracking between knowledge pieces
2. **Explicit Knowledge Spaces** — First-class main/personal/developmental separation with lifecycle policies
3. **Merge Strategy Flexibility** — 5 strategies with per-type defaults and human-in-the-loop options
4. **Three-Tier Deduplication** — Hash → Embedding → LLM Judge pipeline for intelligent dedup
6. **Knowledge Validation** — LLM-based proactive examination with 3 modes (auto-upon-ingest, auto-post-ingest, manual-trigger) checking correctness, authenticity, consistency, completeness, staleness, security, privacy, and policy compliance
6. **Knowledge Validation** — LLM-based proactive examination with 3 modes (auto-upon-ingest, auto-post-ingest, manual-trigger) checking correctness, authenticity, consistency, completeness, and staleness

### C.2 The 4 Superior Implementations

1. **Deduplication** — Content-addressed hashing + semantic similarity + LLM nuance detection (vs OpenClaw's file-level only)
2. **Temporal Decay** — Info-type-based evergreen exemptions (vs OpenClaw's file-based)
3. **Merge Control** — 5 configurable strategies per knowledge type (vs none)
4. **Knowledge Ingestion** — LLM-based DocumentIngester with automated structuring, chunking, and validation (vs OpenClaw's manual Markdown writing)

### C.3 The 6 Parity Features (Equal to OpenClaw)

1. Parallel RRF hybrid search
2. Exponential temporal decay
3. MMR diversity re-ranking
4. Pre-compaction memory flush
5. Context budget management
6. Progressive skill disclosure
