# ClawHub Skill Registry — Compatibility Analysis with AgentFoundation Knowledge Module

**Date:** 2026-02-27
**Analyst:** Code audit of `agent_foundation.knowledge` (Python, ~50 files) vs `clawhub` (TypeScript, Convex+React)
**Purpose:** Determine compatibility for importing ClawHub skills into AgentFoundation's knowledge system

---

## 1. Executive Summary

ClawHub and AgentFoundation's knowledge module are **fundamentally different systems** that operate in **complementary, non-overlapping domains**. They are not directly compatible, but integration is feasible through a well-designed adapter layer.

```yaml
compatibility_verdict: "Architecturally incompatible but practically integrable"
effort_estimate: "Medium — requires adapter, no core rewrites needed"
risk_level: "Low — both systems have clean extension points"
```

### Key Findings

| Dimension | ClawHub | AgentFoundation Knowledge |
|-----------|---------|--------------------------|
| **Language** | TypeScript (Node/Bun) | Python |
| **Skill Format** | `SKILL.md` (Markdown + YAML frontmatter) | `KnowledgePiece` (attrs dataclass, JSON-serializable) |
| **Storage** | Convex (cloud DB + file storage) | Pluggable stores (LanceDB, in-memory, adapters) |
| **Search** | OpenAI `text-embedding-3-small` (1536d) via Convex vector index | Pluggable (BM25, vector, hybrid via `HybridRetriever`) |
| **Skill Model** | Markdown document with metadata frontmatter | Procedure-type `KnowledgePiece` with `info_type="skills"` |
| **Versioning** | Semver versions with tags (`latest`) | `version` integer field + `supersedes` chain |
| **Distribution** | Registry API (`clawhub.ai`) with CLI (`clawhub install`) | Local store, no registry |
| **Security** | LLM analysis + VirusTotal scanning + moderation | Content validation (sensitive patterns, privacy checks) |

---

## 2. Detailed Architecture Comparison

### 2.1 What is a "Skill" in Each System?

#### ClawHub Skill

A ClawHub skill is a **folder** containing:

```
my-skill/
├── SKILL.md          # Required — Markdown with YAML frontmatter
├── helpers.py        # Optional — supporting text files
├── config.yaml       # Optional
└── .clawhub/         # CLI metadata (auto-generated)
    └── origin.json
```

The `SKILL.md` frontmatter defines the skill's identity and requirements:

```yaml
---
name: my-skill
description: Short summary of what this skill does.
version: 1.0.0
metadata:
  openclaw:
    requires:
      env: [MY_API_KEY]          # Required environment variables
      bins: [curl]               # Required CLI binaries
      anyBins: [jq, yq]          # At least one required
      config: [~/.myrc]          # Required config files
    primaryEnv: MY_API_KEY       # Main credential env var
    always: false                # Auto-activate without install?
    skillKey: my-skill           # Override invocation key
    emoji: "🔧"
    homepage: https://example.com
    os: [macos, linux]           # OS restrictions
    install:                     # Dependency install specs
      - kind: brew
        formula: jq
        bins: [jq]
---

# My Skill

Instruction content that tells the agent how to use this skill...
```

**Key characteristics:**
- Skills are **text-based instruction documents** meant to be injected into an agent's context
- They describe *how the agent should behave* when using a specific tool/API/capability
- Runtime requirements (env vars, binaries) are explicitly declared in frontmatter
- The body is free-form Markdown — instructions, examples, decision trees
- Skills are versioned (semver), distributed via registry, and moderated

**Data model (Convex schema):**
```typescript
skills: {
  slug: string,           // URL-safe unique identifier
  displayName: string,
  summary: string,        // From frontmatter description
  ownerUserId: Id<'users'>,
  tags: Record<string, Id<'skillVersions'>>,  // e.g., { latest: "xxx" }
  stats: { downloads, stars, versions, comments },
  badges: { highlighted, official, deprecated, ... },
  moderationStatus: 'active' | 'hidden' | 'removed',
  quality: { score, decision, signals, ... },
}

skillVersions: {
  skillId: Id<'skills'>,
  version: string,          // "1.2.3"
  files: Array<{ path, size, storageId, sha256, contentType }>,
  parsed: {
    frontmatter: Record<string, any>,
    clawdis: {              // Runtime metadata
      always, skillKey, primaryEnv, emoji, homepage, os,
      requires: { bins, anyBins, env, config },
      install: Array<{ kind, formula, package, bins, ... }>,
      nix: { plugin, systems },
    },
  },
  llmAnalysis: { status, verdict, confidence, dimensions, ... },
  changelog: string,
}
```

#### AgentFoundation Knowledge "Skill"

In AgentFoundation, a "skill" is a `KnowledgePiece` with specific field values:

```python
KnowledgePiece(
    content="# Optimizing Attention\n\n## Steps\n1. Profile current...",
    piece_id="uuid-xxx",
    knowledge_type=KnowledgeType.Procedure,   # WHAT: multi-step process
    info_type="skills",                        # WHERE: injected as skill
    tags=["attention", "optimization"],
    domain="model_optimization",               # Retrieval classification
    source="skill_synthesis",                   # How it was created
    summary="3-step technique for...",          # Progressive disclosure
    # ... versioning, validation, merge fields
)
```

**Key characteristics:**
- Skills are `Procedure`-type knowledge pieces routed to the `skills` info_type
- They are **auto-synthesized** by `SkillSynthesizer` from clusters of related knowledge pieces
- No concept of runtime requirements (env vars, binaries, OS constraints)
- No file bundles — skill is a single text content field
- Versioning is a simple integer (`version: 1, 2, 3...`) with `supersedes` chain
- Discovery is via the same retrieval pipeline (hybrid search, MMR, temporal decay)
- Budget-aware injection via `BudgetAwareKnowledgeProvider` (2000 tokens for skills)

**The `SkillSynthesizer` creates skills as:**
```python
KnowledgePiece(
    content=f"# {skill_name}\n\n{description}\n\n## Steps\n{formatted_steps}",
    knowledge_type=KnowledgeType.Procedure,
    info_type="skills",
    domain=source_pieces[0].domain,
    tags=aggregated_tags,
    source="skill_synthesis",
)
```

### 2.2 Fundamental Semantic Gap

This is the **critical insight**: the two systems use "skill" to mean different things.

| Aspect | ClawHub Skill | AF Knowledge Skill |
|--------|--------------|-------------------|
| **Ontological nature** | An **artifact** (document bundle distributed via registry) | A **knowledge unit** (synthesized from evidence) |
| **Creation** | Authored by humans, published to registry | Auto-generated by LLM from knowledge clusters |
| **Content model** | Multi-file bundle (SKILL.md + supporting files) | Single text field within a `KnowledgePiece` |
| **Metadata model** | Rich runtime requirements (env, bins, OS, install specs) | Domain/tag classification only |
| **Lifecycle** | Human-managed (publish, version, deprecate) | System-managed (synthesize, merge, expire) |
| **Distribution** | Global public registry with auth, moderation, stars | Local knowledge store, no sharing |
| **Audience** | Agent runtime (injected into context) | Agent runtime (injected into context) |
| **Format** | Markdown with YAML frontmatter | Markdown-formatted text (no frontmatter) |

**The ONLY overlap is the end-use**: both are ultimately Markdown-ish text meant to be injected into an agent's prompt to give it capabilities. This is the integration seam.

---

## 3. Compatibility Gap Analysis

### 3.1 Data Model Gaps

#### Gap A: No Runtime Requirements in KnowledgePiece

ClawHub's `requires` metadata (`env`, `bins`, `anyBins`, `config`, `os`) has **no equivalent** in `KnowledgePiece`. The AF model classifies knowledge by *topic* (domain, tags) but not by *runtime dependencies*.

**Impact:** If you import a ClawHub skill that requires `TODOIST_API_KEY` and `curl`, the knowledge system has no way to:
- Check if the agent's environment has these
- Warn the user about missing dependencies
- Filter skills by available runtime capabilities

**Severity:** HIGH — this is the most significant gap for practical use.

#### Gap B: No Multi-File Support

ClawHub skills can include supporting files (Python scripts, config templates, examples). `KnowledgePiece` has a single `content` field.

**Impact:** Supporting files would need to be either:
- Concatenated into the content field (losing file boundaries)
- Stored as separate `KnowledgePiece` entries linked by entity graph
- Stored in a separate file system with references

**Severity:** MEDIUM — most SKILL.md files are self-contained; supporting files are less common.

#### Gap C: Versioning Model Mismatch

ClawHub uses semver (`1.2.3`) with named tags (`latest`). AF uses integer version + `supersedes` chain.

**Impact:** Semver information would need to be stored in a custom field (e.g., `custom_tags` or metadata). The `supersedes` chain doesn't support semver ordering.

**Severity:** LOW — version tracking is secondary for runtime use.

#### Gap D: Taxonomy Mismatch

ClawHub skills use a **flat tag system** (user-defined strings like `"todoist"`, `"api"`, `"productivity"`). AF uses a **structured domain taxonomy** (12 ML-focused domains with predefined tags).

**Impact:** ClawHub tags don't map to AF domains. A ClawHub skill tagged `["todoist", "api"]` doesn't fit into `model_optimization`, `training_efficiency`, etc. The AF taxonomy is heavily ML-focused and would need extension.

**Severity:** HIGH — the taxonomy is a core retrieval filter; misclassified skills won't be found.

#### Gap E: No Moderation/Quality Metadata

ClawHub has rich moderation (`moderationStatus`, `quality.score`, `llmAnalysis`, `vtAnalysis`). AF has `validation_status` and `validation_issues` but no concept of community moderation, quality signals, or security analysis.

**Impact:** Trust signals from ClawHub would need custom storage or be lost.

**Severity:** LOW — the knowledge system's own validation can re-evaluate imported skills.

### 3.2 API and Protocol Gaps

#### Gap F: Cross-Language Barrier

ClawHub is TypeScript; AF is Python. Direct code sharing is impossible.

**Impact:** Integration must happen at the **data/API level**, not code level.

**Severity:** Intrinsic — not a "gap" to fix, but a constraint on integration approach.

#### Gap G: No Registry Client

AF has no HTTP client for the ClawHub API. ClawHub's public API is well-documented:
- `GET /api/v1/search?q=...` — vector search
- `GET /api/v1/skills` — list skills
- `GET /api/v1/skills/{slug}` — get skill detail
- `GET /api/v1/skills/{slug}/file?path=SKILL.md` — get file content
- `GET /api/v1/download?slug=...&version=...` — download zip

**Severity:** MEDIUM — straightforward to build.

### 3.3 Search and Retrieval Gaps

#### Gap H: Embedding Dimension Mismatch (Potential)

ClawHub uses OpenAI `text-embedding-3-small` (1536 dimensions). AF's embedding model is pluggable (via `KnowledgePieceStore` implementations). If the AF store uses a different embedding model, imported skills' embeddings can't be reused.

**Impact:** Imported skills would need re-embedding with AF's model.

**Severity:** LOW — re-embedding is a one-time cost per import.

#### Gap I: Skill Discovery Flow

ClawHub has a rich discovery flow: search, explore, trending, leaderboards, stars. AF has `KnowledgeBase.retrieve()` which does semantic/hybrid search over all pieces. There's no concept of "browsing skills" vs "searching knowledge."

**Impact:** Users can't browse or discover ClawHub skills within AF without building a discovery UX layer.

**Severity:** MEDIUM — depends on desired UX.

---

## 4. Integration Approaches

### 4.1 Approach A: Lightweight Import Adapter (Recommended)

**Strategy:** Build a Python adapter that fetches skills from ClawHub's HTTP API, transforms them into `KnowledgePiece` objects, and ingests them into the knowledge store.

```
┌─────────────────┐     HTTP API      ┌──────────────────┐
│   ClawHub.ai    │◄─────────────────►│  ClawhubClient   │
│   (Registry)    │  /api/v1/skills   │  (Python HTTP)   │
└─────────────────┘                   └────────┬─────────┘
                                               │
                                    ┌──────────▼─────────┐
                                    │  SkillTransformer   │
                                    │  - Parse SKILL.md   │
                                    │  - Extract metadata  │
                                    │  - Map to KP fields  │
                                    └──────────┬─────────┘
                                               │
                                    ┌──────────▼─────────┐
                                    │  KnowledgeBase      │
                                    │  .add_piece(piece)  │
                                    └─────────────────────┘
```

**Implementation outline:**

```python
# New file: knowledge/integrations/clawhub_adapter.py

@dataclass
class ClawhubSkillMetadata:
    """Parsed ClawHub skill metadata from YAML frontmatter."""
    name: str
    description: str
    version: str
    slug: str
    requires_env: List[str] = field(default_factory=list)
    requires_bins: List[str] = field(default_factory=list)
    requires_any_bins: List[str] = field(default_factory=list)
    requires_config: List[str] = field(default_factory=list)
    primary_env: Optional[str] = None
    always: bool = False
    skill_key: Optional[str] = None
    emoji: Optional[str] = None
    homepage: Optional[str] = None
    os_restrictions: List[str] = field(default_factory=list)
    install_specs: List[Dict[str, Any]] = field(default_factory=list)


class ClawhubClient:
    """HTTP client for the ClawHub registry API."""

    def __init__(self, base_url: str = "https://clawhub.ai", token: str = None):
        self.base_url = base_url
        self.token = token

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for skills via vector search."""
        # GET /api/v1/search?q={query}&limit={limit}
        ...

    def list_skills(self, sort: str = "updated", limit: int = 50) -> List[Dict]:
        """List skills with sorting."""
        # GET /api/v1/skills?sort={sort}&limit={limit}
        ...

    def get_skill(self, slug: str) -> Dict:
        """Get skill details."""
        # GET /api/v1/skills/{slug}
        ...

    def get_skill_file(self, slug: str, path: str = "SKILL.md",
                       version: str = None) -> str:
        """Get raw file content."""
        # GET /api/v1/skills/{slug}/file?path={path}&version={version}
        ...

    def download_skill(self, slug: str, version: str = None) -> bytes:
        """Download skill as zip."""
        # GET /api/v1/download?slug={slug}&version={version}
        ...


class ClawhubSkillTransformer:
    """Transform ClawHub skills into KnowledgePiece objects."""

    def transform(self, skill_data: Dict, skill_content: str,
                  supporting_files: Dict[str, str] = None) -> KnowledgePiece:
        """Transform a ClawHub skill into a KnowledgePiece.

        Args:
            skill_data: API response from GET /api/v1/skills/{slug}
            skill_content: Raw SKILL.md content
            supporting_files: Optional {path: content} for supporting files

        Returns:
            KnowledgePiece with knowledge_type=Procedure, info_type="skills"
        """
        metadata = self._parse_frontmatter(skill_content)
        body = self._strip_frontmatter(skill_content)

        # Build content: include requirement header + body
        content_parts = [body]
        if metadata.requires_env or metadata.requires_bins:
            req_section = self._format_requirements(metadata)
            content_parts.insert(0, req_section)

        # Append supporting files inline if present
        if supporting_files:
            for path, file_content in supporting_files.items():
                content_parts.append(
                    f"\n---\n## Supporting File: `{path}`\n\n{file_content}"
                )

        full_content = "\n\n".join(content_parts)

        return KnowledgePiece(
            content=full_content,
            knowledge_type=KnowledgeType.Procedure,
            info_type="skills",
            tags=self._map_tags(skill_data, metadata),
            domain=self._infer_domain(metadata, body),
            source=f"clawhub:{skill_data.get('skill', {}).get('slug', 'unknown')}",
            summary=metadata.description,
            custom_tags=[
                f"clawhub-slug:{skill_data.get('skill', {}).get('slug', '')}",
                f"clawhub-version:{skill_data.get('latestVersion', {}).get('version', '')}",
                *[f"requires-env:{e}" for e in metadata.requires_env],
                *[f"requires-bin:{b}" for b in metadata.requires_bins],
                *[f"os:{o}" for o in metadata.os_restrictions],
            ],
        )

    def _format_requirements(self, meta: ClawhubSkillMetadata) -> str:
        """Format runtime requirements as a Markdown section."""
        lines = ["## Runtime Requirements\n"]
        if meta.requires_env:
            lines.append(f"**Environment Variables:** {', '.join(meta.requires_env)}")
        if meta.primary_env:
            lines.append(f"**Primary Credential:** `{meta.primary_env}`")
        if meta.requires_bins:
            lines.append(f"**Required Binaries:** {', '.join(meta.requires_bins)}")
        if meta.requires_any_bins:
            lines.append(f"**Any Of Binaries:** {', '.join(meta.requires_any_bins)}")
        if meta.os_restrictions:
            lines.append(f"**OS:** {', '.join(meta.os_restrictions)}")
        return "\n".join(lines)
```

**Pros:**
- Minimal changes to existing AF code (no core model changes)
- ClawHub metadata preserved in `custom_tags` and content body
- Works with the existing retrieval pipeline unchanged
- Can be incrementally improved

**Cons:**
- Runtime requirements are embedded as text, not structured data — can't be programmatically checked
- Supporting files lose their file-level identity
- No bidirectional sync (AF → ClawHub)

### 4.2 Approach B: Extended KnowledgePiece with Runtime Metadata

**Strategy:** Extend the `KnowledgePiece` model with optional runtime requirement fields, then build the adapter on top.

```python
# Extension to KnowledgePiece — new optional fields:

@attrs
class KnowledgePiece:
    # ... existing fields ...

    # ── Runtime Requirements (for imported external skills) ──
    requires_env: List[str] = attrib(factory=list)
    requires_bins: List[str] = attrib(factory=list)
    requires_any_bins: List[str] = attrib(factory=list)
    requires_config: List[str] = attrib(factory=list)
    primary_env: Optional[str] = attrib(default=None)
    os_restrictions: List[str] = attrib(factory=list)
    install_specs: List[Dict[str, Any]] = attrib(factory=list)
    external_source_url: Optional[str] = attrib(default=None)
    external_version: Optional[str] = attrib(default=None)
```

Additionally extend the taxonomy:

```python
# New domains for the taxonomy
DOMAIN_TAXONOMY["agent_skills"] = {
    "description": "Agent capabilities, tool usage, and external service integration",
    "tags": [
        "api-integration", "cli-tool", "web-automation", "file-management",
        "communication", "productivity", "development", "system-admin",
        "data-analysis", "security", "monitoring",
    ],
}
DOMAIN_TAXONOMY["agent_configuration"] = {
    "description": "Agent setup, configuration, and environment management",
    "tags": [
        "env-setup", "credentials", "permissions", "plugin-config",
        "tool-config", "model-config",
    ],
}
```

And add a runtime requirements checker:

```python
class RuntimeRequirementsChecker:
    """Check if the current environment satisfies a skill's requirements."""

    def check(self, piece: KnowledgePiece) -> RequirementsResult:
        missing_env = [e for e in piece.requires_env if e not in os.environ]
        missing_bins = [b for b in piece.requires_bins if not shutil.which(b)]
        any_bins_met = not piece.requires_any_bins or any(
            shutil.which(b) for b in piece.requires_any_bins
        )
        os_ok = not piece.os_restrictions or sys.platform in self._map_os(
            piece.os_restrictions
        )
        return RequirementsResult(
            satisfied=not missing_env and not missing_bins and any_bins_met and os_ok,
            missing_env=missing_env,
            missing_bins=missing_bins,
            any_bins_met=any_bins_met,
            os_compatible=os_ok,
        )
```

**Pros:**
- Runtime requirements are structured and programmatically checkable
- Clean model extension (all new fields have defaults, backward compatible)
- Enables smart skill filtering: "only show skills I can actually use"
- Foundation for future skill marketplace features

**Cons:**
- Requires model changes (all stores need to handle new fields in serialization)
- More engineering effort upfront
- New fields are ClawHub-specific; may not generalize to other skill sources

### 4.3 Approach C: Dedicated Skill Registry Abstraction

**Strategy:** Create a separate `SkillRegistry` abstraction that wraps ClawHub (and potentially other sources) while keeping `KnowledgePiece` focused on knowledge.

```
┌───────────────────────────────────────────────┐
│           Agent Runtime                        │
│                                                │
│  ┌─────────────────┐  ┌────────────────────┐  │
│  │ KnowledgeBase   │  │ SkillRegistry      │  │
│  │ (facts, procs,  │  │ (external skills)  │  │
│  │  instructions)  │  │                    │  │
│  └────────┬────────┘  └────────┬───────────┘  │
│           │                    │               │
│  ┌────────▼────────────────────▼───────────┐  │
│  │     BudgetAwareKnowledgeProvider        │  │
│  │     (merges knowledge + skills)         │  │
│  └─────────────────────────────────────────┘  │
│                                                │
└───────────────────────────────────────────────┘

class SkillRegistry(ABC):
    """Abstract interface for skill sources."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Skill]: ...

    @abstractmethod
    def get(self, slug: str) -> Optional[Skill]: ...

    @abstractmethod
    def check_requirements(self, skill: Skill) -> RequirementsResult: ...

    @abstractmethod
    def get_content(self, skill: Skill) -> str: ...


class ClawhubSkillRegistry(SkillRegistry):
    """ClawHub-backed skill registry."""
    ...


class LocalSkillRegistry(SkillRegistry):
    """File-system backed skill registry (for local SKILL.md files)."""
    ...
```

**Pros:**
- Clean separation of concerns (knowledge vs skills)
- Supports multiple skill sources (ClawHub, local files, other registries)
- No model pollution — `KnowledgePiece` stays focused
- Natural extension point for future registries

**Cons:**
- Most engineering effort
- Duplicates some retrieval logic (search in knowledge + search in skills)
- Integration with `BudgetAwareKnowledgeProvider` needs careful design
- Skills don't benefit from knowledge system features (dedup, merge, validation)

---

## 5. Critical Analysis and Recommendations

### 5.1 What Approach Should We Take?

**Recommended: Approach A (Lightweight Import) first, evolve toward B as needs grow.**

Rationale:

1. **Approach A is immediately useful** with ~200 lines of code. It lets us import ClawHub skills today and benefit from the existing retrieval pipeline.

2. **Approach B's model extensions are only justified** when we have a concrete use case for programmatic requirements checking (e.g., an agent that auto-installs dependencies or filters skills by available tools).

3. **Approach C is over-engineered** for the current state. The knowledge system already handles "skills" as a first-class info_type with progressive disclosure. Creating a parallel system fragments the retrieval surface.

### 5.2 Taxonomy Extension is Non-Negotiable

Regardless of approach, the domain taxonomy **must be extended**. The current taxonomy is ML-focused (`model_optimization`, `training_efficiency`, etc.) and can't classify general-purpose agent skills like "Todoist task management" or "Git operations."

**Minimum additions:**

```python
"agent_skills": {
    "description": "Agent tool usage, API integrations, and task automation",
    "tags": ["api-integration", "cli-tool", "web-automation", "file-management",
             "communication", "productivity", "development", "devops",
             "data-analysis", "system-admin", "monitoring", "security"],
},
"external_knowledge": {
    "description": "Knowledge imported from external registries and sources",
    "tags": ["clawhub", "community-skill", "official-skill", "third-party",
             "verified", "unverified"],
},
```

### 5.3 The SKILL.md Frontmatter Parsing Problem

ClawHub's frontmatter uses YAML with a specific schema (`metadata.openclaw.requires.env`, etc.). The AF knowledge system has no YAML frontmatter parser. This is needed regardless of approach.

**Implementation: use `python-frontmatter` library (or manual YAML parsing):**

```python
import yaml
import re

def parse_skill_md(content: str) -> Tuple[Dict, str]:
    """Parse SKILL.md into (frontmatter_dict, body_markdown)."""
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return {}, content
    frontmatter = yaml.safe_load(match.group(1))
    body = content[match.end():]
    return frontmatter or {}, body
```

### 5.4 Security Considerations

**Trust boundary:** ClawHub skills are user-authored content from the internet. When importing into AF:

1. **Content validation is already handled** — `KnowledgeBase.add_piece()` calls `_validate_content()` which checks for sensitive patterns (API keys, passwords, tokens).

2. **BUT the validation is insufficient for imported skills** — a malicious SKILL.md could contain:
   - Prompt injection attempts ("Ignore previous instructions...")
   - Social engineering (fake error messages, urgency triggers)
   - Obfuscated exfiltration instructions

3. **Mitigation:** Leverage ClawHub's own security analysis (`llmAnalysis.verdict`, `moderationStatus`) during import:
   ```python
   def should_import(skill_data: Dict) -> Tuple[bool, str]:
       mod = skill_data.get("moderation")
       if mod and (mod.get("isMalwareBlocked") or mod.get("isSuspicious")):
           return False, "Skill flagged by ClawHub moderation"
       quality = skill_data.get("quality", {})
       if quality.get("decision") == "reject":
           return False, "Skill rejected by quality checks"
       return True, "OK"
   ```

### 5.5 Embedding and Search Integration

**Key decision:** Should imported ClawHub skills be searchable via the same retrieval pipeline as other knowledge?

**Yes.** This is one of the main benefits. An agent asking "How do I manage Todoist tasks?" should find both:
- Knowledge pieces about Todoist (if any were ingested from docs)
- The imported ClawHub skill for Todoist

Since `KnowledgePiece` objects are indexed by the same `KnowledgePieceStore`, imported skills will automatically participate in:
- Hybrid search (keyword + vector)
- MMR re-ranking (diversity)
- Temporal decay (freshness)
- Domain filtering (with extended taxonomy)
- Budget-aware injection (2000 token budget for skills)

The `BudgetAwareKnowledgeProvider._format_skills()` already does progressive disclosure (summary list → expand top skill), which works well for imported ClawHub skills.

### 5.6 Sync and Update Strategy

ClawHub skills are versioned and updated by their authors. Imported skills will go stale.

**Options:**

1. **Manual re-import:** User runs `clawhub_adapter.import_skill(slug)` again. Simple, explicit.

2. **Version-aware sync:** Store `clawhub-slug:xxx` and `clawhub-version:1.2.3` in `custom_tags`. On sync, check `GET /api/v1/skills/{slug}` for newer `latestVersion`. If newer, re-import with `supersedes` chain.

3. **Background sync job:** Periodic check of all imported skills for updates. Similar to `PostIngestionMergeJob` pattern.

**Recommendation:** Start with option 1, add option 2 when there are enough imported skills to justify it.

---

## 6. Proposed Implementation Plan

### Phase 1: Foundation (Minimum Viable Integration)

**New files:**
```
knowledge/integrations/__init__.py
knowledge/integrations/clawhub_client.py      # HTTP client
knowledge/integrations/clawhub_transformer.py  # SKILL.md → KnowledgePiece
knowledge/integrations/frontmatter_parser.py   # YAML frontmatter extraction
```

**Changes to existing files:**
```
knowledge/ingestion/taxonomy.py   # Add agent_skills, external_knowledge domains
knowledge/__init__.py             # Export new integration classes
```

**Deliverables:**
- [ ] `ClawhubClient` — search, get, download via HTTP API
- [ ] `ClawhubSkillTransformer` — SKILL.md → `KnowledgePiece` with proper classification
- [ ] Frontmatter parser (YAML) for extracting ClawHub metadata
- [ ] Extended taxonomy with `agent_skills` and `external_knowledge` domains
- [ ] Import function: `import_clawhub_skill(slug, kb) -> KnowledgePiece`
- [ ] Unit tests for transformer (frontmatter parsing, field mapping, edge cases)

### Phase 2: Smart Features

**New files:**
```
knowledge/integrations/requirements_checker.py  # Runtime requirements validation
knowledge/integrations/clawhub_sync.py          # Version-aware sync
```

**Deliverables:**
- [ ] `RuntimeRequirementsChecker` — check env vars, binaries, OS
- [ ] Version-aware sync: detect updates, re-import with supersedes chain
- [ ] Batch import: `import_clawhub_skills(slugs, kb)` with progress
- [ ] CLI integration: `kb import-clawhub <slug>` command

### Phase 3: Deep Integration (If Needed)

- [ ] Extend `KnowledgePiece` with optional runtime requirement fields (Approach B)
- [ ] Skill filtering by available capabilities in `BudgetAwareKnowledgeProvider`
- [ ] Two-way skill sharing (synthesized AF skills → ClawHub format)
- [ ] Background auto-discovery of relevant ClawHub skills based on agent's usage patterns

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ClawHub API changes or goes offline | Low | High | Cache imported skills locally; degrade gracefully |
| Rate limiting blocks bulk imports | Medium | Low | Respect rate limits (120 read/min); implement backoff |
| SKILL.md format evolves | Medium | Low | Frontmatter parser is lenient; unknown fields ignored |
| Taxonomy extension causes retrieval regression | Low | Medium | Test retrieval quality before/after extension |
| Imported skills contain prompt injection | Low | High | Combine ClawHub moderation signals + AF validation |
| Embedding dimension mismatch | Low | Low | Re-embed during import; don't reuse ClawHub embeddings |

---

## 8. Appendices

### A. ClawHub API Quick Reference (for implementers)

```
# Search (no auth)
GET https://clawhub.ai/api/v1/search?q=todoist&limit=10

# List skills
GET https://clawhub.ai/api/v1/skills?sort=trending&limit=50

# Get skill detail
GET https://clawhub.ai/api/v1/skills/todoist-cli

# Get file content
GET https://clawhub.ai/api/v1/skills/todoist-cli/file?path=SKILL.md

# Download zip
GET https://clawhub.ai/api/v1/download?slug=todoist-cli&tag=latest

# Resolve version
GET https://clawhub.ai/api/v1/resolve?slug=todoist-cli&hash=abc123...

# Rate limits: 120/min read (IP), 600/min read (token)
```

### B. Field Mapping Reference

| ClawHub Field | KnowledgePiece Field | Notes |
|--------------|---------------------|-------|
| `skill.slug` | `custom_tags: ["clawhub-slug:xxx"]` | Also encoded in `source` |
| `skill.displayName` | Content title (`# DisplayName`) | Embedded in content |
| `skill.summary` | `summary` | Direct mapping |
| `latestVersion.version` | `custom_tags: ["clawhub-version:1.2.3"]` | |
| `latestVersion.changelog` | Not mapped | Could go in content footnote |
| SKILL.md body | `content` | Main content field |
| `metadata.openclaw.requires.env` | `custom_tags: ["requires-env:KEY"]` | Phase 1 |
| `metadata.openclaw.requires.bins` | `custom_tags: ["requires-bin:curl"]` | Phase 1 |
| `metadata.openclaw.os` | `custom_tags: ["os:macos"]` | Phase 1 |
| `metadata.openclaw.always` | Not mapped | AF handles differently |
| `metadata.openclaw.primaryEnv` | Not mapped | Could go in requirements section |
| `metadata.openclaw.emoji` | Not mapped | Not used in AF |
| `metadata.openclaw.homepage` | Not mapped | Could go in `source` |
| `skill.stats` | Not mapped | Not relevant for local use |
| `skill.badges` | Not mapped | Could inform trust level |
| `moderation.isSuspicious` | Import gate (reject if true) | Security filter |
| `owner.handle` | `custom_tags: ["clawhub-author:handle"]` | |

### C. ClawHub Convex Schema Summary

```
Tables: skills, skillVersions, skillEmbeddings, skillVersionFingerprints,
        skillBadges, skillDailyStats, skillLeaderboards, skillStatEvents,
        souls, soulVersions, soulEmbeddings, soulVersionFingerprints,
        users, comments, commentReports, skillReports, soulComments,
        stars, soulStars, auditLogs, apiTokens, rateLimits, ...

Embeddings: OpenAI text-embedding-3-small (1536 dimensions)
Vector Index: Convex native vector index on skillEmbeddings.embedding
Filter Fields: visibility (for embedding visibility control)
```

### D. Existing AF Knowledge Skill Pipeline

```
Knowledge Pieces → SkillSynthesizer detects cluster
                 → Calls LLM with SKILL_SYNTHESIS_PROMPT
                 → Creates Procedure-type KnowledgePiece with info_type="skills"
                 → Stored in KnowledgePieceStore
                 → Retrieved via standard retrieval pipeline
                 → Injected via BudgetAwareKnowledgeProvider._format_skills()
                 → Budget: 2000 tokens, progressive disclosure
```

### E. Files That Would Be Modified/Created

```
# New files (Phase 1)
knowledge/integrations/__init__.py
knowledge/integrations/clawhub_client.py
knowledge/integrations/clawhub_transformer.py
knowledge/integrations/frontmatter_parser.py

# Modified files (Phase 1)
knowledge/ingestion/taxonomy.py            # Add new domains
knowledge/__init__.py                      # Export integration classes

# New files (Phase 2)
knowledge/integrations/requirements_checker.py
knowledge/integrations/clawhub_sync.py

# Test files
tests/knowledge/integrations/test_clawhub_client.py
tests/knowledge/integrations/test_clawhub_transformer.py
tests/knowledge/integrations/test_frontmatter_parser.py
tests/knowledge/integrations/test_requirements_checker.py
```
