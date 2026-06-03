# SOP Runtime Enablement Plan — Yolo, Meta-Tags, First-Class SOP Resources, Auto-Advance, Session Storage

**Author:** Rovo Dev (drafted in conversation with Tony Chen)
**Date:** 2026-05-26 v1 (13:54) · v2 (15:17 — Claude integration round)
**Status:** Draft v2 — Claude empirical corrections applied; ready for review
**Companion to:** `conversational_workflows_and_sop_framework_INTEGRATED_v7.2_plan.md` (this plan is the next step after v7.2's architecture lands; v7.2 already specified the SOP runtime substrate, this plan operationalizes it for the first real SOP — `role_creation.md` — to run end-to-end)

---

## §-1. v1 → v2 integration delta (transparency)

Re-read at 2026-05-26 15:17: Claude's plan (`/Users/tchen7/.claude/plans/can-you-help-take-ticklish-whisper.md`, 268 lines) was compared line-by-line against v1. Three empirical bugs in v1 were caught by Claude and are corrected in v2:

| # | v1 bug | Empirical evidence | v2 fix |
|---|---|---|---|
| **D1** | v1 named only `conversational_inferencer.py:682` as the yolo-stripping site | ✅ `sop_manager.py:568 render_for_mode(content, mode="yolo")` *also* strips `[__requires confirmation__]` lines — verified | §4.4 now lists BOTH deletion sites |
| **D2** | v1 used singular `resources/sop/` | ✅ `resources/skills/`, `resources/tools/` are **plural** in the codebase — verified `ls resources` | §5.2/§5.4/§11 renamed to `resources/sops/` |
| **D3** | v1 invented `<server>/sop/<...>` as a new top-level runtime ladder | ✅ v5.3 `unified_workspace_allocation` already has `allocate_tool_workspace(name)` — reuse it for `<session_root>/sops/<name>_<TS>_<uuid8>/` (no new ladder) | §6.1 rewired to compose with v5.3 instead of paralleling it |

Plus four design refinements adopted from Claude:

| # | Claude's improvement | Adopted |
|---|---|---|
| **D4** | `_handle_conversation_tool` at line 970 — concrete entry point | ✅ §4.4 retargeted |
| **D5** | `ConversationToolType` enum (MULTIPLE_CHOICE, SINGLE_CHOICE, CONFIRMATION, CLARIFICATION, TOOL_ARGUMENT_FORM) — use enum, not strings | ✅ §4.2 strategy table aligned with enum |
| **D6** | Synthetic responses written into `self._messages.append({..., "synthetic": True})` (single record, not parallel `synthetic_trace.json`) | ✅ §4.5 / §6.4 simplified |
| **D7** | Per-turn log = `session.jsonl` (RankEvolve convention), not split `messages.jsonl + tool_calls.jsonl` | ✅ §6.1 layout simplified |

Three v1 items v2 *preserves* against Claude's simpler alternative (with rationale):

| # | v1 retains | Why (vs Claude's choice) |
|---|---|---|
| **R1** | `sop.json` (not `sop.config.yaml`) | Existing convention: `tool.json`, `skill.json`. YAML would be inconsistent |
| **R2** | Six strategies including `static:<json>` escape hatch | Defensive — SOP authors WILL need this when default strategy doesn't fit; cheaper to add upfront |
| **R3** | Risk register (12 rows), Open questions (10 rows), Feature flag, OpenStartup re-export pattern | Operational discipline; Claude's plan is leaner but skips guardrails. Keeping mine |

One v1 design **simplified** by Claude's pressure:

| # | v1 design | v2 simplification |
|---|---|---|
| **S1** | `sop.meta: dict[str, list[str]]` (over-abstracted for 2 use cases) | First-class `sop.keywords: list[str]` + `sop.example_requests: list[str]` fields. Future meta-tags can be added with a one-line attrib each. Avoids the "stringly-keyed dict" anti-pattern |

### v2 → v2.1 patch (2026-05-26 15:38 — external review round)

External reviewer audited v1 (pre-Claude). v2.1 evaluates each claim against v2 (current):

| # | Reviewer claim | Verdict | Action |
|---|---|---|---|
| **A** | "v1's `sop.meta: dict` is better than first-class fields" | **REJECTED** | v2's first-class design (S1 above) is correct: typed access, IDE/lint coverage, fail-loud on unknown tags. The "future-proof" abstraction is YAGNI for 2 use cases. **v2 stands.** |
| **B** | "v1's `resources/sop/` (singular) matches existing convention" | **REJECTED — factually false** | Empirically verified: `ls resources/` shows `skills/` and `tools/` are **plural**. v2's `sops/` is correct. **v2 stands.** |
| **C** | "v1/v2's §2.3 description leaks implementation detail (`--yolo`)" | **VALID** — applied | §2.3 description rewritten to remove 4 leakage sites: "multi-choice prompt" → "with the user"; "SKILL.md files and tool specifications" → "drafts those skills and tools"; "yolo mode (`--yolo`)" + "configured default responses" → dropped entirely. New description: 432 chars (was 651); AC2-3 strengthened to explicitly list banned term categories (phase names, tool type names, artifact filenames, invocation mechanics). |

### v2.1 → v3 patch (2026-05-26 15:42 — second external plan integration)

A second external agent submitted a complete alternative plan (different framing). Empirically verified against the actual codebase, **the other agent's plan is more accurate in 5 critical empirical claims** (v2 was wrong); v2 retains operational discipline the other plan lacks. v3 = (their empirical accuracy) ∩ (v2's operational discipline). Five v2 claims corrected:

| # | v2 was wrong | Other plan was right | v3 fix |
|---|---|---|---|
| **C1 — Yolo fix target** | "Delete `conversational_inferencer.py:682-690`" | **`yolo_mode` is dead at line 111 — zero other refs in the file**. Line 682-690 is an unrelated unconditional auto-strip (orthogonal bug). The actual yolo fix is wiring `yolo_mode` into `_handle_conversation_tools:1099`. | §4.4 rewritten; line 682 separated as a parallel cleanup |
| **C2 — Config file format** | "`sop.json` sidecar (matches `tool.json`)" | **Skills use YAML frontmatter in the markdown itself (verified `_parse_skill_md` at `skills/registry.py:31`).** SOPs are markdown-bearing like skills; should match skills, not tools. | §5.3 rewritten — YAML frontmatter inside `SOP.md`; no sidecar |
| **C3 — Prompt slots** | "Add 2 new prompt sections (Available SOPs + Active SOPs)" | **`available_workflows` + `ongoing_workflows` slots already exist** at `initial.jinja2:7-15` — just unpopulated. | §5.5 simplified — populate existing slots |
| **C4 — Meta-tag layer** | "`[__keywords__]` / `[__example_requests__]` are SOP-level parser meta-tags" | **Wrong layer.** They belong in frontmatter (sibling to skills), NOT in the `[__directive__]` grammar (which is for phase-level control flow). | §3 reframed: keywords/example_requests are frontmatter fields, no SOPManager regex change needed |
| **C5 — Render-for-mode deletion** | "Recommendation: delete `render_for_mode` entirely" | **Premature** — line-682 block is a separate, orthogonal defect; needs to be addressed in its own ticket | §4.4 retains `render_for_mode` for backward compat |

Ten additional improvements from the other plan adopted in v3:

| # | Adopted | v3 location |
|---|---|---|
| **N1** | YAML frontmatter (`--- ... ---`) with `name`/`description`/`keywords`/`example_requests`/`metadata` | §5.3 |
| **N2** | Reuse `_parse_skill_md` verbatim (zero new parser code) | §3, §5.4 |
| **N3** | Per-tool `yolo_default` block in `tool.json` with concrete table (5 tool types × strategy + synthetic_text) | §4.3 |
| **N4** | `MessageMetadata.is_synthetic_default: bool` in `server/schema.py` (parallel to existing `is_auto_advance`) | §4.5 |
| **N5** | SOP entry via `{"type":"sop","name":"<sop>"}` tool envelope; new dispatch in `tool_call_parser.py` | §5.7 (NEW) |
| **N6** | MVP scope-cut: 1 active SOP at first; schema multi-capable | §6.6 (NEW) |
| **N7** | `get_session_sops_dir(session_id)` helper alongside existing `get_session_tasks_dir` | §6.1 |
| **N8** | Delete orphaned `_variables/workflow/sop.jinja2` + `.sop.config.yaml` | §11 |
| **N9** | New `tool_argument_form/tool.json` (currently missing — only enum exists) | §4.3 |
| **N10** | `SOPSubInferencer` thin wrapper class (each active SOP is a sub-conversational-session) | §6.2 (NEW) |

v2 strengths retained against other plan's simplifications:
- Audit history (§-1) — other plan had none
- Risk register (§8) + Open questions (§9) — other plan had neither
- Feature flag for rollback (§7) — other plan had no rollback strategy
- `static:<json>` per-SOP override escape hatch (§4.2) — other plan had no per-SOP override
- Tag scope taxonomy (§3.2) — other plan didn't formalize layers
- OpenStartup SOP registry re-export pattern (§5.6) — other plan only mentioned the dir
- 30+ ACs — richer than other plan's verification bullets

### v3 → v3.1 patch (2026-05-26 15:54 — third audit round; v1→v2→v3 residue cleanup)

External reviewer audited v3 for stale references and protocol-shape correctness. **All 10 claims verified VALID** (empirically checked against `conversational_inferencer.py:970-1187`). Highest-impact finding: **synthetic response shapes were invented and would have silently misrouted**. All applied:

| # | Severity | Issue | v3.1 fix |
|---|---|---|---|
| **4** | **HIGH** | §4.4 synthetic response format (`{"selected": [...]}`, `{"confirmed": True}`) doesn't match real protocol — `_handle_conversation_tools:1099` returns `Optional[dict[str, str]]` from `user_input.get("values", ...)`; internal post-processing uses `choice_index`, `choice`, `custom_text`, `param_overrides`, `variable_override` (verified at lines 1060-1090) | §4.4 rewritten with concrete shape table; new `_synthesize_yolo_collected` signature returns `dict[str, str]`; reuses existing `_process_widget_response` instead of reinventing |
| **1** | MED | AC3-1 says `sop.meta` (stale v1 dict design) | Rewritten to `SOPInfo(keywords=[...], example_requests=[...], metadata={...})` per v3 frontmatter approach |
| **2** | MED | AC3-4 says "only meta dict grows" but §3.5 chose fail-loud first-class fields | Rewritten: "one `attrib` on `SOPInfo` + one line in registry mapping; unknown frontmatter keys silently ignored by YAML loader" |
| **3** | MED | §3.4 obsolete preamble code sets `sop.meta = preamble_meta` (stale v2 design) | Lines patched to populate `sop.keywords`/`sop.example_requests`; added "OBSOLETE under v3" banner pointing to §3.3/§3.4 v3 path |
| **5** | MED | Missing `ToolDefinition.yolo_default_response` field spec in §11 inventory | Added `models.py` + `conversation_tools.py` rows with explicit field/propagation spec |
| **6** | MED | §6.2 Decision 3 + §6.5 + AC6-3 still reference `synthetic_trace.json` (contradicts D6) | All 3 sites updated: single-record persistence via `session.jsonl` UserInput record fields |
| **7** | MED | AC6-2 references `messages.jsonl + tool_calls.jsonl` per-turn split files (contradicts D7) | Rewritten: turn folder for streaming cache + prompts only; events to single `session.jsonl` |
| **8** | LOW | §6.3 uses singular `<runtime_root>/sop/` (rest of plan uses plural `sops/`) | Changed to `<session_root>/sops/` consistent with D2/§6.1 |
| **9** | LOW | §5.5 prompt example leaks "multi-choice prompt" (banned by AC2-3) | Replaced with AC2-3-compliant `clarifies … with the user, researches the …` |
| **10** | LOW | §7 Phase 2 description references `SOP.meta` field | Rewritten: "registry uses parsed frontmatter via `_parse_skill_md`" |

**v3.1 net effect:** Plan now has consistent semantics across §3 (frontmatter), §4 (yolo protocol shapes), §6 (single-file session log), §11 (complete file inventory). Two new file-modification rows added (`models.py`, `conversation_tools.py`) — Claim 5 was a real implementation gap, not just a doc fix. Plan grew ~70 lines; net code-implementation surface unchanged (the new rows reflect existing-but-undocumented work).

---

## §0. Scope & framing

The v7.2 plan formalized the SOP framework architecture (parser, `SOPWorkGraphNode`, `BranchBarrierNode`, conversation tools, WorkflowManager). This plan is the **runtime enablement** layer: it takes one real SOP (`role_creation.md`) and makes it *actually executable* end-to-end. Five orthogonal concerns:

| # | Concern | One-line summary |
|---|---|---|
| 1 | **SOP description** | Refine `role_creation.md`'s overview so it functions as a skill-like description (info-dense, comprehensive, enters-well from the conversation prompt) |
| 2 | **SOP-level meta-tags** | Add `__keywords__` and `__example_requests__` parsing — orthogonal to existing phase-level tags like `[__initial__]` |
| 3 | **Synthetic auto-advance for yolo mode** | Replace the current "strip `[__requires confirmation__]` from text" approach with "fire the conversation tool, return a configured default response labeled as `synthetic`" |
| 4 | **First-class SOP resources** | Promote SOPs from `_variables/workflow_sop/*.md` to `resources/sops/<sop_name>/{SOP.md, sop.json}` — mirroring the skill / tool pattern; expose them in the conversation prompt as a discoverable list + active-instances list |
| 5 | **SOP runtime session storage** | Each running SOP = one conversational session; `<server>/sop/<sop_session>/turns/<turn_id>/` with synthetic-vs-human turn labels; task tool workspaces nested under the SOP session's `tasks/` folder |

These five concerns are sequenced for low-risk incremental landing — concern 1 (text edit) is trivial; concerns 2–4 build on each other; concern 5 is the only one that interacts with the OpenStartup runtime layout (and the `unified_workspace_allocation` v5.3 work).

**Non-goals:**
- Re-architecting the v7.2 substrate (already decided, already implemented up to `SOPWorkGraphNode` / `BranchBarrierNode` / parser / registry).
- Replacing the existing tool/skill registry pattern (we *adopt* it for SOPs).
- Touching non-yolo confirmation flow (keep `confirm_action` tool unchanged for interactive mode).

---

## §1. Empirical baseline (verified 2026-05-26)

| Component | Path | State |
|---|---|---|
| SOP parser | `RichPythonUtils/.../template_manager/sop_manager.py:244` `SOPManager.parse_markdown` | ✅ Exists; tokenizes `[__directive__]` at phase top; **does NOT recognize SOP-level (pre-Phase 0) directives** |
| `SOPPhase` class | `sop_manager.py:150` (`StateNode` subclass) | ✅ Has `name`, `description`, `directives`, `subsections` |
| `SOP` class | `sop_manager.py:164` (`StateGraph` subclass) | ✅ Has phases; **no `keywords` / `example_requests` fields** |
| `WorkflowDefinition` | `AgentFoundation/.../workflow/definition.py` | ✅ Frozen dataclass; has `workflow_id`, `name`, `description`, `available_tools`, `available_modes` (default `["default", "yolo"]`) |
| `WorkflowRegistry.load_all` | `AgentFoundation/.../workflow/registry.py:33` | ✅ Globs `*.md`; description = lines before first `## Phase` heading; **takes only first 500 chars**; default search path is `resources/prompt_templates/conversation/main/_variables/workflow_sop/` |
| SOP tool | `AgentFoundation/.../resources/tools/sop/` | ✅ Exists; `tool.json` declares `--yolo` flag; `executor.py:execute` parses args, calls `WorkflowRegistry.load_all`, builds `WorkflowManager` and runs |
| `role_creation.md` (current draft) | `_variables/workflow_sop/role_creation.md` | Has `[__keywords__]` and `[__example_requests__]` markers **but parser ignores them** |
| Yolo mode handling (current) | `conversational_inferencer.py:682` | ❌ "Auto-complete confirmation-gate phases" if phase has `"requires confirmation"` in directives — **silently skips, no tool fires, no synthetic response is logged**. This is the behavior the user wants to replace. |
| Conversation tool schema | `resources/tools/{single_choice,clarification,...}/tool.json` | ✅ Each has `tool_type: "Conversation"`, `category: "conversation"`, `parameters`. **No `yolo_default_response` field today.** |
| Skills registry pattern | `resources/skills/registry.py` | ✅ Scans `resources/skills/*/SKILL.md`, parses YAML frontmatter; supplies `SkillInfo(name, description, labels, metadata, file_path, body)` |
| Conversational session storage | `conversational_inferencer.py:143-187, 321-326` | ✅ `on_new_turn(turn_number, user_input)` callback creates "turn directory"; messages get `turn_number` metadata; **synthetic-vs-human label NOT in turn record today** |

**Key empirical gap matrix:**

| Gap | Severity | Affects concern |
|---|---|---|
| `SOPManager.parse_markdown` ignores top-of-file directives | HIGH | #2 |
| `WorkflowDefinition` has no field for SOP keywords / example requests | HIGH | #2, #4 |
| Yolo mode silently skips confirmation-gate phases without logging anything | HIGH | #3 |
| Conversation `tool.json` has no `yolo_default_response` configuration | MEDIUM | #3 |
| SOPs live under prompt_templates (an internal-variables location), not under `resources/sops/` | MEDIUM | #4 |
| Conversation prompt has no "available SOPs" section (only available tools/skills) | MEDIUM | #4 |
| Turn records have no `source: human | synthetic` label | LOW | #5 |
| No dedicated `<server>/sop/<session>/` runtime folder | MEDIUM | #5 |

---

## §2. Concern 1 — Refine SOP description (role_creation.md)

### §2.1 What the description is for

The text *before the first `## Phase` heading* in an SOP file becomes `WorkflowDefinition.description` (via `WorkflowRegistry._parse_definition`, lines 67-79, truncated to 500 chars). That description has three downstream uses:

1. **`/sop` tool argument completion** — when the user types `/sop <TAB>`, the tool surfaces workflows with their descriptions.
2. **Discovery in the conversation prompt** — once concern #4 (first-class SOP resources) lands, all `WorkflowDefinition`s will be listed in a "Available SOPs" section of the prompt, with each entry showing `name: description`. The LLM uses this to decide *when to enter* an SOP — exactly like deciding when to invoke a skill.
3. **Active-instance prompt section** — once entered, the description + current phase status + next-step guidance render together (per v7.2 §6.6).

This means the description's quality determines whether the LLM enters the SOP at the right moments. A vague description (today's `"The Orchestrator follows a phased workflow for creating and deploying AI employees..."`) loses against more concrete skill descriptions in the same context window.

### §2.2 Skill-description quality bar (the model to copy)

The existing skill registry (`SkillInfo.description`) uses one-paragraph, info-dense, action-oriented descriptions. Verified sample patterns from `resources/skills/*/SKILL.md`:

- Starts with a verb describing what the skill does ("Conducts deep, thorough research on…").
- Names the trigger conditions explicitly ("Use when…").
- Mentions inputs/outputs concretely ("Results are synthesized into a cohesive, theme-organized markdown report saved in the research-reports directory.").
- 2-4 sentences total; ~300-500 characters.

### §2.3 Refined `role_creation.md` overview

Replace lines 1-7 of `role_creation.md` with:

```markdown
# AI Employee Role Creation

Creates a new AI employee role end-to-end: clarifies the role's responsibility categories with the user, researches the domain to produce a comprehensive responsibility document, decomposes the role into the skills and tools it needs, drafts those skills and tools, and deploys the role. Use when the user wants to hire, onboard, design, or stand up a new AI employee for any function (engineering, PM, support, data science, etc.).

[__keywords__] hire employee, create role, onboard AI employee, build AI agent, design AI role, deploy AI worker, stand up new role
[__example_requests__]
- hire a machine learning engineer
- create a TPM AI role
- I want to build an AI customer support lead
- onboard an AI data scientist for our analytics team
- design a new AI role that handles release coordination
```

### §2.4 Acceptance criteria

- AC2-1: `WorkflowRegistry._parse_definition("role_creation.md").description` is ≤500 chars, starts with a verb, contains the phrase "Use when".
- AC2-2: Manually rendered against existing skill-listing format in the prompt, the SOP description visually parallels the skill descriptions in info density (subjective; reviewed by Tony).
- AC2-3: The refined text contains no implementation detail leakage — specifically no (a) phase names, (b) conversation tool type names like "multi-choice prompt", (c) concrete artifact filenames like "SKILL.md", (d) invocation mechanics like "yolo" / "--yolo" / "configured default responses". Those belong in phase guidance, `sop.json`, or the SOP tool's `tool.json`, not the SOP description.

---

## §3. Concern 2 — SOP routing metadata (`keywords`, `example_requests`) — v3: frontmatter, not parser directive

### §3.1 The problem (v3 framing)

`role_creation.md` needs `keywords` and `example_requests` for the conversation prompt's "Available Workflows" routing. v2 framed these as new `[__keywords__]` / `[__example_requests__]` parser directives — **wrong layer**. These are SOP-level routing metadata (sibling to skill frontmatter labels), not phase-level control-flow directives.

**v3 correct framing:** SOPs become markdown-with-YAML-frontmatter (exactly like skills); the existing skill frontmatter parser `_parse_skill_md` at `skills/registry.py:31-75` handles them verbatim. The `[__...__]` directive grammar inside the phase body remains untouched; SOPManager regex catalog (`sop_manager.py:46-91`) needs **zero changes**.

### §3.2 Tag scope taxonomy (v3 — formalized as 3 layers)

| Scope | Location | Examples | Affects |
|---|---|---|---|
| **SOP-level frontmatter** *(NEW)* | YAML block before first markdown content | `name`, `description`, `keywords`, `example_requests`, `metadata.requires_confirmation`, `metadata.default_yolo` | `SOPInfo` fields; discovery; prompt routing block |
| Phase-level directive | Line after `## Phase X` heading | `[__initial__]`, `[__depends on Phase X]`, `[__branch__]`, `[__requires confirmation__]`, `[__goto Phase X__]` | `SOPPhase.directives`; runtime control flow |
| Subsection instruction tag | After subsection name (e.g., `Tools[__must__]:`) | `[__must__]`, `[__optional__]` | LLM guidance only (no runtime effect) |

**Critical v3 correction:** v2 conflated SOP-level routing metadata with phase-level directives. v3 keeps them in different layers: routing in frontmatter (YAML-parsed), control flow in body (regex-parsed by SOPManager). This matches the skill convention exactly.

### §3.3 SOP.md file format (v3 — YAML frontmatter, matches skill convention)

```markdown
---
name: role_creation
description: >
  Creates a new AI employee role end-to-end: clarifies the role's responsibility
  categories with the user, researches the domain to produce a comprehensive
  responsibility document, decomposes the role into the skills and tools it needs,
  drafts those skills and tools, and deploys the role. Use when the user wants
  to hire, onboard, design, or stand up a new AI employee for any function.
keywords:
  - hire employee
  - create role
  - onboard AI employee
  - build AI agent
  - design AI role
  - deploy AI worker
example_requests:
  - hire a machine learning engineer
  - create a TPM AI role
  - I want to build an AI customer support lead
  - onboard an AI data scientist for our analytics team
metadata:
  requires_confirmation: true
  default_yolo: follow_judgment
---

# AI Employee Role Creation

## Phase 0 [__initial__]
... (existing phase body, unchanged from current role_creation.md) ...
```

**No EBNF extension needed.** The YAML block is parsed by the existing `_parse_skill_md` regex (`^---\s*\n(.*?)^---\s*\n`); the body is then handed to `SOPManager.parse_markdown` exactly as today. v2's proposed EBNF/preamble parser is unnecessary.

### §3.4 Parser changes (v3 — no SOPManager change; reuse `_parse_skill_md`)

**`sop_manager.py` is unchanged.** All SOP-level routing metadata is in YAML frontmatter, which the SOPManager phase parser never sees (frontmatter is stripped by the registry before the body is handed to SOPManager).

**Registry loader** (`AgentFoundation/.../resources/sops/registry.py:load_sop`):

```python
def load_sop(name: str, base_dir: Path | None = None) -> SOPInfo:
    """Mirrors load_skill() exactly."""
    sop_path = (base_dir or _DEFAULT_BASE) / name / "SOP.md"
    content = sop_path.read_text()
    frontmatter, body = _parse_skill_md(content)  # REUSED VERBATIM
    return SOPInfo(
        name=frontmatter.get("name", name),
        description=frontmatter.get("description", "").strip(),
        keywords=frontmatter.get("keywords", []),
        example_requests=frontmatter.get("example_requests", []),
        metadata=frontmatter.get("metadata", {}),
        file_path=str(sop_path),
        body=body,
    )
```

**Old parser-state pseudocode v2 spec'd (DELETED — no longer needed):**

The current parser walks the markdown line-by-line, switching state on heading detection. We add a preamble state:

**Before:** parser starts in "expect first phase" state.
**After:** parser starts in "preamble" state; transitions to "phase" state on first `## Phase` / `### Phase` heading.

**Preamble-state handling:**

```python
# Pseudo-spec (concrete diff in §11 phase 1)
preamble_meta: dict[str, list[str]] = {}
current_meta_key: str | None = None

for line in lines_before_first_phase_heading:
    stripped = line.strip()
    # SOP-level tag (inline form, e.g. `[__keywords__] foo, bar`)
    inline_match = re.match(r'^\[__(\w+)__\]\s*(.*)$', stripped)
    if inline_match:
        key, value = inline_match.group(1), inline_match.group(2).strip()
        # value (if present) is comma-split for keywords / single line for free-text
        if value:
            preamble_meta.setdefault(key, []).extend(_split_meta_value(key, value))
        current_meta_key = key  # subsequent list items append here
        continue
    # List item under last-seen meta tag (e.g. "- hire machine learning engineer")
    if current_meta_key and stripped.startswith("- "):
        preamble_meta.setdefault(current_meta_key, []).append(stripped[2:].strip())
        continue
    # Non-tag, non-list-item line ends the metadata block (back to description)
    if stripped and not stripped.startswith("#"):
        current_meta_key = None

sop.keywords = preamble_meta.get("keywords", [])           # v3.1: first-class
sop.example_requests = preamble_meta.get("example_requests", [])  # v3.1: first-class
# NOTE (v3): This v2 preamble-state code is OBSOLETE under v3 (frontmatter approach).
#            Kept here only as historical reference. The current v3 path uses
#            _parse_skill_md on the SOP.md frontmatter; SOPManager body parser is unchanged.
#            See §3.3 / §3.4 for the v3 path.
```

`_split_meta_value(key, value)` helper: for `keywords`, split on commas; for `example_requests`, return single-element list. Future meta-names can extend this dispatcher.

### §3.5 `SOP` class additions (`sop_manager.py:164`) — v2: first-class fields

Add first-class typed fields (S1 — v2 simplification of v1's `sop.meta` dict):

```python
@attrs(slots=False, kw_only=True)
class SOP(StateGraph):
    # ... existing fields ...
    keywords: list[str] = attrib(factory=list)            # NEW: from [__keywords__]
    example_requests: list[str] = attrib(factory=list)    # NEW: from [__example_requests__]
```

**Rationale (v2 revision):** v1 used a generic `meta: dict[str, list[str]]` to "future-proof" for unknown meta-tags. After Claude review, we adopted Claude's first-class field approach because (a) only two meta-tags exist today, (b) consumers benefit from typed access (`sop.keywords` vs `sop.meta.get("keywords", [])`), (c) IDE/lint coverage works on attribs but not dict lookups, (d) adding a new meta-tag later is a one-line `attrib` addition, not a schema change. The "future-proof" abstraction was premature.

**Parser dispatch (v2):** When the preamble state sees `[__keywords__]`, it appends to `sop.keywords`; `[__example_requests__]`, to `sop.example_requests`. Unknown meta-tags raise a `SOPParseError("Unknown SOP-level meta-tag '__{name}__'. Known tags: keywords, example_requests")` instead of silently accumulating in a dict — fails loudly.

### §3.6 `WorkflowDefinition` propagation (`definition.py`)

Add two derived fields mirroring SOP class (v2):

```python
@dataclass(frozen=True)
class WorkflowDefinition:
    # ... existing fields ...
    keywords: list[str] = field(default_factory=list)           # NEW (from sop.keywords)
    example_requests: list[str] = field(default_factory=list)   # NEW (from sop.example_requests)
```

`WorkflowRegistry._parse_definition` populates them from `sop.keywords` / `sop.example_requests` (1:1 mapping). Description-line accumulator (lines 70-78) is updated to **skip** any line matching `^\[__\w+__\]` so meta-tag lines don't leak into description.

### §3.7 Acceptance criteria

- AC3-1: For an SOP with frontmatter, `load_sop("role_creation")` returns `SOPInfo(keywords=[...], example_requests=[...], metadata={...})`. (v3.1 fix: was `sop.meta` dict — but §3.5 chose first-class fields; v3 chose frontmatter; AC now matches both decisions.)
- AC3-2: `WorkflowDefinition` exposes `keywords: list[str]` and `example_requests: list[str]`.
- AC3-3: Description string does **not** contain `[__keywords__]` or `[__example_requests__]` substrings (regression: today they leak).
- AC3-4: Adding a new frontmatter field (e.g., `estimated_duration: 30m`) requires one `attrib` on `SOPInfo` + one line in the registry's frontmatter→SOPInfo mapping. Unknown frontmatter keys are silently ignored by the YAML loader (no fail-loud needed — frontmatter is freeform per skill convention). (v3.1 fix: replaces stale `_split_meta_value`/dict-grows claim; aligns with §3.4 v3 frontmatter approach.)
- AC3-5: A phase-level `[__initial__]` is still parsed correctly (regression: no scope confusion).

---

## §4. Concern 3 — Synthetic auto-advance for yolo mode

### §4.1 Why we need a different yolo

The current behavior at `conversational_inferencer.py:682-690` is:

> Inspect each available phase; if it has no tools AND its directives contain "requires confirmation", silently mark it `completed`.

This is wrong for three reasons:

1. **It's text-based and brittle.** A phase like Phase 0 of `role_creation.md` carries `[__requires confirmation__]` but **also** invokes the `multiple_choice` tool — the current logic explicitly skips this case (`if not has_tools and ...`). So Phase 0 in yolo mode still fires the tool and **blocks** waiting for user input. Yolo mode doesn't work for the most important case.
2. **No audit trail.** A skipped gate leaves no turn record, no synthetic response, no trace of "what would have been asked." Re-running in interactive mode is the only way to see what was bypassed.
3. **It's not consistent with v7.2's `confirm_action` tool.** v7.2 §5.4 already established that confirmation gates flow through the `confirm_action` SOP-scoped tool. The right yolo behavior is: **the tool still fires, but with a configured default response, logged as `synthetic`**.

### §4.2 The new yolo model

**Principle:** In yolo mode, every conversation tool invocation produces a synthetic response according to a per-tool default policy. The tool's `executor.py` is responsible for synthesizing the response (because it knows the tool's response schema); the inferencer is responsible for labeling the turn as `source: synthetic`.

**Per-tool default policy (configurable in `tool.json`):**

| Tool | `yolo_default_response` (new tool.json field) | Synthetic response shape |
|---|---|---|
| `multiple_choice` | `{"strategy": "select_all"}` | `{"selected": [<all option values>]}` |
| `single_choice` | `{"strategy": "first"}` | `{"selected": <first option value>}` |
| `clarification` | `{"strategy": "judgment"}` | `{"response": "Follow your best judgment."}` |
| `confirm_action` | `{"strategy": "confirm"}` | `{"confirmed": true}` |
| `free_text` (if added) | `{"strategy": "judgment"}` | `{"response": "Follow your best judgment."}` |

**Strategies (registered enum, lives in `tools/_shared/yolo_strategies.py`):**
- `select_all` — return all options as selected (for multiple_choice).
- `first` — return the first option (for single_choice).
- `judgment` — return the literal string "Follow your best judgment." (for free-text / clarification).
- `confirm` — return `{confirmed: true}` (for confirmation tools).
- `decline` — return `{confirmed: false}` (anti-default; rarely used).
- `static:<json>` — return a literal JSON payload (escape hatch for SOP authors who want a specific stub answer).

### §4.3 `tool.json` schema extension

Add an optional top-level field to conversation tools:

```jsonc
{
  "name": "multiple_choice",
  "tool_type": "Conversation",
  "category": "conversation",
  "yolo_default_response": {
    "strategy": "select_all"
  },
  "parameters": [...]
}
```

If the field is **absent**, the tool falls back to `{"strategy": "judgment"}` and emits a warning at registry-load time (so SOP authors know they should configure it explicitly).

**DO NOT** put yolo policy inside the SOP phase text. Tool-level configuration keeps the policy uniform across all SOPs that use the tool; per-SOP override happens via the `static:<json>` strategy in phase guidance when truly needed (rare).

### §4.4 Inferencer integration — wire dead `yolo_mode` to synthesize responses (v3: corrected target)

**Critical v3 correction:** v2 said "delete line 682-690." Empirically verified that block does NOT reference `yolo_mode` — it unconditionally strips confirmation-gate phases regardless of yolo mode. **It is an orthogonal bug** (auto-strip happens even in interactive mode). The actual yolo fix is wiring `yolo_mode` into the conversation-tool handler.

**Real fix — wire `self.yolo_mode` (currently dead at `conversational_inferencer.py:111`) into `_handle_conversation_tools` at line 1099** (the compound tool dispatcher; also handles single-tool case at line 1124 via `_handle_conversation_tool` at line 970). At the top of `_handle_conversation_tools`:

```python
async def _handle_conversation_tools(self, tools, assistant_text, ...) -> Optional[dict[str, str]]:
    if self.yolo_mode:
        return self._synthesize_yolo_collected(tools)  # v3 NEW
    # ... existing interactive path unchanged ...
```

**Parallel cleanup (separate ticket):** The block at `conversational_inferencer.py:682-690` (unconditional auto-strip of confirmation-gate phases) is an orthogonal defect. Once yolo synthetic auto-advance lands, that block becomes dead in yolo mode AND is incorrect in interactive mode. Address as its own PR with its own AC; **NOT** part of this plan's scope.

**`render_for_mode` at `sop_manager.py:568`:** Keep as-is for backward compatibility. v2 recommended deletion; v3 retains it because (a) line-682 block depends on the `[__requires confirmation__]` marker that `render_for_mode` strips in yolo, and (b) any external callers (Buck targets, scripts) shouldn't break.

```python
async def _handle_conversation_tool(self, tool, ...):
    if self.yolo_mode:
        synthetic = self._synthesize_yolo_response(tool)
        # Single-record persistence (D6 — no parallel synthetic_trace.json file):
        self._messages.append({
            "role": "user",
            "content": f"[Synthetic auto-advance] {synthetic}",
            "synthetic": True,
            "synthetic_strategy": self._strategy_for(tool),
            "synthetic_tool": tool.name,
        })
        return synthetic
    # ... existing interactive path unchanged ...
```

**v3.1 CRITICAL CORRECTION — synthetic response shape must match real protocol** (Claim 4 / HIGH severity):

`_handle_conversation_tools` (line 1099) returns `Optional[dict[str, str]]` — a flat `{output_variable_name: string_value}` dict assembled from `user_input.get("values", ...)` (verified at line 1187). It does NOT return `{"selected": [...]}` or `{"confirmed": True}` shapes (those would silently misroute).

`_handle_conversation_tool` (singular, line 970) returns `str`. Internal response dict uses these keys (verified at lines 1060-1090):
- `choice_index: int` — selected option index (single_choice, multiple_choice)
- `choice: str` — explicit choice value (confirmation widget)
- `custom_text: str` — free-text override
- `variable_override: str` — when user re-targets the output variable
- `param_overrides: dict` — confirmation widget action-tool parameter overrides

**Synthetic strategies must return the SAME structures**, then let the existing post-processing in `_handle_conversation_tool` map them to the final `str` (single tool) or contribute to the `dict[str, str]` (compound). Concrete strategy outputs:

| Tool type | Strategy | Synthetic raw response (passed through existing post-processing) | Final return value |
|---|---|---|---|
| `MULTIPLE_CHOICE` | `select_all` | `{"choice_index": 0}, {"choice_index": 1}, ...` (one per option) — or `{"custom_text": "<comma-joined option values>"}` | string of comma-joined option values |
| `SINGLE_CHOICE` | `first` | `{"choice_index": 0}` | option[0] value |
| `CONFIRMATION` | `confirm` | `{"choice": "yes"}` | `"yes"` |
| `CLARIFICATION` | `judgment` | `{"custom_text": "Follow your best judgment"}` | `"Follow your best judgment"` |
| `TOOL_ARGUMENT_FORM` | `judgment` | `{"values": {<param>: "Follow your best judgment", ...}}` | dict propagated through compound `collected` |
| (any) | `static:<json>` | `<json>` literal | as-shaped by SOP author |

The synthetic dispatcher MUST NOT invent new keys. AC3 must verify the synthetic-mode dict-shape equality with interactive-mode for the same answer (regression test: replay an interactive log under yolo and assert byte-identical `_messages` output for non-synthetic tools).

```python
# v3.1 CORRECTED synthetic dispatcher signature:
def _synthesize_yolo_collected(self, tools: list[ConversationTool]) -> dict[str, str]:
    """Builds the same {var_name: str_value} dict that the interactive path returns."""
    collected: dict[str, str] = {}
    for tool in tools:
        raw_response = self._yolo_strategy_for(tool).synthesize(tool)
        # Reuse existing per-tool post-processing (do NOT reimplement):
        processed = self._process_widget_response(tool, raw_response)
        collected[tool.output_variable] = str(processed)
    return collected

def _synthesize_response(self, tool_info, arguments):
    policy = tool_info.yolo_default_response or {"strategy": "judgment"}
    strategy = policy["strategy"]
    if strategy == "select_all":
        return {"selected": [opt["value"] for opt in arguments["choices"]]}
    elif strategy == "first":
        return {"selected": arguments["choices"][0]["value"]}
    elif strategy == "judgment":
        return {"response": "Follow your best judgment."}
    elif strategy == "confirm":
        return {"confirmed": True}
    elif strategy == "decline":
        return {"confirmed": False}
    elif strategy.startswith("static:"):
        return json.loads(strategy[len("static:"):])
    else:
        raise ValueError(f"Unknown yolo strategy: {strategy}")
```

### §4.5 Turn record schema extension

Each turn record (created via `on_new_turn` callback at line 321-326) gains a new metadata field:

```jsonc
{
  "turn_number": 7,
  "source": "synthetic",       // NEW — "human" | "synthetic" | "agent"
  "synthetic_strategy": "select_all",   // NEW (only when source == "synthetic")
  "tool_invoked": "multiple_choice",    // NEW (only when source == "synthetic")
  // ... existing fields ...
}
```

This lets a post-hoc reviewer replay an SOP run and see exactly where yolo took shortcuts vs. where the user actually engaged.

### §4.6 Removing the text-strip behavior

The user explicitly said: *"we no longer require yolo mode to remove `[__requires confirmation__]` related instructions from the phase guidance text."* This means the directive text stays in the phase prompt verbatim — the LLM still sees "this phase requires confirmation" and still invokes the confirmation tool — but the tool returns a synthetic answer instead of blocking.

**Audit:** grep for any code that mutates SOP guidance text based on yolo mode. Currently this is the line-682 block above; once deleted, no other stripping logic exists (verified by grep `yolo.*requires\|requires.*yolo` returning only that one site).

### §4.7 Acceptance criteria

- AC4-1: Running `/sop role_creation --yolo` invokes every conversation tool in the SOP and completes end-to-end without blocking.
- AC4-2: Each yolo-mode turn record has `source: "synthetic"` and `synthetic_strategy` set.
- AC4-3: `multiple_choice` tool in yolo mode returns all options selected (verified for Phase 0).
- AC4-4: `confirm_action` tool in yolo mode returns `{confirmed: true}` (verified for Phase 1b).
- AC4-5: Phase guidance text is **unchanged** between yolo and interactive runs (regression: previously yolo stripped `[__requires confirmation__]` markers).
- AC4-6: A conversation tool without `yolo_default_response` falls back to `{"strategy": "judgment"}` AND emits a registry-load warning.
- AC4-7: Re-running an SOP in interactive mode after a yolo run shows exactly which turns were synthetic in the session log.

---

## §5. Concern 4 — First-class SOP resources

### §5.1 Current state vs target state

| Aspect | Current | Target |
|---|---|---|
| Location | `resources/prompt_templates/conversation/main/_variables/workflow_sop/*.md` | `resources/sops/<sop_name>/SOP.md` + `sop.json` (v2: PLURAL `sops` — matches `skills`/`tools`) |
| Discovery | `WorkflowRegistry._default_search_paths()` hardcoded to one path | Glob `resources/sops/*/SOP.md` (mirrors skill pattern) |
| Conversation prompt | SOPs invisible — rendered only as `available_workflows` via `WorkflowManager` (per v7.2) | Available SOPs section (like skills) + Active SOPs section (per active instance) |
| Per-SOP configuration | Only what fits in markdown (description, phase directives, meta-tags) | `sop.json` carries machine-readable config (default modes, default param values, version, tags, related SOPs) |
| OpenStartup parity | Only AgentFoundation has SOPs | Both AgentFoundation (built-in) and OpenStartup (project-specific) have `resources/sops/` (plural) |

### §5.2 New folder structure (v2 — PLURAL `sops`)

```
CoreProjects/AgentFoundation/src/agent_foundation/resources/sops/
├── __init__.py
├── registry.py                  ← Mirrors skills/registry.py and tools/registry.py
└── role_creation/
    ├── SOP.md                   ← The SOP markdown (was workflow_sop/role_creation.md)
    └── sop.json                 ← NEW machine-readable config

CoreProjects/OpenStartup/src/openteam/server/resources/sops/
├── __init__.py
└── <project_specific_sop>/
    ├── SOP.md
    └── sop.json
```

### §5.3 `sop.json` schema

```jsonc
{
  "name": "role_creation",
  "version": "1.0",
  "description_override": null,            // Optional; if null, derived from SOP.md preamble
  "default_mode": "interactive",           // "interactive" | "yolo"
  "supported_modes": ["interactive", "yolo"],
  "default_params": {                      // CLI --params defaults
    "target_path": null
  },
  "tags": ["onboarding", "ai-employee"],   // For grouping in the available-SOPs prompt section
  "related_sops": ["role_update", "role_decommission"],  // Suggested follow-ups
  "max_concurrency": 1,                    // Default for __branch__ fan-out
  "estimated_duration_minutes": 30         // For UI display
}
```

`sop.json` fields are **additive** — `WorkflowDefinition` already has most equivalents (or can derive them from SOP.md). The reason to have a JSON file is two-fold:

1. **Discoverability without parsing markdown.** Future tooling (CLI tab-complete, UI registry browser) can read `sop.json` alone without invoking the full SOP parser.
2. **Machine-editable defaults.** A user can change `default_mode: yolo` without touching the markdown.

**DO NOT** duplicate fields between SOP.md and sop.json. If both specify the description, sop.json wins (via `description_override`); otherwise SOP.md preamble is used.

### §5.4 Discovery — `resources/sops/registry.py`

Mirror `resources/skills/registry.py` structure exactly:

```python
@dataclass(frozen=True)
class SOPInfo:
    name: str
    description: str
    keywords: list[str]
    example_requests: list[str]
    tags: list[str]
    file_path: Path        # path to SOP.md
    config_path: Path      # path to sop.json
    config: dict           # parsed sop.json
    workflow_definition: WorkflowDefinition  # composed via existing WorkflowRegistry

def discover_sops(search_paths: list[Path]) -> dict[str, SOPInfo]:
    """Glob <path>/*/SOP.md across all search_paths; pair with sop.json."""
    ...
```

**Note on the existing `WorkflowRegistry`:** It stays. `SOPRegistry` is a thin wrapper that composes:
- `WorkflowRegistry` for the SOP.md → WorkflowDefinition parsing
- `sop.json` reading for the metadata layer
- A unified `SOPInfo` for prompt rendering

This keeps the workflow-definition logic (v7.2) untouched while adding the resource-layer wrapper.

### §5.5 Conversation prompt rendering

Two new prompt sections (in the conversation Jinja template, mirroring the existing skills section):

**Section A — Available SOPs (always rendered, like skills/tools):**

```
## Available SOPs

You can invoke an SOP via `/sop <name>` (interactive) or `/sop <name> --yolo` (autonomous).
Each SOP runs as a multi-phase workflow.

- **role_creation** [tags: onboarding, ai-employee]
  Creates a new AI employee role end-to-end: gathers responsibility categories via a
  clarifies the role's responsibility categories with the user, researches the…  Use when the user wants to hire, onboard,
  design, or stand up a new AI employee.
  Keywords: hire employee, create role, onboard AI employee, build AI agent, …
  Example: "hire a machine learning engineer", "create a TPM AI role"

- **<other_sop_name>** [tags: …]
  …
```

The `Keywords:` and `Example:` lines come from concern #2's meta-tags.

**Section B — Active SOPs (rendered only when `WorkflowManager` has instances):**

```
## Active SOPs (your current workflows)

You currently have N SOPs in progress. Use `enter_workflow(<instance_id>)` to re-focus
or `complete_phase(...)` to advance the focused instance.

### role_creation (instance abc12345) — FOCUSED
  Status: in Phase 1b (Role Document Review), awaiting confirmation
  Started: 2026-05-26 12:30 UTC by user (interactive mode)
  Next step: <next_step_guidance from v7.2 §6.6>

### code_optimization (instance def67890) — SUSPENDED
  Status: in Phase 2 (Refactoring), suspended at user request
  Started: 2026-05-26 11:00 UTC by user (yolo mode)
  Resume with: resume_workflow(def67890)
```

The FOCUSED entry expands fully (with next-step guidance); SUSPENDED entries collapse to one-line summaries to save tokens.

### §5.6 Migration of existing SOPs

`workflow_sop/role_creation.md` and `workflow_sop/code_optimization.md` (and any others) move to `resources/sops/<name>/SOP.md`. The old `_default_search_paths()` location in `WorkflowRegistry` becomes a **deprecated fallback** (keeps loading SOPs from the old path with a deprecation warning) for one release, then removed.

### §5.7 Acceptance criteria

- AC5-1: `resources/sops/role_creation/{SOP.md, sop.json}` exists; the old `_variables/workflow_sop/role_creation.md` is removed.
- AC5-2: `SOPRegistry.discover_sops()` returns a `SOPInfo` for `role_creation` with all fields populated from both SOP.md and sop.json.
- AC5-3: The conversation prompt contains an "Available SOPs" section listing all discovered SOPs with description + keywords + examples.
- AC5-4: When `WorkflowManager` has ≥1 active instance, prompt also contains "Active SOPs" section with FOCUSED + SUSPENDED groupings.
- AC5-5: `WorkflowRegistry` loading from the old `_variables/workflow_sop/` path emits a deprecation warning but still works for one release.
- AC5-6: OpenStartup `resources/sops/` directory exists (may be empty) with `__init__.py` + `registry.py` re-exporting AgentFoundation's `SOPRegistry`.

---

## §6. Concern 5 — SOP runtime session storage

### §6.1 Layout target (v2 — composes with v5.3 `allocate_tool_workspace`)

Each SOP run is a **conversational session** (per v7.2 §6 — `WorkflowInstance` wraps a `ConversationalInferencer`). It reuses v5.3's `allocate_tool_workspace(name)` primitive — **no new top-level ladder** (D3 — corrects v1's parallel `<server>/sop/` invention):

```
<session_root>/                              ← From v5.3 unified_workspace_allocation
├── tasks/                                    ← Existing (task tool workspaces, per v5.3)
│   └── task_<id>_<TS>_<uuid8>/
└── sops/                                     ← NEW (plural; matches resources/sops/)
    └── role_creation_<TS>_<uuid8>/           ← Allocated via allocate_tool_workspace("sop_role_creation")
        ├── instance.json                     ← WorkflowInstance state snapshot
        ├── workgraph_checkpoint.pkl          ← v7.2 SOPWorkGraph state for resume
        ├── session.jsonl                     ← RankEvolve-style append-only per-turn log (D7)
        ├── tasks/                            ← Task tool workspaces invoked DURING this SOP
        │   └── task_<id>_<TS>_<uuid8>/       ← Same shape as top-level tasks/
        └── turn_001/                         ← One folder per phase execution
        │   └── (streaming cache, prompt snapshots)
        └── turn_002/                         ← e.g., conversation tool — synthetic OR real
        └── turn_003/
```

**`session.jsonl` record schema** (D7 — single append-only log, not split files):

```jsonc
{"type": "TurnStart",       "turn": 7, "phase_id": "1b", "ts": "..."}
{"type": "UserInput",       "turn": 7, "item": "[Synthetic auto-advance] yes", "synthetic": true, "synthetic_strategy": "confirm", "synthetic_tool": "confirm_action"}
{"type": "PromptTemplate",  "turn": 7, "name": "role_creation", "ts": "..."}
{"type": "RenderedPrompt",  "turn": 7, "tokens": 4321, "ts": "..."}
{"type": "InferenceResponse","turn": 7, "content": "...", "ts": "..."}
{"type": "TurnEnd",         "turn": 7, "ts": "..."}
```

Synthetic responses are differentiated **only** by `"synthetic": true` on the `UserInput` record. No parallel `synthetic_trace.json` file. The LLM doesn't see the `synthetic` flag — it's metadata for post-hoc analysis only.

### §6.2 Key design decisions (and rejections)

**Decision 1 — `sop/` lives sibling to `tasks/`, not nested under it.**
Rationale: SOP sessions and standalone task invocations are peer concepts. A task tool invoked *during* an SOP nests under that SOP's `tasks/` subfolder (per layout above). A task tool invoked *outside* an SOP lives at top-level `tasks/`. This matches the existing v5.3 architecture where the `tasks/` folder is server-wide.

**Decision 2 — Each turn gets its own folder.**
The existing `on_new_turn(turn_number, user_input)` callback already creates "turn directories" per `conversational_inferencer.py:172-173, 322-326` — we extend it for SOP sessions to populate the schema above. **No callback signature change needed.**

**Decision 3 — Synthetic turns are first-class, not hidden.**
Synthetic turns produce `turn_<N>/` folders just like human turns (for streaming cache + rendered prompts). Synthetic differentiation is **only** in the session-level `session.jsonl` `UserInput` record via `"synthetic": true` + `"synthetic_strategy"` + `"synthetic_tool"` fields (D6). Re-playing the SOP in interactive mode can grep `session.jsonl` for synthetic records and compare to what the user *would have* answered. **DO NOT** elide synthetic records from `session.jsonl` (regression risk: would hide bugs in yolo strategies). **v3.1 fix:** removed parallel `synthetic_trace.json` reference — single-record persistence per D6.

**Decision 4 — Task tool workspace IS nested under SOP session.**
The user's question: *"Task tool workspace is under the session's tasks folder right?"* — Yes, when invoked from inside an SOP. The `SOPWorkGraphNode` (v7.2 §5.5) passes its own `session_context` down to any tool it invokes, and the v5.3 task tool already respects `session_root`. We need to make `SOPWorkGraphNode` set `session_context["session_root"]` to the SOP session's folder. **One-line change** in `_execute_phase`.

**Decision 5 — Rejected: turns indexed by phase_id only.**
The reviewer might suggest organizing turns under `phases/<phase_id>/turns/` instead of flat `turns/`. Rejected because: (a) re-entering a phase via `__goto__` would conflict on phase_id; (b) BranchBarrierNode aggregation logically belongs to no single phase; (c) flat numbering matches existing conversational session convention.

### §6.3 SOP session ID + folder name

```
sop_session_id  = f"{workflow_id}_{started_at_utc:%Y%m%d_%H%M%S}_{uuid8}"
folder          = <session_root>/sops/sop_{sop_session_id}/   # v3.1: plural sops/ (consistent with D2/§6.1)
```

(Mirrors v5.3 `task_<id>` naming.)

### §6.4 Turn metadata schema

```jsonc
// turns/turn_0007/turn_metadata.json
{
  "turn_number": 7,
  "phase_id": "1b",                         // Which phase produced this turn; null if pre-phase
  "started_at": "2026-05-26T13:42:11Z",
  "ended_at":   "2026-05-26T13:42:13Z",
  "source": "synthetic",                    // "human" | "synthetic" | "agent"
  "user_input_preview": null,               // Only if source == "human"
  "tool_invoked": "confirm_action",         // Only if source == "synthetic"
  "synthetic_strategy": "confirm"           // Only if source == "synthetic"
}
```

### §6.5 Persistence hooks (where to wire)

| Site | What it persists | Code location |
|---|---|---|
| `WorkflowManager.create_instance` | `session_metadata.json` + folder creation | New: `workflow/manager.py` |
| `WorkflowInstance.checkpoint` | `workflow_instance.json` + `workgraph_checkpoint.pkl` | v7.2 §6.4 already specifies; this plan adds the SOP-folder pathing |
| `on_new_turn` callback in `conversational_inferencer.py:322-326` | `turns/turn_<N>/` folder + `turn_metadata.json` (basic fields) | Extend existing callback with `phase_id` from `prior_context` |
| Synthetic tool dispatch (§4.4) | `session.jsonl` UserInput record with `"synthetic": true` + `"synthetic_strategy"` + `"synthetic_tool"` (D6 — no parallel file) | New: in `_handle_conversation_tools` yolo branch |
| `SOPWorkGraphNode._execute_phase` | Set `session_context["session_root"]` to SOP session folder before invoking nested tools | One-line addition (v7.2 §5.5 entry point) |

### §6.6 Cross-plan dependency on `unified_workspace_allocation_INTEGRATED_v5_FINAL_plan.md`

v5.3 already specifies the `<server>/tasks/` layout. This plan adds a sibling `<server>/sop/` and reuses v5.3's allocation primitives. **No conflict** — the two plans compose. **DO NOT** re-litigate v5.3's tasks/ layout decisions in this plan.

### §6.7 Acceptance criteria

- AC6-1: Running `/sop role_creation` creates `<server>/sop/sop_role_creation_<TS>_<uuid8>/` with `session_metadata.json` populated.
- AC6-2: Each turn produces a `turn_NNN/` folder (streaming cache + prompt snapshots only); turn-level events (TurnStart, UserInput, RenderedPrompt, InferenceResponse, TurnEnd) are appended to the SOP-scope `session.jsonl` (D7 — single append-only log, not split files). (v3.1 fix: removed stale `messages.jsonl`/`tool_calls.jsonl` references.)
- AC6-3: A yolo-mode turn's `UserInput` record in `session.jsonl` contains `"synthetic": true`, `"synthetic_strategy": "<strategy_name>"`, `"synthetic_tool": "<tool_name>"`. (v3.1 fix: D6 — no separate `synthetic_trace.json` file; single-record persistence.)
- AC6-4: A task tool invoked during the SOP creates its workspace at `<server>/sop/<sop_session>/tasks/task_<id>/` (not at top-level `tasks/`).
- AC6-5: Re-running the SOP via `resume_workflow(instance_id)` re-opens the same `sop_<...>` folder and continues turn numbering.
- AC6-6: `session_metadata.json.mode` correctly reflects `"interactive"` vs `"yolo"` for the run.

---

## §7. Phased rollout

| Phase | Concern | Deliverable | Effort | LOC est. | Risk |
|---|---|---|---|---|---|
| **0** | All | RED tests for the 6 AC sets (AC2/3/4/5/6 + parser regression) | 0.5 day | ~250 (tests) | LOW |
| **1** | #1 | Edit `role_creation.md` overview (one file, one paragraph) | 5 min | 7 lines | TRIVIAL |
| **2** | #2 | SOPManager parser preamble state + `SOP.meta` field + `WorkflowDefinition.keywords/example_requests` propagation; description sanitization | 1 day | ~80 | LOW |
| **3** | #3 | `tool.json` `yolo_default_response` schema; yolo strategies module; `_dispatch_conversation_tool` synthetic path; delete current line-682 block; turn metadata schema | 1.5 days | ~150 | MEDIUM (touches inferencer dispatch) |
| **4** | #4 | `SOPRegistry`; `sop.json` schema; folder migration of `role_creation`; conversation Jinja template "Available SOPs" + "Active SOPs" sections; deprecation warning on old path | 2 days | ~200 | MEDIUM |
| **5** | #5 | `<server>/sop/` runtime layout; `WorkflowManager.create_instance` folder creation; `on_new_turn` extension; `synthetic_trace.json`; SOP-scoped task tool nesting | 1.5 days | ~180 | MEDIUM |
| **6** | E2E | Run `/sop role_creation --yolo` against the migrated SOP; verify all AC pass; verify replay-in-interactive parity | 0.5 day | — (verification) | LOW |

**Total:** ~7 days; ~870 LOC + ~250 LOC tests.

**Sequencing rationale:** Phase 1 first (zero-risk content edit). Phases 2 and 3 are independent — could parallelize but sequencing 2→3 lets phase 3 tests use the frontmatter-equipped SOP. Phase 4 depends on phase 2 (the registry uses the parsed frontmatter via `_parse_skill_md`). Phase 5 depends on phase 4 (needs the new SOPRegistry to know where to anchor the runtime folder). Phase 6 ties it all together. **v3.1 fix:** replaced "SOPInfo composition uses `sop.meta`" → "registry uses parsed frontmatter via `_parse_skill_md`" (aligns with v3 frontmatter approach).

---

## §8. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Synthetic strategy returns malformed payload for some tool (e.g., wrong key names) → LLM crashes trying to parse | HIGH | Each strategy is bound to the tool's response schema; add a per-tool unit test that runs the synthetic-response through the same JSON-schema validator the tool uses for human responses |
| 2 | Removing the line-682 skip block breaks an existing non-yolo SOP that depended on auto-completion of pure-text confirmation phases | MEDIUM | Audit existing SOPs for "pure-text confirmation" phases (Phase 1b of role_creation IS one); these must invoke `confirm_action` instead of being implicit. Migrate as part of phase 4 |
| 3 | `WorkflowRegistry`'s deprecation fallback (loading from old `_variables/workflow_sop/`) shadows a newly-named `resources/sops/` SOP with the same workflow_id | MEDIUM | Registry deduplicates by `workflow_id`; the new path wins (sort search_paths so new path is first). Emit explicit warning on shadowing |
| 4 | `sop.json` and SOP.md preamble specify conflicting descriptions | LOW | Documented precedence: `sop.json.description_override` wins if set; else SOP.md preamble. Add an AC test for both branches |
| 5 | Turn-folder explosion for long SOPs (1000+ turns) → filesystem pressure | LOW | Each turn folder is small (<10 KB typical); 10k turns = ~100 MB. Acceptable. If becomes an issue: introduce turn-bundle archives at every N turns |
| 6 | Synthetic-vs-human label gets out of sync between turn_metadata and message records | MEDIUM | Single source of truth: the `_dispatch_conversation_tool` synthetic path writes both atomically. Add an AC that asserts label consistency across both records |
| 7 | OpenStartup's `resources/sops/` re-export creates a circular import with AgentFoundation | LOW | Re-export is a one-line `from agent_foundation.resources.sop.registry import SOPRegistry` — no circularity (AgentFoundation doesn't import OpenStartup) |
| 8 | The `[__keywords__]` / `[__example_requests__]` parser preamble state misclassifies a phase tag if it appears unusually early (e.g., user puts `[__initial__]` before `## Phase 0`) | LOW | Parser raises `SOPParseError("Phase-level directive '[__initial__]' found before first '## Phase' heading — directives must follow a phase heading")` |
| 9 | Yolo "select_all" semantics conflict with phase intent (e.g., a multi-choice tool meant to elicit a single primary direction) | MEDIUM | Per-invocation override: SOP author can pass `--yolo-strategy static:{...}` in phase guidance; documented in `sop.json` per-phase override schema (future extension; not required for v1) |
| 10 | Task tool workspace nested under SOP session folder breaks existing reporters that expect top-level `tasks/` | LOW | The reporter glob already accepts `<server>/**/tasks/task_*/` per v5.3 §10 (verified). No change needed |
| 11 | Active-SOPs prompt section grows unbounded with N suspended instances | LOW | Cap at 10 most recent suspended instances; "…and N more older instances" trailing line |
| 12 | Concern #3's deletion of line-682 block lands but concerns #4/#5 don't, leaving the system in an inconsistent state | MEDIUM | Phase ordering enforces atomic landing: phase 3 (yolo) ships with feature flag `OPENTEAM_USE_SYNTHETIC_YOLO=true` (defaults to false); flip to true only after phases 4/5 land |

---

## §9. Open questions

| # | Question | Suggested resolution |
|---|---|---|
| Q1 | Should `[__keywords__]` support multiple lines (continuation) or only single-line comma-list? | Single-line only; multi-keyword sets use multiple `[__keywords__]` lines |
| Q2 | Should `[__example_requests__]` items accept embedded markdown (e.g., links)? | No — plain strings only; rendered as quoted text in the prompt |
| Q3 | Should `sop.json` allow per-phase yolo strategy overrides? | Defer to v2 — not needed for `role_creation`'s 4 phases |
| Q4 | Should synthetic turns count toward conversation history token budget? | Yes — they take real prompt space and the LLM sees them; budget enforcement unchanged |
| Q5 | Should the "Active SOPs" prompt section render in non-conversational inferencers (e.g., a one-shot inferencer)? | No — only when `WorkflowManager` is attached, which is conversational-only |
| Q6 | Should `WorkflowRegistry` (the existing one) be renamed `LegacyWorkflowRegistry` to make the SOPRegistry promotion clearer? | No — `WorkflowRegistry` is the underlying parser; `SOPRegistry` is the resource-layer composer. Keep both names |
| Q7 | What happens if a user invokes `/sop role_creation` while another SOP is already FOCUSED? | Auto-suspend the focused one; focus the new one. (v7.2 §6.5 `enter_workflow` semantics) |
| Q8 | Should the synthetic-trace `strategy` field carry the actual `tool.json.yolo_default_response` payload, or just the strategy name? | Carry the full policy dict — enables exact replay even if `tool.json` changes later |
| Q9 | Does `sop.json` need an Atlassian-internal compass/CODEOWNERS field? | Defer — out of scope for runtime enablement |
| Q10 | Should `confirm_action` (v7.2 §5.4) honor `tool.json.yolo_default_response = {"strategy": "decline"}` to support "fail closed" SOPs? | Yes — supported by design; no extra code |

---

## §10. Closing — relationship to v7.2

v7.2 (`conversational_workflows_and_sop_framework_INTEGRATED_v7.2_plan.md`) specified:
- The SOP runtime architecture (parser, `SOPWorkGraphNode`, `BranchBarrierNode`).
- The conversation tools (`enter_workflow`, `exit_workflow`, `resume_workflow`, `complete_phase`, `confirm_action`, `abort_phase`).
- The WorkflowManager structure.

This plan (sop_runtime_enablement) is what makes the first real SOP — `role_creation.md` — actually run end-to-end on top of that architecture:
- Concern #1 fixes the description so the LLM enters the SOP correctly.
- Concern #2 adds the SOP-level meta-tags v7.2's grammar didn't formalize.
- Concern #3 replaces v7.2's hand-waved yolo mode with a concrete synthetic-response mechanism.
- Concern #4 promotes SOPs from internal-variable status to first-class resources, exposing them in the prompt at the same tier as skills and tools.
- Concern #5 adds the runtime session storage that v7.2 specified at the API level but not the filesystem level.

After this plan lands, the next planning step would be **OpenStartup deployment** — package `role_creation` for the openteam server and wire it into the project-creation UX.

---

## §11. File inventory

| Path | Action | Notes |
|---|---|---|
| `AgentFoundation/src/agent_foundation/resources/prompt_templates/conversation/main/_variables/workflow_sop/role_creation.md` | Move | → `resources/sops/role_creation/SOP.md` (phase 4) |
| `AgentFoundation/src/agent_foundation/resources/sops/__init__.py` | Create | Phase 4 |
| `AgentFoundation/src/agent_foundation/resources/sops/registry.py` | Create | Phase 4 — mirrors skills/registry.py |
| `AgentFoundation/src/agent_foundation/resources/sops/role_creation/SOP.md` | Create | Phase 1 + 4 (refined overview from §2.3) |
| ~~`resources/sops/role_creation/sop.json`~~ | **v3 REMOVED** | YAML frontmatter inside `SOP.md` replaces the `.json` sidecar (matches skill convention via `_parse_skill_md`) |
| `AgentFoundation/src/agent_foundation/resources/tools/tool_argument_form/tool.json` | Create | Phase 3 — currently only the enum variant exists; tool.json missing |
| `AgentFoundation/src/agent_foundation/server/schema.py` | Modify | Phase 3 — add `is_synthetic_default: bool` to `MessageMetadata` (parallel to existing `is_auto_advance`) |
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/tool_call_parser.py` | Modify | Phase 5 — recognize `{"type":"sop","name":"<sop>"}` tool envelope for SOP entry |
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/sop_sub_inferencer.py` | Create | Phase 5 — thin wrapper class; each active SOP is a sub-conversational-session |
| `OpenStartup/src/openteam/server/services/session_store.py` | Modify | Phase 5 — add `get_session_sops_dir(session_id)` helper (parallel to `get_session_tasks_dir`) |
| ~~`_variables/workflow/sop.jinja2`~~ | **Delete (v3)** | Stale; no longer the canonical SOP path |
| ~~`_variables/workflow/.sop.config.yaml`~~ | **Delete (v3)** | Directives subsumed by SOP.md frontmatter `metadata` block |
| ~~`RichPythonUtils/.../template_manager/sop_manager.py`~~ | **v3 NO CHANGE** | Frontmatter is parsed by `_parse_skill_md` before the body reaches SOPManager. Phase body grammar unchanged. |
| ~~`AgentFoundation/.../common/workflow/definition.py`~~ | **v3 NO CHANGE** | `WorkflowDefinition` not extended; routing data lives on `SOPInfo` (registry-side) |
| `AgentFoundation/src/agent_foundation/common/workflow/registry.py` | Modify | Phase 2 — populate new fields; phase 4 — deprecation warning on old path |
| `AgentFoundation/src/agent_foundation/common/workflow/manager.py` | Modify | Phase 5 — create SOP session folder |
| `AgentFoundation/src/agent_foundation/resources/tools/_shared/yolo_strategies.py` | Create | Phase 3 (§4.2 strategies) |
| `AgentFoundation/src/agent_foundation/resources/tools/models.py` | Modify | Phase 3 (v3.1 — Claim 5) — add `yolo_default_response: dict \| None = None` field to `ToolDefinition`; extend `from_dict()` to extract from `tool.json` |
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversation_tools.py` | Modify | Phase 3 (v3.1) — propagate `yolo_default_response` from `ToolDefinition` to `ConversationTool` so the inferencer's `_synthesize_yolo_collected` can read it per-tool |
| `AgentFoundation/src/agent_foundation/resources/tools/multiple_choice/tool.json` | Modify | Phase 3 — add `yolo_default_response: {strategy: select_all}` |
| `AgentFoundation/src/agent_foundation/resources/tools/single_choice/tool.json` | Modify | Phase 3 — add `yolo_default_response: {strategy: first}` |
| `AgentFoundation/src/agent_foundation/resources/tools/clarification/tool.json` | Modify | Phase 3 — add `yolo_default_response: {strategy: judgment}` |
| `AgentFoundation/src/agent_foundation/resources/tools/confirm_action/tool.json` | Modify | Phase 3 (file from v7.2) — add `yolo_default_response: {strategy: confirm}` |
| `AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py` | Modify | Phase 3 — delete line-682 block; add synthetic dispatch path; extend turn metadata. Phase 5 — wire `session_root` to SOP folder |
| `AgentFoundation/src/agent_foundation/resources/prompt_templates/conversation/main/_variables/available_sops.jinja2` | Create | Phase 4 — §5.5 Section A |
| `AgentFoundation/src/agent_foundation/resources/prompt_templates/conversation/main/_variables/active_sops.jinja2` | Create | Phase 4 — §5.5 Section B |
| `OpenStartup/src/openteam/server/resources/sops/__init__.py` | Create | Phase 4 |
| `OpenStartup/src/openteam/server/resources/sops/registry.py` | Create | Phase 4 — re-exports AgentFoundation's SOPRegistry |
| `AgentFoundation/test/test_sop_meta_tags.py` | Create | Phase 0/2 — AC3 tests |
| `AgentFoundation/test/test_sop_yolo_synthetic.py` | Create | Phase 0/3 — AC4 tests |
| `AgentFoundation/test/test_sop_registry.py` | Create | Phase 0/4 — AC5 tests |
| `AgentFoundation/test/test_sop_runtime_storage.py` | Create | Phase 0/5 — AC6 tests |

---
