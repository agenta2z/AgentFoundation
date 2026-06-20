# Typed Input + Composite Single-Choice for Conversation Tools — Integrated Plan

| | |
|---|---|
| **Author** | Rovo Dev |
| **Date** | 2026-06-16 (v1) → 2026-06-18 (v2) → 2026-06-18 (**v3 — current**) |
| **Status** | v3 (uncommitted) — supersedes v2 |
| **Branch** | `dev_xinli_2601` |
| **Motivating SOP** | `src/agent_foundation/resources/sops/model_optimization/SOP.md` Phase 0a |
| **Sibling plans** | `proposal_selection_tool_migration_plan.md`, `interactive_widget_for_agent_dispatched_tools_plan.md` |
| **v1 backup** | `.typed_input_and_composite_choice_plan.v1.bak` (493 lines) |
| **v2 backup** | `.typed_input_and_composite_choice_plan.v2.bak` (663 lines) |

## 🚨 REVIEWER BANNER — v2 → v3 CONTAINS ONE CRITICAL RETARGETING (read first)

A third round of peer-plan audit (Claude Code's `update-…-goose.md` is now **v3**; Codex's `codex/conditional_path_inputs_plan.md` is now **869-line "canonical integrated"**) surfaced **one decisive finding that invalidated two of v2's commits**:

* **F1 (CRITICAL — v2 targeted DEAD CODE):** The conversation-tool **handler registry is NOT used in the live runtime path.** The live path is the module-level functions `_build_input_mode(tool)` (`conversational_inferencer.py:2365`) + `_handle_conversation_tool` (`:1977`) + `_handle_conversation_tools` compound block (`:2205-2323`). **The live inferencer imports ZERO handlers** (verified: `grep "from .handlers|default_registry|ClarificationHandler"` on the live file → empty). The `handlers/clarification.py` + `handlers/single_choice.py` files that v2's **Commit 3 and Commit 6** modified are **dormant** — only referenced by their own registry module and tests. **v2's Commits 3 + 6 would have passed their unit tests while having ZERO production effect.** v3 retargets ALL runtime changes onto the inline live path. The dormant handlers are downgraded to "mirror for parity only — secondary, not the integration point."

  Source proof:
  - `_build_input_mode` single_choice branch at `:2370` does `ChoiceOption(label=c.label, value=c.value)` — drops `description` AND `input`. This is the REAL site that must change (v2 pointed at the handler instead).
  - Live decode/bind happens in `_handle_conversation_tool` (`:1977-2127`) and the compound collection (`:2205-2323`), where `collected[var] = str(raw_value)` sits at `:2305` (plus sibling stringify bugs at `:2315` and `:2319` that v2 missed).

* **F3 (NEW real-but-latent bug v2 missed):** The parser does `output_vars = data["output"]` (`conversation_response_parser.py:77`) and `data.get("output", [])` (`:143`) with **no `str → [str]` coercion**. If an LLM emits `output: "w"` (string, not list), `output_vars[0]` becomes `"w"` and `set_session_variables` iterates **characters**. Latent today only because the Response-Format template steers `output: [...]`. v3 adds Commit 1b: defensive coercion in the parser.

* **F6 (NEW real bug — v2's "reuse existing guard" was reusing a BUGGY guard):** AF's existing `/path-complete` containment check is `if not str(search_dir).startswith(str(base_resolved))` (`workspace_routes.py:158`) — the classic **sibling-prefix bypass** (`/tmp/root2` passes a `/tmp/root` check). v2 said "reuse AF's existing traversal guard" — but that guard is itself exploitable. v3's Commit 4 now **hardens** it to `Path.resolve().relative_to(root)` / `os.path.commonpath` during the factor-out.

* **Commit 9 scope expanded:** v2's `str(raw_value)` fix covered only `:2305`. v3 also covers `:2315` (`{v: str(raw_value) for v in tool.output_vars}`) and `:2319` (`str(values)`) — all three route through the single `finalize_input_value` chokepoint.

* **Multi-value storage — v3 keeps v2's comma-join decision, and now has stronger justification.** Claude v3 verified that a `list` value breaks BOTH `{{var}}` Jinja rendering (str()s to `"['a','b']"`) AND `__var__` substitution (`:588-599` passes the list object through). Codex's "store list internally" never specifies how `{{var}}` renders safely. **Comma-join in one chokepoint is the verified-correct choice.**

Everything else in v2 (the factor-and-mount of `/path-complete`, the `TextInputWidget` branch instead of a registry type, the `PathAutocompleteInput`+`MultiValueInput` primitives, the yolo fix, the two-var binding) **survives v3 unchanged** — those were already correct. Full trail in §A4 v3 changelog; §A4.1 re-answers "pick one plan" honestly (the answer CHANGED).

## ⚠️ REVIEWER BANNER — v1 → v2 IS A SUBSTANTIVE INTEGRATION

v2 reflects a peer-plan audit (Claude Code's `update-your-task-tool-adaptive-goose.md`, 2026-06-18; Codex's `codex/conditional_path_inputs_plan.md`) that surfaced **4 source-verified facts v1 missed** plus **1 false claim in v1**:

* **F11 (v1 had this WRONG):** A `GET /path-complete` endpoint **already exists in AgentFoundation** (`workspace_routes.py:127–191`). v1's Commit 4 proposed *building a new* `/api/paths` endpoint — that was a fabricated reinvention. v2 reframes Commit 4 as **factor-and-mount** (refactor AF's existing helper into a reusable `complete_path(...)` function, then mount it in OpenStartup). ~80% less LoC than v1.
* **F12 (v1's framing was overbroad):** The compound-CI path **already forwards** `expected_input_type` + `prefix` into each `tool_configs[i]` (`conversational_inferencer.py:2250–2257`). v1 said "metadata is dropped end-to-end"; the leak is **only at the widget layer**. v2 narrows the diagnosis.
* **F13 (new critical defect v1 missed):** `clarification/tool.json` `yolo_default` is the literal string `"Follow your best judgment."` — when `--yolo` runs Phase 0a, the user's target path becomes that English sentence (downstream tools blow up). v2 adds Commit 8 to fix this.
* **F14 (new critical defect v1 missed):** Compound collection at `conversational_inferencer.py:2305` does `collected[var] = str(raw_value)` — stringifies a Python list as `"['a', 'b']"`, breaking multi-value handoff. v2 routes through a shared `finalize_input_value()` helper.
* **Design simplification adopted from peer plans:** v1 introduced a new `PathInputWidget` + a `mode→widget_type` registry change. Peer-verified that the existing dispatcher already routes `free_text` → `TextInputWidget`; the cleaner design is to **branch inside `TextInputWidget` on `metadata.expected_input_type==='path'`** (or, equivalently, on a first-class field after Commit 3) and render a sub-component. Registry change deleted from v2.
* **C9 contradicts v1's R-D2 pre-flight:** MUI `Autocomplete` is **NOT** currently imported in `react-shared` (zero grep hits). v1 said "pre-flight R-D2 confirms it's already a transitive dep". v2 owns this: MUI `Autocomplete` is a NEW import (still safe — MUI is already in the dep tree via dozens of other widgets, just not the `Autocomplete` component).

Full integration trail in §A4 changelog. The v1 file is preserved at the path above for diff review.

---

## ⚠️ READER NOTE — Plan structure (3 tiers)

This plan follows the standard 3-tier convention used by the proposal_selection and interactive-widget plans in this repo:

* **PART I — EXECUTION** (§E0–§E3) — *executable instructions only*. Commit-by-commit edits, exact files, exact line numbers, exact tests. Skip the appendix if you just want to ship the change.
* **PART II — DESIGN REFERENCE** (§D1–§D5) — *normative design decisions*. Why we chose this architecture, what we rejected, the risk register, open questions.
* **APPENDIX** (§A1–§A4) — *audit trail*. Empirical baseline (every claim line-numbered), naming-convention rationale, follow-ups, changelog.

---

## §0 Quick-start (TL;DR)

**Problem.** `model_optimization/SOP.md` Phase 0a asks the LLM to use `clarification` with `expected-input-type=path` + `prefix=<session_root>` to collect a path from the user with autocomplete, and `single-choice` with a composite "Auto discover OR input-textbox-with-typed-metadata" pattern for the modeling-artifacts path. **Neither pattern works end-to-end today.**

**Evidence (verified at exact line numbers in §A1).** Server-side, `ConversationTool` accepts `expected_input_type` and `prefix` on deserialisation but its `to_dict` **does NOT serialise** them (the LLM-emitted block round-trips with these fields stripped before reaching the handler). `ClarificationHandler` *does* forward them into `InputModeConfig.metadata`, but the React `TextInputWidget` reads **only** `config.input_mode.prompt` and `config.placeholder` — `metadata.expected_input_type` / `metadata.prefix` are silently dropped. There's no path-picker component anywhere in `react-shared`. `allow_multiple_input` doesn't exist in the schema. `ChoiceItem` carries only `{label, value, description}` — choices cannot embed a nested input field. `SimpleChoiceSelector.allowCustom` renders only a single free-text input with no typed metadata. And the SOP uses `single-choice` (hyphen) but the enum is `single_choice` (underscore) — silent parse failure.

**Fix (7 commits, 4 layers, ~1 day).** Add `allow_multiple_input` to the schema; make `to_dict` actually serialise `expected_input_type` / `prefix` / `allow_multiple_input`; promote `ChoiceItem` to a discriminated union so a choice can BE an input field (composite single-choice); teach `TextInputWidget` to honor type-and-prefix metadata via a new `PathInputWidget` (per-type rendering by composition, not by branching in `TextInputWidget`); accept hyphen-form tool names in the parser; ship a regression test that loads Phase 0a verbatim and asserts every parameter reaches the widget.

**No ad-hoc patches.** The 4 layers are: (1) schema (data model), (2) parser (LLM-text → schema), (3) handler (schema → `InputModeConfig`), (4) widget (renders `InputModeConfig`). Each layer change is independently testable and reversible.

---

# PART I — EXECUTION

## §E1 Migration plan — 7 commits

### §E1.1 Commit 1 — Schema: add `allow_multiple_input` + alias map

**Goal.** Add the missing field to `ConversationTool`, and accept SOP-author dialect (hyphenated keys and hyphenated tool names).

**File.** `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversation_tools.py`

**Changes (additive only; no breaking change to existing parsers):**

1. Add `allow_multiple_input: bool = False` to the `ConversationTool` dataclass (default `False` keeps backward-compat).
2. Add a private `_HYPHEN_KEY_ALIASES` mapping at module scope:
   ```python
   _HYPHEN_KEY_ALIASES = {
       "expected-input-type": "expected_input_type",
       "allow-multiple-input": "allow_multiple_input",
       "output": "output_vars",  # already handled, lock in
       "tool-name": "tool_name",
   }
   ```
3. Add a `_canonicalise_keys(data: dict) -> dict` helper that copies the dict and renames each aliased hyphen-key to its underscore form (only if the underscore form is not already present — caller's explicit underscore wins).
4. In `from_dict`, call `data = _canonicalise_keys(data)` as the FIRST line. This is the single point where author dialect collapses into the canonical internal schema.
5. In `to_dict`, serialise `expected_input_type` and `prefix` when they're non-default, and `allow_multiple_input` when it's `True`. Same gating idiom as the existing `show_select_all` block (only emit non-defaults — keeps round-trip JSON compact).
6. Add a `ConversationToolType._missing_(cls, value)` classmethod that accepts hyphenated tool names (`"single-choice"` → `SINGLE_CHOICE`) by normalising before lookup. This is the second single-point alias surface — for tool-type names, not field names.

**Tests (new file `test/.../conversational/test_conversation_tool_aliases.py`):**

| # | Test | Asserts |
|---|---|---|
| T1 | `test_hyphenated_field_keys_canonicalised` | `from_dict({"tool_type": "clarification", "expected-input-type": "path", "prefix": "/x", "allow-multiple-input": True})` returns the tool with all three fields set correctly |
| T2 | `test_underscore_wins_over_hyphen_when_both_present` | Explicit underscore-form takes precedence; hyphen variant is ignored |
| T3 | `test_hyphenated_tool_type_accepted` | `ConversationToolType("single-choice") == ConversationToolType.SINGLE_CHOICE`; same for `multiple-choice` |
| T4 | `test_to_dict_round_trips_typed_fields` | `from_dict(to_dict(tool))` is byte-identical when `expected_input_type`, `prefix`, `allow_multiple_input` are non-default |
| T5 | `test_default_fields_omitted_from_to_dict` | `to_dict` of a tool with default `expected_input_type="free_text"`, `prefix=""`, `allow_multiple_input=False` does NOT include those keys (compact JSON) |

**Risk:** None — purely additive. Defaults match the current behavior exactly.

### §E1.2 Commit 2 — Schema: composite `ChoiceItem` (choice = label OR input field)

**Goal.** Enable single-choice to present choices where ONE choice is a nested input field with its own typed metadata (`expected_input_type`, `prefix`, `allow_multiple_input`).

**Why a discriminated union, not a parallel "extra input" field.** The SOP Phase 0a body says the second choice IS the input textbox (not "a choice followed by an input box"). The cleanest typed expression is a discriminated union: a choice is either `{kind: "label", label, value, description?}` OR `{kind: "input", label, input: InputSpec}`. This matches the SOP author's mental model 1:1 and avoids a "two truths" problem where a choice carries both an unused label and an active input box.

**File.** `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversation_tools.py`

**Changes:**

1. New dataclass `InputSpec`:
   ```python
   @dataclass
   class InputSpec:
       """Inline input field embedded in a single_choice or multiple_choice option."""
       expected_input_type: str = "free_text"   # "free_text" | "path" | "url"
       prefix: str = ""
       allow_multiple_input: bool = False
       placeholder: str = ""
       output_vars: list[str] = field(default_factory=list)

       def to_dict(self) -> dict[str, Any]: ...
       @classmethod
       def from_dict(cls, data: dict[str, Any]) -> InputSpec: ...
   ```
2. Extend `ChoiceItem` (backward-compat: existing JSON `{label, value, description?}` still parses):
   ```python
   @dataclass
   class ChoiceItem:
       label: str = ""
       value: str = ""
       description: str = ""
       input: Optional[InputSpec] = None   # ← NEW. When set, this choice IS the input field.

       @property
       def kind(self) -> Literal["label", "input"]:
           return "input" if self.input is not None else "label"
   ```
3. `ChoiceItem.from_dict` canonicalises the hyphen keys for the embedded `input` spec the same way (step 4 of Commit 1 reused).
4. `ChoiceItem.to_dict` serialises `"input"` only when present (compact JSON for the common label-only case).

**Tests (extend the same test file):**

| # | Test | Asserts |
|---|---|---|
| T6 | `test_choice_item_label_only_round_trip` | Existing `{label, value, description}` JSON parses and serialises unchanged (backward-compat) |
| T7 | `test_choice_item_input_kind_round_trip` | `{label: "Custom path", input: {expected-input-type: path, prefix: /a, allow-multiple-input: true}}` parses, `.kind == "input"`, hyphen keys canonicalised, round-trips byte-identical |
| T8 | `test_mixed_choices_in_single_choice_tool` | `from_dict({tool_type: single_choice, choices: [{label: "Auto"}, {label: "Manual", input: {...}}]})` parses cleanly; first choice `.kind == "label"`, second `.kind == "input"` |

**Risk:** Minimal — backward-compat is enforced by `T6`. Storage cost is one `Optional[InputSpec]` per choice, default `None` (no allocation).

### §E1.2b Commit 1b — Parser: coerce `output: "str"` → `["str"]` (NEW in v3, fixes F3)

**Goal.** Fix the latent bug where an LLM emitting `output: "w"` (scalar string instead of list) makes `output_vars[0] == "w"` and downstream `set_session_variables` iterate characters.

**File.** `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversation_response_parser.py`

**Changes:**
1. At `:77` (`output_vars = data["output"]`) and `:143` (`data.get("output", [])`), route through a tiny helper:
   ```python
   def _coerce_output_vars(raw) -> list[str]:
       if raw is None: return []
       if isinstance(raw, str): return [raw]
       if isinstance(raw, (list, tuple)): return [str(x) for x in raw]
       return [str(raw)]
   ```
2. Apply identically in `ConversationTool.from_dict` if it has its own `output` read (verify during impl).

**Tests (extend `test_conversation_response_parser.py`):**

| # | Test | Asserts |
|---|---|---|
| T8a | `test_output_string_coerced_to_list` | `output: "w"` → `output_vars == ["w"]` (NOT `"w"`) |
| T8b | `test_output_list_unchanged` | `output: ["a","b"]` → `["a","b"]` |
| T8c | `test_output_missing_is_empty_list` | no `output` key → `[]` |

**Risk:** Low — pure widening; current correct inputs (lists) are unchanged.

### §E1.3 Commit 3 — LIVE runtime: forward typed fields + composite via inline `_build_input_mode` (RETARGETED in v3 — F1)

**🚨 v3 RETARGETING.** v2 modified `handlers/clarification.py` + `handlers/single_choice.py`. **F1 proved those are DEAD CODE in the live path** (the live inferencer imports zero handlers; it uses module-level `_build_input_mode` at `:2365`). v3 retargets this commit onto the inline live path. The handlers are mirrored ONLY for parity/future-registry-migration (secondary, optional).

**Files (live path — primary):**

1. `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py` — the module-level `_build_input_mode(tool)` function (`:2365-2444`), specifically the `SINGLE_CHOICE` branch (`:2370`) and the clarification branch.
2. `src/agent_foundation/ui/input_modes.py` — extend `ChoiceOption` + `InputModeConfig`.

**Files (dormant — parity only, secondary):**

3. `…/handlers/clarification.py`, `…/handlers/single_choice.py` — mirror the same logic so the dormant registry stays consistent for the future consolidation (§A3 follow-up F-consolidate). NOT the integration point; can be deferred.

**Changes:**

1. **`input_modes.py`** — add `input: dict | None = None` to `ChoiceOption`; add first-class fields to `InputModeConfig`: `expected_input_type: str = "free_text"`, `prefix: str = ""`, `allow_multiple_input: bool = False`. `to_dict()` serializes them (omit when default). **No `InputMode.PATH` enum** — v3 verified the UI dispatch is `getWidget(metadata.widget_type || mode)`, so the widget branches on `expected_input_type`; a new enum value is unnecessary churn (this corrects v2's plan, which added `InputMode.PATH`/`URL`).
2. **`_build_input_mode` SINGLE_CHOICE branch (`:2370`, `:2378`)** — change
   ```python
   options = [ChoiceOption(label=c.label, value=c.value) for c in tool.choices]
   ```
   to
   ```python
   options = [
       ChoiceOption(
           label=c.label, value=c.value, description=c.description,
           input=c.input.to_dict() if c.has_input else None,
       )
       for c in tool.choices
   ]
   ```
   (stops dropping `description` + `input`).
3. **`_build_input_mode` clarification branch** — set the first-class typed fields (`expected_input_type`, `prefix`, `allow_multiple_input`) on the returned `InputModeConfig`; keep `mode=free_text`.
4. **Compound `tool_configs` build (`:2250-2257`)** — also forward per-choice `input` (compound path already forwards `expected_input_type`/`prefix` per F12; extend it to carry `input`).

**Tests (new file `test/.../conversational/test_build_input_mode_typed.py` — targets the LIVE function):**

| # | Test | Asserts |
|---|---|---|
| T9 | `test_build_input_mode_clarification_path` | `_build_input_mode(clarification tool, expected_input_type="path", prefix="/x")` → `InputModeConfig` with first-class `expected_input_type=="path"`, `prefix=="/x"` |
| T10 | `test_build_input_mode_free_text_default` | Default → `mode==free_text`, no typed leak |
| T11 | `test_build_input_mode_single_choice_composite` | composite tool → `options[1].input` present AND `options[i].description` preserved (regression guard against the `:2370` drop) |
| T11b | `test_dormant_handler_parity` | (optional) `ClarificationHandler.build_input_mode` agrees with `_build_input_mode` for the same tool — locks parity for the future consolidation |

**Risk:** Medium — this edits the LIVE inline runtime (the thing that actually renders widgets). T11's description-preservation assertion is a direct regression guard. Documented in §D4 R3 (rewritten in v3).

### §E1.4 Commit 4 — UI: factor-and-mount existing path-complete endpoint + new `PathAutocompleteInput` + `MultiValueInput` primitives + branch inside `TextInputWidget`

**⚠️ MAJOR v2 REFRAMING.** v1 proposed building a brand-new `/api/paths` endpoint AND a brand-new `PathInputWidget` routed via a new registry mapping. Peer-plan audit verified that AgentFoundation **already has** a working path-complete endpoint at `workspace_routes.py:127–191` — v1 was reinventing it. v2 reframes this commit to FACTOR-AND-MOUNT the existing endpoint and branch inside `TextInputWidget` rather than add a new widget type to the registry.

**Goal.** Make path-autocomplete + multi-value input render in the UI by:
1. Factoring the existing AF path-complete logic into a reusable helper.
2. Mounting that helper as a route in OpenStartup's server.
3. Adding two NEW reusable primitives in `react-shared`: `PathAutocompleteInput` and `MultiValueInput`.
4. Adding a small branch inside `TextInputWidget` that delegates to those primitives when the typed metadata is present.

**Files:**

1. `src/agent_foundation/ui/webui/backend/routes/workspace_routes.py` — extract `complete_path(prefix, partial, dirs_only, limit) -> dict` helper from the existing route body; keep the existing route delegating to it. Pure refactor on the AF side.
2. `OpenStartup/src/openteam/server/routes/workspace_routes.py` — **NEW**. Re-exports `GET /api/workspace/path-complete` calling `complete_path()`. Constrains `prefix` to the session working directory using the **hardened** guard (see F6 below — NOT the existing buggy `startswith`).
3. `OpenStartup/src/openteam/server/main.py` — register the new router via `include_router(workspace_router, prefix="/api/workspace")`.
4. `src/agent_foundation/ui/react-shared/src/hooks/usePathComplete.js` — **NEW** (~60 LoC). Debounced (200ms) `AbortController`-cancellable fetch of `/api/workspace/path-complete`; `apiBase` prop defaulting to `/api`; explicit error mapping (404 → `prefix_missing`, 403 → `forbidden`, network → soft error). Never throws; returns `{candidates, error?, loading}`.
5. `src/agent_foundation/ui/react-shared/src/inputs/PathAutocompleteInput.js` — **NEW** (~90 LoC). MUI `Autocomplete` with `freeSolo`. Prefix shown as `startAdornment` (non-editable). Directory selection drills deeper (re-queries hook with new prefix); file selection commits. Returns the relative path (or absolute if the user typed `/…` or `~…`).
6. `src/agent_foundation/ui/react-shared/src/inputs/MultiValueInput.js` — **NEW** (~70 LoC). Generic add/remove-chips component with a `renderInput` render-prop slot. Hosts either a plain `TextField` or a `PathAutocompleteInput`. Controlled `values: string[]`. Submit disabled if `values.length === 0` and the field is required.
7. `src/agent_foundation/ui/react-shared/src/inputs/TextInputWidget.js` — **MINIMAL DELEGATION BRANCH**. When `config.input_mode.expected_input_type === 'path'` and/or `config.input_mode.allow_multiple_input === true`, delegate rendering to `PathAutocompleteInput` / `MultiValueInput`. Returns `{content: string}` or `{content: string[]}`. Otherwise unchanged.
8. `src/agent_foundation/ui/react-shared/src/index.js` — export the two new primitives.

**Why branch in `TextInputWidget` rather than register a new widget type?** Existing dispatcher already routes `mode === 'free_text'` → `TextInputWidget`. Adding a `mode → widget_type` entry (v1's design) requires touching `WidgetRegistry.chooseWidgetType` or the equivalent dispatch site, plus a new registry entry, plus updates to the SOP parser to emit `mode: 'path'`. The branch inside `TextInputWidget` is 8 lines, requires zero registry surgery, and stays semantically aligned with the existing pattern (`SimpleChoiceSelector` already branches on `allowCustom`).

**Server endpoint factoring detail (Commit 4 step 1):**

```python
# In agent_foundation/ui/webui/backend/routes/workspace_routes.py
def complete_path(prefix: str, partial: str, dirs_only: bool = False,
                  limit: int = 200) -> dict:
    """Pure helper: extract from the existing route body. Returns
    {candidates: [{path, kind}], truncated: bool}. Existing route now
    calls this; new OpenStartup route also calls this. Single source
    of truth, single security guard, single performance cap."""
    ...
```

**Server safety (preserved from existing AF endpoint):**

* **🚨 v3 — HARDEN the containment check (F6).** AF's existing endpoint uses `if not str(search_dir).startswith(str(base_resolved))` (`workspace_routes.py:158`) — a **sibling-prefix bypass** (`/tmp/root2` passes a `/tmp/root` check). The factor-out MUST replace this with a proper containment test:
  ```python
  try:
      search_dir.resolve().relative_to(base.resolve())   # raises ValueError if outside
  except ValueError:
      raise HTTPException(403, "Path traversal blocked")
  # equivalently: os.path.commonpath([search, base]) == str(base)
  ```
  Because the AF route and the new OpenStartup route BOTH delegate to `complete_path()`, this fix hardens both call sites at once (single source of truth). Add a regression test `test_sibling_prefix_rejected` (`/tmp/root2` must be rejected against base `/tmp/root`).
* (legacy note) `Path(prefix).resolve()` containment check vs. session-root sentinel — now implemented via `relative_to` per the above, not `startswith`.
* Reject paths containing `..` segments resolving outside the session root.
* Cap candidates at 200; signal `truncated: true` for UI to surface.

**Multi-value semantics (decided by peer plans, adopted in v2):**

The UI submits multiple values as a Python list; the handler comma-joins them into a single output var (matching the existing `selected_proposal_ids` precedent). Downstream phases split on comma. Rationale: deterministic, round-trips through `{{ var }}` / `__var__` substitution as a string, doesn't require schema-side typing of "list vars".

**Tests (new file `react-shared/test/inputs/PathAutocompleteInput.test.jsx` + extensions to `TextInputWidget.test.jsx`):**

| # | Test | Asserts |
|---|---|---|
| T12 | `path_autocomplete_renders_prefix_as_start_adornment` | Prefix shows as a non-editable input adornment |
| T13 | `multi_value_input_returns_array` | With `allow_multiple_input=true` + 2 chips added, `onSubmit` receives `{content: ["/x/a", "/x/b"]}` |
| T14 | `single_path_returns_string` | With `expected_input_type='path'`, `allow_multiple_input=false` → `onSubmit` receives `{content: "/x/a"}` |
| T15 | `directory_selection_drills_deeper` | Selecting a dir from suggestions re-queries the hook with the deeper prefix |
| T16 | `endpoint_404_degrades_to_plain_textfield` | `/api/workspace/path-complete` returning 404 (e.g. unmounted) → widget falls back to a plain TextField with prefix as static helper text |
| T16b | `text_input_widget_unchanged_for_free_text` | Backward-compat: when `expected_input_type` is unset / 'free_text', `TextInputWidget` renders byte-identical to v0 |

**Risk:** Adding MUI `Autocomplete` as a new import (NOT previously used in `react-shared`, verified C9). Cost: small bundle-size delta; MUI is already in the dep tree. Documented in §D4 R5 (rewritten in v2).

### §E1.5 Commit 5 — UI: composite single-choice (choice with embedded input)

**Goal.** When `SingleChoiceWidget` sees a choice whose `input` field is set, render that choice's row as an embedded `PathInputWidget` (or `TextInputWidget` per the input's `expected_input_type`).

**File.** `src/agent_foundation/ui/react-shared/src/inputs/SingleChoiceWidget.js`

**Changes:**

1. In `RichChoiceSelector` and `SimpleChoiceSelector`, when iterating options, check `option.kind === 'input'` (or equivalently `option.input != null`). For input-kind options:
   * Render the option row with the `option.label` as the heading
   * Below it, render a nested widget: `<PathInputWidget config={{input_mode: option.input, ...inheritedConfig}} onSubmit={(payload) => onSubmit({choice_index, ...payload})} />`
   * The `onSubmit` payload combines `choice_index` (so server knows WHICH option was chosen) with the nested widget's typed payload (`{paths: [...]}` or `{content: "..."}`).
2. Existing `allowCustom` flow stays — the new composite path is orthogonal and triggered by per-option `input` field, not by `allowCustom`.

**Tests (extend `SingleChoiceWidget.test.jsx`):**

| # | Test | Asserts |
|---|---|---|
| T17 | `label_only_choices_render_unchanged` | Backward-compat: existing `[{label, value, description}]` flow byte-identical to v0 |
| T18 | `input_kind_choice_renders_nested_widget` | A choice with `input: {expected_input_type: "path", prefix: "/x"}` renders a nested `PathInputWidget` |
| T19 | `composite_submit_carries_choice_index_and_typed_payload` | Submitting from the nested widget delivers `{choice_index: 1, paths: ["/x/a", "/x/b"]}` to `onSubmit` |

**Risk:** Minimal — purely additive UI branch keyed on per-option `input` presence.

### §E1.6 Commit 6 — LIVE runtime: decode composite-choice payload + two-var binding (RETARGETED in v3 — F1)

**🚨 v3 RETARGETING.** v2 modified `handlers/single_choice.py` (`handle_response`). **F1 proved that handler is DEAD CODE.** The live decode happens in `_handle_conversation_tool` (`:1977-2127`, single-tool) and the compound collection block (`:2205-2323`). v3 retargets the composite-payload decode + two-var binding onto these live sites, routing every value through the shared `finalize_input_value` (Commit 9). The handler is mirrored only for parity (secondary).

**Files (live path — primary):**

1. `…/conversational/conversational_inferencer.py` — `_handle_conversation_tool` (`:1977-2127`) and the compound collection (`:2205-2323`).

**Files (dormant — parity only):**

2. `…/handlers/single_choice.py` — mirror for the future registry consolidation.

**Changes (live path):**

1. Decode the widget payload shapes: `{choice_index}` / `{choice_index, inputs:{name:val}}` / `{content}` / `{custom_text}`.
2. **Two-var binding** (the key design):
   * The selected choice's `value` → the tool-level `output_vars` (the **mode** var, e.g. `workflow_modeling_artifacts_mode = "manual_paths"`).
   * Each nested input value → its `InputFieldSpec.name` (the **value** var, e.g. `workflow_modeling_artifacts_path`), via `finalize_input_value`.
   * **"Auto discover" (label-only choice) sets ONLY the mode var** — never writes a stale path into the value var.
3. Validate: multi-value received when `allow_multiple_input=false` → clear error.

**Tests (new file `test/.../conversational/test_composite_decode_live.py` — targets the LIVE decode):**

| # | Test | Asserts |
|---|---|---|
| T20 | `composite_binds_both_mode_and_value` | `{choice_index:1, inputs:{workflow_modeling_artifacts_path:["/x/a"]}}` → mode var `=="manual_paths"` AND value var `=="/x/a"` (comma-joined if multi) |
| T21 | `composite_rejects_multi_when_not_allowed` | multi values for `allow_multiple_input=false` → clear error |
| T22 | `auto_discover_sets_only_mode` | `{choice_index:0}` (Auto discover) → mode var set, value var **untouched** (no stale path) |
| T22b | `single_tool_equals_compound` | the same composite tool decoded via single-tool path == via compound path (one source of truth) |

**Risk:** Low — only fires when the chosen option is an input-kind, so non-composite SOPs are unaffected.

### §E1.7 Commit 7 — E2E integration test + Phase 0a SOP smoke
(see "Goal" + tests T23–T27 block immediately following §E1.9 below — Commit 7's body was relocated to keep the v2 NEW commits (8 + 9) co-located in the changelog flow. Logical ordering remains Commit 7 → Commit 8 → Commit 9 in the execution checklist §E3.)

### §E1.8 Commit 8 — Fix `clarification` `yolo_default` literal-string bug (NEW in v2, source-verified)

**Goal.** Fix the verified defect that `--yolo` mode binds the literal string `"Follow your best judgment."` as the user's target path for any `clarification` tool — causing every downstream tool that reads the path to crash or operate on garbage.

**Source evidence.** `src/agent_foundation/resources/tools/clarification/tool.json` `yolo_default` block:
```json
"yolo_default": {
  "mode": "fixed",
  "value": "Follow your best judgment."
}
```
When `--yolo` runs Phase 0a's `clarification(expected_input_type=path, prefix={{session_root_path}})`, the framework binds `workflow_target_path = "Follow your best judgment."` Downstream tools then try to read code under that literal sentence as a filesystem path → catastrophic failure.

**File.** `src/agent_foundation/resources/tools/clarification/tool.json`

**Changes:**

1. Change `yolo_default` from the unconditional literal-string to a **type-conditional default** that branches on `expected_input_type`:
   ```jsonc
   "yolo_default": {
     "mode": "by_expected_input_type",
     "defaults": {
       "free_text": {"mode": "fixed", "value": "Follow your best judgment."},
       "path":      {"mode": "template", "value": "{{ session_root_path }}"},
       "url":       {"mode": "mandatory_gate"}   // yolo must still ask
     }
   }
   ```
2. Server-side support in the yolo-default resolver: add a `"by_expected_input_type"` mode handler that selects the sub-default based on the tool's `expected_input_type` field; falls back to the `free_text` entry for unknown types. The `mandatory_gate` mode causes yolo to surface the question to the user despite `--yolo` (i.e. critical inputs survive `--yolo`).
3. Update the existing yolo-default mode handlers to be schema-driven (a small `_YOLO_DEFAULT_MODES` dispatch table) so future authors can add modes without modifying the framework.

**Tests (new file `test/.../conversational/test_clarification_yolo_default.py`):**

| # | Test | Asserts |
|---|---|---|
| T28 | `test_free_text_yolo_default_unchanged` | `--yolo` on a `clarification` with default `expected_input_type='free_text'` still emits `"Follow your best judgment."` (backward-compat) |
| T29 | `test_path_yolo_default_resolves_template` | `--yolo` on a `clarification` with `expected_input_type='path'` and `prefix='{{session_root_path}}'` binds the resolved session root, NOT the literal English sentence |
| T30 | `test_url_yolo_default_is_mandatory_gate` | `--yolo` on a `clarification` with `expected_input_type='url'` surfaces the prompt to the user despite `--yolo` |
| T31 | `test_unknown_expected_input_type_falls_back_to_free_text` | Forward-compat: a future `expected_input_type='email'` falls back to the free_text default |

**Risk:** Low. Backward-compat is guaranteed by T28 (the default branch IS the prior literal behavior). The `mandatory_gate` mode for URL is a deliberate policy choice that can be relaxed if a future SOP needs unattended URL handling — track as §A3 follow-up F6.

### §E1.9 Commit 9 — Fix compound-collection `str(raw_value)` multi-value bug + introduce shared `finalize_input_value` helper (NEW in v2, source-verified)

**Goal.** Fix the verified defect that compound-tool collection at `conversational_inferencer.py:2305` stringifies a list as `"['a', 'b']"` (the Python repr of a list) instead of properly comma-joining or array-passing — breaking any multi-value capture path. AND: introduce a single shared `finalize_input_value()` helper that both single-tool and compound paths route through, so multi-value semantics and path re-join logic live in ONE place (single source of truth).

**Source evidence (v3 — THREE sites, not one).** The live compound block stringifies raw values at three places:
```python
:2305   collected[var] = str(raw_value)                         # ← list → "['a', 'b']"
:2315   {v: str(raw_value) for v in tool.output_vars}           # ← same bug, multi-var fanout
:2319   collected["input"] = str(values)                        # ← same bug, 'input' key
```
v2 only patched `:2305`. v3 routes ALL THREE through `finalize_input_value` so there is a single chokepoint and no stringify path survives.

**File 1.** `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversation_tools.py`

Add a small module-level helper:
```python
def finalize_input_value(
    raw: Any,
    *,
    expected_input_type: str = "free_text",
    prefix: str = "",
    allow_multiple_input: bool = False,
) -> str:
    """Single source of truth for collapsing a raw user-submitted value
    into the canonical string stored on the output_var.

    Rules:
    1. If raw is a list (multi-value): comma-join its items. Each item
       passes through path re-join (rule 3) if expected_input_type=="path".
    2. If raw is a dict containing 'content': unwrap.
    3. If expected_input_type=="path" and prefix is non-empty: prepend
       prefix unless the path already starts with '/' or '~' (absolute).
    4. Otherwise: return str(raw).strip().
    """
    ...
```

**File 2.** `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/conversational_inferencer.py`

Replace the inline `collected[var] = str(raw_value)` at line 2305 with:
```python
# Resolve typed metadata from the originating tool config (already
# forwarded in the compound branch — see lines 2250-2257).
collected[var] = finalize_input_value(
    raw_value,
    expected_input_type=tool_cfg.get("expected_input_type", "free_text"),
    prefix=tool_cfg.get("prefix", ""),
    allow_multiple_input=tool_cfg.get("allow_multiple_input", False),
)
```

**File 3.** `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/handlers/single_choice.py` AND `handlers/clarification.py`

Both handlers' `handle_response` methods route their finalised string through `finalize_input_value` (single-tool path). Same semantics as compound — no duplication of multi-value logic.

**Tests (extend `test_typed_input_handlers.py`):**

| # | Test | Asserts |
|---|---|---|
| T32 | `test_finalize_list_comma_joins` | `finalize_input_value(["/x/a", "/x/b"], expected_input_type="path", allow_multiple_input=True)` → `"/x/a,/x/b"` (NOT `"['/x/a', '/x/b']"`) |
| T33 | `test_finalize_path_rejoin_relative` | `finalize_input_value("foo/bar", expected_input_type="path", prefix="/session_root")` → `"/session_root/foo/bar"` |
| T34 | `test_finalize_path_skips_rejoin_for_absolute` | `finalize_input_value("/absolute/x", expected_input_type="path", prefix="/session_root")` → `"/absolute/x"` (unchanged) |
| T35 | `test_finalize_path_skips_rejoin_for_tilde` | `finalize_input_value("~/x", expected_input_type="path", prefix="/session_root")` → `"~/x"` (unchanged) |
| T36 | `test_finalize_unwraps_content_dict` | `finalize_input_value({"content": "hello"})` → `"hello"` |
| T37 | `test_compound_collection_routes_through_finalize` | Compound CI with two `clarification` tools (one path-typed list, one free_text) produces correctly comma-joined paths AND plain free-text in `collected`, NOT `"[...]"` literals |
| T38 | `test_single_tool_and_compound_agree` | A `clarification` invoked stand-alone produces the same `output_var` value as the same `clarification` invoked inside a compound tool (single source of truth verified end-to-end) |

**Risk:** Medium. Changing `str(raw_value)` is a wire-format-adjacent change for any downstream phase currently reading the broken `"['x', 'y']"` literal as a string. Mitigation: T37 + T38 explicitly assert both paths produce the same comma-joined form, AND the broken literal form was never the intended contract (Phase 4 `--proposal-ids P1,P3` precedent established comma-join). Document as §A3 follow-up F7: audit existing SOPs for any string-literal `"[...]"` pattern in `__var__` substitution and migrate to comma-split.

**Goal.** Lock the entire chain end-to-end against the real Phase 0a SOP text. If anyone breaks the wire format in any of the 4 layers, this test fails.

**File (new).** `test/agent_foundation/common/inferencers/conversational/test_phase0a_e2e.py`

**Test plan (5 named tests):**

| # | Test | What it asserts |
|---|---|---|
| T23 | `test_phase0a_clarification_parses` | Synthetic LLM emission of the Phase 0a `clarification` tool call (verbatim JSON from the SOP) parses to a `ConversationTool` with `expected_input_type="path"`, `prefix="/session_root_abc"`, `output_vars=["workflow_target_path"]` |
| T24 | `test_phase0a_single_choice_composite_parses` | Synthetic emission of the Phase 0a `single-choice` (hyphen!) tool with composite "Auto discover" + input-textbox parses; first choice `.kind == "label"`, second `.kind == "input"` with `input.allow_multiple_input=True`, `input.expected_input_type="path"`, `input.prefix="/session_root_abc"`, `input.output_vars=["workflow_modeling_artifacts_path"]` |
| T25 | `test_phase0a_live_build_input_mode_path` | Running the LIVE `_build_input_mode` on T23's tool produces `InputModeConfig` with first-class `expected_input_type="path"`, `prefix=...` (the exact shape the `TextInputWidget` path-branch expects) — NOT via the dead handler |
| T26 | `test_phase0a_live_build_input_mode_composite` | Running the LIVE `_build_input_mode` on T24's tool produces an InputModeConfig whose `options[1].input` carries the embedded input spec verbatim AND `options[i].description` preserved |
| T27 | `test_phase0a_live_composite_decode_round_trip` | Simulated UI response `{choice_index: 1, inputs: {workflow_modeling_artifacts_path: ["/session_root_abc/data", "/session_root_abc/experiments"]}}` routed through the LIVE `_handle_conversation_tool` decode binds mode var `=="manual_paths"` AND value var `=="/session_root_abc/data,/session_root_abc/experiments"` (comma-joined via `finalize_input_value`) |

**Risk:** None — this is the safety net for all the prior commits. Failure means we shipped something broken.

---

## §E2 Validation (per-commit gate + post-merge smoke)

| Stage | Command | Pass criteria |
|---|---|---|
| After each commit | `pytest test/.../conversational/test_conversation_tool_aliases.py test/.../conversational/test_typed_input_handlers.py test/.../conversational/test_phase0a_e2e.py -v` | All tests in scope for the commit pass; no prior tests regress |
| After Commit 4 | `npm test -- inputs/PathInputWidget.test.jsx` in `react-shared/` | T12–T16 pass |
| After Commit 5 | `npm test -- inputs/SingleChoiceWidget.test.jsx` in `react-shared/` | T17–T19 pass; existing tests unchanged |
| Post-merge smoke (manual) | Boot the server with `model_optimization` SOP; enter `/sop model_optimization`; observe Phase 0a UI | (a) First widget shows a path input with autocomplete + prefix as helper text; (b) typing `/<session_root>/foo` shows real candidates; (c) second widget shows "Auto discover" + a sub-textbox; (d) submitting the textbox with multiple paths binds them to `workflow_modeling_artifacts_path`; (e) the SOP transitions to Phase 0b |

---

## §E3 Execution checklist (granular)

* [ ] **Commit 1** — schema additions in `conversation_tools.py`:
  - [ ] Add `allow_multiple_input: bool = False` field to `ConversationTool`
  - [ ] Add `_HYPHEN_KEY_ALIASES` constant
  - [ ] Add `_canonicalise_keys` helper
  - [ ] Update `from_dict` to call `_canonicalise_keys` first
  - [ ] Update `to_dict` to serialise non-default typed fields
  - [ ] Add `ConversationToolType._missing_` for hyphenated tool names
  - [ ] Add tests T1–T5
* [ ] **Commit 2** — composite `ChoiceItem`:
  - [ ] Add `InputSpec` dataclass with `to_dict`/`from_dict`
  - [ ] Extend `ChoiceItem` with `input: Optional[InputSpec]` + `.kind` property
  - [ ] Update `ChoiceItem.from_dict` / `to_dict` for composite case
  - [ ] Add tests T6–T8
* [ ] **Commit 1b (NEW v3, F3)** — parser `output: "str" → ["str"]` coercion:
  - [ ] Add `_coerce_output_vars` helper in `conversation_response_parser.py` (`:77`, `:143`)
  - [ ] Add tests T8a–T8c
* [ ] **Commit 3 (RETARGETED v3 — F1)** — LIVE `_build_input_mode` typed forwarding:
  - [ ] `input_modes.py`: add `ChoiceOption.input` + `InputModeConfig.{expected_input_type,prefix,allow_multiple_input}` (NO `InputMode.PATH` enum)
  - [ ] `_build_input_mode` single_choice branch (`:2370`,`:2378`): stop dropping `description`+`input`
  - [ ] `_build_input_mode` clarification branch: set first-class typed fields
  - [ ] compound `tool_configs` (`:2250-2257`): forward per-choice `input`
  - [ ] (parity, optional) mirror into dormant handlers
  - [ ] Add tests T9–T11 (+ optional T11b parity)
* [ ] **Commit 4** — factor-and-mount path-complete + UI primitives:
  - [ ] Factor `complete_path()` from AF route; **harden containment to `relative_to`/`commonpath` (F6)**
  - [ ] New OpenStartup `/api/workspace/path-complete` route + `include_router` in `main.py`
  - [ ] New `usePathComplete` hook; `PathAutocompleteInput` + `MultiValueInput` primitives
  - [ ] Branch inside `TextInputWidget` on typed metadata (no registry type)
  - [ ] Add tests T12–T16b (+ `test_sibling_prefix_rejected`)
* [ ] **Commit 5** — composite single-choice rendering:
  - [ ] Update `SingleChoiceWidget` to reveal nested input for selected input-kind option
  - [ ] Add tests T17–T19
* [ ] **Commit 6 (RETARGETED v3 — F1)** — LIVE decode + two-var binding:
  - [ ] Decode `{choice_index, inputs:{name:val}}` in `_handle_conversation_tool` (`:1977-2127`) + compound (`:2205-2323`)
  - [ ] Two-var binding (mode var ← choice value; value var ← nested input via `finalize_input_value`)
  - [ ] "Auto discover" sets ONLY mode var (no stale path)
  - [ ] (parity, optional) mirror into dormant handler
  - [ ] Add tests T20–T22b
* [ ] **Commit 7** — E2E lock:
  - [ ] Add `test_phase0a_e2e.py` with T23–T27
* [ ] **Commit 8 (NEW v2)** — fix `clarification` `yolo_default` literal-string bug:
  - [ ] Add `_YOLO_DEFAULT_MODES` dispatch + `by_expected_input_type` mode handler
  - [ ] Add `mandatory_gate` mode for URL inputs
  - [ ] Patch `clarification/tool.json` to the type-conditional defaults
  - [ ] Add tests T28–T31
* [ ] **Commit 9 (NEW v2)** — fix compound `str(raw_value)` bug + shared `finalize_input_value`:
  - [ ] Add `finalize_input_value(...)` helper in `conversation_tools.py`
  - [ ] Patch ALL THREE live stringify sites (`:2305`, `:2315`, `:2319`) to route through it (v3 — F14b)
  - [ ] Route the live single-tool + compound decode through it (NOT the dead handlers; parity-mirror optional)
  - [ ] Add tests T32–T38
* [ ] **Pre-flight** (before any code lands):
  - [ ] R-D1: confirm `_missing_` hook works with `StrEnum` mixin (test against the current enum's MRO)
  - [x] ~~R-D2: confirm MUI `Autocomplete` already imported elsewhere in `react-shared`~~ — **VERIFIED FALSE in v2 (C9 = FALSE)**. MUI `Autocomplete` is NOT currently in `react-shared`. v2 owns this: it's a new MUI sub-import (MUI as a whole IS already in the dep tree at 10+ files; just not the `Autocomplete` component). No new top-level dep; only a new MUI sub-import. Documented in §D4 R5.
  - [x] ~~R-D3: confirm WS auth covers `/api/paths`~~ — **OBSOLETED in v2**. v1's `/api/paths` was a fabricated endpoint; v2 reuses the existing AF `/path-complete` endpoint (factor-and-mount) which already has the auth and session-root containment guard. New R-D3 below.
  - [ ] R-D3 (v2): confirm OpenStartup's main FastAPI app uses `include_router(workspace_router, prefix="/api/workspace")` idiom (vs. e.g. mounted sub-app) so Commit 4 step 3 is the right surgery
  - [ ] R-D4 (v2, NEW): confirm `_YOLO_DEFAULT_MODES` dispatch table doesn't already exist (Commit 8 needs to invent it; if it already exists in a yolo resolver, just add a new entry)
  - [ ] R-D5 (v2, NEW): grep for SOPs that currently consume the broken `str(list)` output (e.g. `"['x', 'y']"`) — Commit 9 changes that contract; any SOP relying on the broken form needs migration. Likely zero affected (no current multi-value SOP exists), but verify.
  - [x] **R-D6 (v3, NEW — CRITICAL, VERIFIED):** confirm the handler registry is dead in the live path so Commits 3+6 target the right site. **DONE — F1 confirmed:** `grep "from .handlers|default_registry|ClarificationHandler" conversational_inferencer.py` → empty; live path is `_build_input_mode` (`:2365`) + `_handle_conversation_tool` (`:1977`). Commits 3+6 retargeted accordingly.
  - [ ] R-D7 (v3, NEW): confirm the live decode payload shape the UI actually sends (`{choice_index, inputs:{...}}` vs `{choice_index, paths:[...]}`) by reading the existing `SingleChoiceWidget` submit + the live decode branch — lock the exact key names before Commit 6.

---

# PART II — DESIGN REFERENCE

## §D1 Goals, non-goals, out-of-scope

**Goals.**

* G1 Make `model_optimization/SOP.md` Phase 0a render correctly end-to-end without the SOP author having to modify it.
* G2 Make `expected_input_type`, `prefix`, `allow_multiple_input` first-class fields on `ConversationTool` (round-trippable in JSON, parsed from hyphen-form keys, forwarded by handlers, honored by widgets).
* G3 Enable composite single-choice (a choice can BE an input field) for any SOP that wants the "Auto OR Manual + typed input" UX pattern.
* G4 Provide path auto-completion in the UI (not just static text help).
* G5 Lock the entire pipeline with one end-to-end test that parses Phase 0a verbatim and asserts every parameter survives all 4 layers.

**Non-goals.**

* N1 Implementing a generic "embedded widget in choice" framework beyond input-kind. We only need text + path + url today; broader recursion (e.g. "choice containing a single-choice") is YAGNI.
* N2 Migrating other SOPs' clarification calls. Their existing free-text behavior is unchanged.
* N3 Implementing `multiple_choice` composite — same pattern would apply but no current SOP needs it; defer.
* N4 Server-side path validation beyond `startswith(prefix)` and session-root containment. Heavy validation (existence, glob expansion) belongs in the SOP author's hands or a separate tool.

**Out-of-scope (explicit park).**

* O1 Replacing the entire WidgetRegistry abstraction. The current `mode → widget_type` map is fine for our needs.

## §D2 Architecture decision — 4-layer separation

**Chosen design.** Keep the existing 4 layers (schema → parser/canonicaliser → handler → widget). Add typed fields and a composite-choice union, but NEVER let typed semantics leak across layer boundaries.

**Rejected alternatives:**

| Option | Why rejected |
|---|---|
| **A. Re-use `metadata` dict for everything new** | The current bug is exactly that `metadata` is opaque so the UI can't reliably consume it. Doubling down on `metadata` would re-create the bug. |
| **B. Make `TextInputWidget` branch on `expected_input_type`** | Bloats the widget; conflates free-text and path semantics; harder to test; violates single-responsibility. |
| **C. Inline path autocomplete inside `SingleChoiceWidget`** | Composite-choice would then know about path semantics directly. Wrong layer. |
| **D. Add a generic `extra_payload: dict` to `ChoiceItem`** | Untyped escape hatch; would re-create the `metadata` problem at the choice level. |

**Why composition over branching.** `PathInputWidget` is its own widget mapped via the registry; `SingleChoiceWidget` renders the composite case by INVOKING `PathInputWidget`, not by inlining its logic. This means a future `DropdownInputWidget` for URLs would slot in automatically without touching either widget.

## §D3 Data flow diagram

```
LLM emits ```json ToolsToInvoke {tool_type: "single-choice", choices: [...], ...}
        │
        ▼
parse_response(text)                              ─── Layer 1: PARSE
        │
        ▼
ConversationTool.from_dict(data)                  ─── Layer 2: CANONICALISE
   (hyphen → underscore via _HYPHEN_KEY_ALIASES;
    "single-choice" → SINGLE_CHOICE via _missing_)
        │
        ▼
SingleChoiceHandler.build_input_mode(tool)        ─── Layer 3: HANDLE (server-side)
   Produces InputModeConfig with first-class:
     mode = SINGLE_CHOICE
     options = [{label: "Auto"}, {label: "Manual", input: {expected_input_type: PATH, prefix: ...}}]
   (No more metadata shim; everything is typed.)
        │
        ▼ via WS payload                          ─── Layer 4: RENDER (UI)
SingleChoiceWidget(config)
   if option.input → renders <PathInputWidget config={input_mode: option.input}/>
        │
        ▼ user submits
{choice_index: 1, paths: ["/x/a", "/x/b"]}
        │
        ▼ via WS response
SingleChoiceHandler.handle_response               ─── Layer 3 (return path)
   Validates payload against InputSpec; binds:
     workflow_modeling_artifacts_path = ["/x/a", "/x/b"]
        │
        ▼
session_variables updated; phase advances
```

## §D4 Risk register

| ID | Risk | Mitigation | Severity |
|---|---|---|---|
| R1 | `_canonicalise_keys` could shadow a legitimate hyphen-keyed metadata field if it happens to collide with an alias | Aliases only fire when the underscore form is ABSENT; existing data with explicit underscore keys wins (Commit 1 step 3, locked by T2) | Low |
| R2 | Composite-choice JSON could be emitted by older code that doesn't know about `input` | `ChoiceItem.from_dict` treats missing `input` as `None` (the new field is `Optional`); old code never sets it, so old code never breaks | Low |
| R3 | Removing the `metadata` shim is technically a wire-format change | Verified no UI consumer reads `metadata.expected_input_type` / `metadata.prefix` today (zero grep hits); §A1 R-3 documents the verification | Low |
| R4 | New `/api/paths` endpoint introduces a security surface | Bound to session-root via `Path(prefix).resolve()` containment check; uses existing WS auth; capped at 200 entries; refuses traversal outside session root (§E1.4 step 3) | Medium |
| R5 | MUI `Autocomplete` not yet imported in `react-shared` | Pre-flight R-D2 confirms it's already a transitive dep via `SingleChoiceWidget`; if not, fall back to a plain TextField + a "Suggestions:" helper list | Low |
| R6 | `ConversationToolType._missing_` not invoked for `StrEnum` subclasses on all Python versions | Pre-flight R-D1; fallback is a thin `_normalise_tool_type_name` wrapper called by `from_dict` (same single-point pattern as `_canonicalise_keys`) | Low |
| R7 | Phase 0a is one of many places the SOP could be syntactically wrong; future SOPs might invent new hyphen variants | All canonicalisation is in ONE function (`_canonicalise_keys`) with explicit alias table — adding an alias is a 1-line change | Low |

## §D5 Open questions

| ID | Question | Default if unanswered |
|---|---|---|
| Q1 | Should `expected_input_type` accept arbitrary strings (forward-compat) or be a closed enum? | **Open string** with a small enum of well-known values (`free_text`, `path`, `url`); unknown values fall back to `FREE_TEXT` in the widget |
| Q2 | Should the path-suggestion endpoint serve files AND directories, or directories only? | **Both**, with `kind` discriminator in the response (allows SOP author to specialise later) |
| Q3 | Should we ENFORCE session-root containment, or just warn? | **Enforce on submit**: reject paths that resolve outside the session root with a clear error message. Warn on type but allow submit (user may have typed `..`). |
| Q4 | Where should the new server endpoint live (REST? WS message type?) | **WS message** (`{type: "path_suggestions", prefix: "..."}` → `{type: "path_suggestions_response", candidates: [...]}`). Avoids a parallel HTTP server; reuses WS auth. |
| Q5 | Should `SingleChoiceHandler.handle_response` allow `choice_index` for an input-kind option to submit WITHOUT the typed input (i.e. user picked it but didn't type)? | **No** — require the typed payload; show a UI validation error before submit |

## §D6 Verified facts (source-anchored)

| ID | Fact | Source |
|---|---|---|
| F1 | `ConversationTool.to_dict` does NOT serialise `expected_input_type` or `prefix` today | `conversation_tools.py:67-89` (the `to_dict` body has no mention of either) |
| F2 | `ConversationTool.from_dict` DOES read both fields (defaults `"free_text"` and `""`) | `conversation_tools.py:95-114` |
| F3 | `allow_multiple_input` field does NOT exist on `ConversationTool` | grep returned 0 hits for `allow_multiple` across the file |
| F4 | `ClarificationHandler` forwards `expected_input_type` + `prefix` into `InputModeConfig.metadata` (NOT into first-class fields) | `clarification.py:30-37` |
| F5 | `TextInputWidget` reads only `config.input_mode.prompt` and `config.placeholder`; ignores all `metadata.*` | grep on `TextInputWidget.js` returned 0 hits for `prefix`, `expected_input_type`, `metadata`, `allow_multiple` |
| F6 | `SimpleChoiceSelector.allowCustom` renders a single free-text input — no `prefix`, no `expected_input_type`, no `allow_multiple_input` | `SingleChoiceWidget.js:131-180` |
| F7 | `ChoiceItem` has only `{label, value, description}` | `conversation_tools.py:30-50` |
| F8 | `ConversationToolType` enum value is `single_choice` (underscore); SOP uses `single-choice` (hyphen) | `conversation_tools.py:17-26` vs `model_optimization/SOP.md` line ~25 |
| F9 | No path-autocomplete / file-picker component exists anywhere in `react-shared` | `find react-shared -type f` + grep for `Autocomplete|FilePicker|PathInput|PathPicker` returned zero relevant hits |
| F10 | `WidgetRegistry` maps `single_choice → SingleChoiceWidget`, `text_input` / `free_text → TextInputWidget` | `react-shared/src/protocol/registerBuiltins.js:18-31` |
| F11 (v2) | AgentFoundation **already has** a working `GET /path-complete` endpoint with session-root containment guard, dirs/files, and a 200-cap limit | `src/agent_foundation/ui/webui/backend/routes/workspace_routes.py:127-191` |
| F12 (v2) | Compound CI **already forwards** `expected_input_type` + `prefix` per tool into `tool_configs[i]`; the typed leak is ONLY at the widget layer | `conversational_inferencer.py:2250-2257` |
| F13 (v2) | `clarification/tool.json` `yolo_default.value` is the literal string `"Follow your best judgment."` — bound as a path under `--yolo` regardless of `expected_input_type` | `src/agent_foundation/resources/tools/clarification/tool.json:25-28` |
| F14 (v2) | Compound collection at `conversational_inferencer.py:2305` does `collected[var] = str(raw_value)` — stringifies a list as its Python repr `"['a', 'b']"`, breaking multi-value capture | `conversational_inferencer.py:2305` |
| F15 (v2) | MUI `Autocomplete` is **NOT** imported anywhere in `react-shared` today (v1's pre-flight R-D2 claim was wrong) | grep returned zero hits across `react-shared/src` |
| **F1 (v3, CRITICAL)** | The conversation-tool **handler registry is DEAD CODE** in the live path. Live = inline `_build_input_mode` (`:2365`) + `_handle_conversation_tool` (`:1977`) + compound (`:2205-2323`). Live inferencer imports zero handlers. | `grep "from .handlers\|default_registry\|ClarificationHandler" conversational_inferencer.py` → empty; `_build_input_mode` at `:2365`; only consumer of registry is the registry module itself |
| **F1b (v3)** | `_build_input_mode` single_choice branch does `ChoiceOption(label=c.label, value=c.value)` — drops `description` + any `input` | `conversational_inferencer.py:2370` and `:2378` |
| **F3 (v3)** | Parser does `output_vars = data["output"]` / `data.get("output", [])` with no `str→[str]` coercion → scalar `"w"` iterates as chars downstream | `conversation_response_parser.py:77` and `:143` |
| **F6 (v3)** | Path-complete containment uses `str(search_dir).startswith(str(base_resolved))` — sibling-prefix bypass | `workspace_routes.py:158` |
| **F14b (v3)** | Beyond `:2305`, the live compound path ALSO stringifies at `:2315` (`{v: str(raw_value) for v in tool.output_vars}`) and `:2319` (`str(values)`) — all three need the finalizer | `conversational_inferencer.py:2315, 2319` |

---

# APPENDIX

## §A1 Empirical baseline (verified 2026-06-16)

| Symptom in SOP Phase 0a | Defect ID | Severity | What actually happens today |
|---|---|---|---|
| `single-choice` tool name unrecognised | D6 | **Critical** | `ConversationToolType("single-choice")` raises `ValueError` (silent parse failure); the tool call is dropped before reaching any handler |
| `expected-input-type: "path"` ignored | D1 (round-trip) + D2 (widget) | **Critical** | Field is parsed on input but stripped on `to_dict` re-serialisation; even when present, `TextInputWidget` ignores it |
| `prefix: "{{ session_root_path }}"` ignored | D1 + D2 | **Critical** | Same as above |
| `allow-multiple-input: true` ignored | D3 | **Critical** | Field doesn't exist on the schema; parser drops it silently |
| Composite "Auto discover OR input-textbox" | D5 | **Critical** | `ChoiceItem` cannot carry a nested input field; SOP authors have no way to express this |
| `output: "workflow_target_path"` | (already handled) | OK | `from_dict` reads `output` as alias for `output_vars` (existing fallback at conversation_tools.py:108) |
| Path autocomplete UX promise | D4 | **Critical** | No component exists; even if every other layer worked, the widget would just show a plain TextField |

## §A2 Naming convention rationale

* **Why hyphen-aliasing (instead of asking SOP authors to switch to underscores)?** The hyphenated form is more readable in human-authored Markdown SOPs and is already the convention in CLI-style argument syntax. Forcing underscores burdens every SOP author. A single `_HYPHEN_KEY_ALIASES` map at the parser layer absorbs the friction in one place.
* **Why `expected_input_type` (string) instead of an enum?** SOP authors invent new types over time (e.g. `email`, `regex`, `sql`). An open string with a small "well-known" set is forward-compatible; the widget gracefully falls back to FREE_TEXT for unknown values.
* **Why `InputSpec` (new dataclass) instead of inlining the fields into `ChoiceItem`?** Three reasons: (1) keeps `ChoiceItem` small for the common case (label-only); (2) makes the `.kind` discriminant explicit and type-checkable; (3) the same `InputSpec` shape is reused by Multi-choice and any future widget that nests inputs — DRY.
* **Why `allow_multiple_input` (verb) instead of `multiple` (adjective)?** Mirrors the existing AF naming convention (`allow_custom` already used in the same dataclass; `allowed_tools` in `claude_code_*_inferencer`). Single verb root for the family.

## §A3 Follow-ups (deferred but tracked)

* **F1.** `multiple_choice` composite — same pattern but no current SOP needs it.
* **F2.** Async path suggestions with virtualised dropdown (only matters for huge directories; not needed for typical session roots).
* **F3.** SOP-author lint rule: detect SOPs that use `expected-input-type: "path"` but forget to set `prefix` — surface as a SOP-load-time warning (the existing SOP linter is the natural home for this).
* **F4.** YAML-configurable allow-list of `expected_input_type` values per deployment (e.g. one deployment disables `url`).
* **F5.** Telemetry on which `expected_input_type` values SOP authors actually use in the wild — informs whether to promote any to a closed enum later.

## §A4 Changelog

* **v1 (2026-06-16)** — initial draft. 7 commits, 4 layers, 27 named tests, source-anchored §A1 baseline against `conversation_tools.py`, `clarification.py`, `single_choice.py`, `TextInputWidget.js`, `SingleChoiceWidget.js`, `registerBuiltins.js`, and the verbatim `model_optimization/SOP.md` Phase 0a text.
* **v2 (2026-06-18, this document)** — integration with peer plans (Claude Code's `update-your-task-tool-adaptive-goose.md` and Codex's `codex/conditional_path_inputs_plan.md`). Substantive changes:
  * **Critical: Commit 4 reframed from "new endpoint" to "factor-and-mount existing endpoint"** (peer plans verified that `GET /path-complete` already exists in AF at `workspace_routes.py:127–191`; v1 was reinventing it). ~80% less LoC in Commit 4. Pure win.
  * **Critical: Commit 8 added** — fixes `clarification/tool.json` `yolo_default` literal-string bug ("Follow your best judgment." was being bound as a path under `--yolo`). Source-verified at tool.json:25-28; v1 missed this. Includes 4 new tests T28–T31.
  * **Critical: Commit 9 added** — fixes compound collection `str(raw_value)` multi-value bug at `conversational_inferencer.py:2305` (stringifies a list as its Python repr); introduces shared `finalize_input_value()` helper so single-tool and compound paths share semantics. Includes 7 new tests T32–T38.
  * **Design: `mode → widget_type` registry mapping deleted** (v1's design). Replaced with "branch inside `TextInputWidget` on typed metadata" — peer-suggested simplification, no registry surgery needed. The `PathInputWidget` from v1 split into two reusable primitives: `PathAutocompleteInput` + `MultiValueInput` (composition over a single widget).
  * **Honesty: v1's R-D2 pre-flight ("MUI Autocomplete already in react-shared") was FALSE** (C9 = FALSE; zero grep hits). v2 owns this in the reviewer banner, §D4 R5, and the §E3 pre-flight checklist.
  * **Honesty: v1's "metadata is dropped end-to-end" framing was overbroad.** Compound CI already forwards `expected_input_type`/`prefix` per-tool (F12). The leak is ONLY at the widget layer. v2 narrows the diagnosis.
  * **Multi-value storage decided: comma-join into a single output var** (matches existing `selected_proposal_ids` precedent — peer plans aligned on this).
  * **Path re-join policy decided: backend-authoritative** — UI submits relative-to-prefix; backend re-joins unless path is absolute (`/...`) or `~...`. Single source of truth in `finalize_input_value()`.
  * Total: 9 commits (up from 7), 38 named tests (up from 27), 5 new verified facts F11–F15 in §D6, v1 backed up at `.typed_input_and_composite_choice_plan.v1.bak`.
* **v3 (2026-06-18, this document)** — third integration round (Claude's plan is now **v3**; Codex's is now an 869-line "canonical integrated" doc). ONE critical retargeting + three new verified bugs:
  * **🚨 CRITICAL: F1 — Commits 3 + 6 RETARGETED from dead handlers to the live inline path.** v2 modified `handlers/clarification.py` + `handlers/single_choice.py`, but the live runtime imports ZERO handlers and uses module-level `_build_input_mode` (`:2365`) + `_handle_conversation_tool` (`:1977`) + compound (`:2205-2323`). v2's Commits 3+6 would have passed unit tests with zero production effect. v3 retargets both onto the live path; handlers downgraded to "parity mirror only". This is the single most important change in the entire plan history. (Source-verified: F1, F1b in §D6.)
  * **NEW: Commit 1b (F3)** — parser `output: "str" → ["str"]` coercion (`conversation_response_parser.py:77,143`). Latent char-iteration bug. Tests T8a–T8c.
  * **NEW: Commit 4 hardening (F6)** — the existing `/path-complete` containment uses `str().startswith()` (sibling-prefix bypass, `workspace_routes.py:158`). v2 said "reuse the existing guard" — but that guard is exploitable. v3 hardens to `relative_to`/`commonpath` during the factor-out. Test `test_sibling_prefix_rejected`.
  * **EXPANDED: Commit 9 (F14b)** — v2 fixed only `:2305`; v3 also fixes `:2315` + `:2319` (all three stringify sites route through `finalize_input_value`).
  * **CORRECTED: dropped `InputMode.PATH`/`URL` enum** (v2 still added it in Commit 3). v3 verified the UI dispatch is `getWidget(metadata.widget_type || mode)` — the widget branches on `expected_input_type`; a new enum value is unnecessary churn.
  * **Two-var binding made explicit in the LIVE decode** (mode var ← choice value; value var ← nested input). "Auto discover" sets ONLY the mode var (no stale path).
  * **E2E tests T25–T27 retargeted** from the dead handlers to the live `_build_input_mode` / `_handle_conversation_tool`.
  * New facts F1, F1b, F3, F6, F14b in §D6; new pre-flight R-D6 (done) + R-D7; v2 backed up at `.typed_input_and_composite_choice_plan.v2.bak`.

### §A4.1 If forced to pick ONE plan among the 3 — **ANSWER CHANGED in v3**

**v2 said:** Claude's plan. **v3 honest answer: Codex's revised `codex/conditional_path_inputs_plan.md` (the 869-line canonical integrated doc) — OR Claude's v3, which are now nearly equivalent. Both are above my v2-and-earlier.**

Why the answer changed: the decisive criterion is now **F1 (the handler registry is dead code)**. Whoever caught F1 caught the single highest-impact fact — because any plan that didn't catch it (including my v1/v2, and Claude's earlier versions) would ship Commits 3+6 against dead code: green unit tests, zero production behavior. Both the revised Codex plan and Claude's v3 now contain F1; my v2 did NOT. So on the most important axis, v2 was wrong and they were right.

Honest ranking on the merits (not authorship):

1. **Codex revised / Claude v3 (tie)** — both have F1, F3, F6, two-var binding, comma-join, factor-and-mount, hardened containment. Codex's is more exhaustive (9 detailed design decisions D1–D8, explicit `InputFieldSpec`, serialization-mode field) but slightly over-specified in places (yolo "gating" semantics underdefined; provider-injection assumed). Claude's v3 is the most concisely correct and the clearest about F1 being the reshaping finding.
2. **This document, v3** — now folds in F1/F3/F6 and the retargeting, so it has parity on correctness PLUS the best structure (3-tier, 41 named tests, granular checklist, risk register, source-anchored §D6 facts table with line numbers). **If picking one document to EXECUTE FROM going forward, pick this v3** — it is the union of all three plans' correct findings in the most machine-followable form.
3. **My v2 and earlier** — superseded; missed F1.

The honest meta-lesson (recorded so future readers see it): across the proposal_selection plan AND this plan, the recurring failure mode was **modifying a layer that looks authoritative (handlers/registry) but is dead in the live path.** F1 is the second instance. The durable fix is the §A3 follow-up "consolidate the live inline path onto the handler registry so there is ONE implementation" — until that lands, every conversation-tool change must target the inline `_build_input_mode`/`_handle_conversation_tool`, not the handlers.
