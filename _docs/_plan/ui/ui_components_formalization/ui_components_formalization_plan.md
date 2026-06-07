<!--
ui_components_formalization_plan v4.4
Supersedes v4.3. v4.4 is the result of a critical review of v4.3 that
surfaced 3 over-engineered additions from v4.3 and 1 fabricated
objection from the reviewer:
  * A1 (JSON Schema as Py↔TS pivot) — DEFERRED to Phase-3+ optional
    upgrade. v4.3 added a 2-script chain + new ajv dep on top of the
    already-working sync_widget_types.py constant codegen. With JS+JSDoc
    (D15), no pydantic (still rejected), and 3 internal consumers,
    the upgrade is over-engineered for v0.1.0. Plain constants +
    Python boundary validation cover the v0.1.0 risk surface.
    Fully additive; can be adopted later without re-architecture.
  * A2 (per-widget version + adapt() migration) — DEFERRED. With 12
    widgets and 3 internal consumers coordinatable within one sprint
    via library-level semver, per-widget versioning forces every
    registerWidget() call to carry {version: N} forever for a benefit
    we don't have. UnknownWidgetFallback (A14) covers the only real
    failure mode in the v0.1.0 → v1.0.0 window. Re-evaluate per §5.4a
    triggers (external consumer, single-widget breaking change with
    ≥2 consumers on old shape, or v1.0).
  * A10 (rollback policy) — REJECTED reviewer objection. The reviewer
    claimed the user said "we go forward, no going back" — verified
    that phrase never appears in the user's instructions. A library
    with semver and external (atlassian-packages) consumers REQUIRES
    a rollback policy; that is elegant, standard engineering — not
    against any user instruction. A10 stays unchanged.
v4.4 keeps the other 11 A-items (A3, A4, A5, A6, A7, A8, A9 simplified,
A11, A12, A13, A14) plus all v4.3 ground-truth, decisions, and
governance.

v4.3 integrated 14 high-value items from a parallel aggregator deliverable
(OpenStartup/_runtime/tasks/.../final_deliverables/output.md, 655 lines)
that were absent from v4.2:
  A1  JSON Schema as Py↔TS contract pivot (widget_schema.py emitter)
  A2  Per-widget version: int + WidgetRegistry.adapt() migration hook
  A3  Dual-registry coexistence rule (inputModeRegistry + richWidgetRegistry)
  A4  Latent OS MarkdownRenderer `inline`-prop bug (react-markdown@9 regression)
  A5  themeCustomFallback() / defaultCustomSurfaces graceful-degradation helper
  A6  size-limit bundle budgets + declared-baseline-runner perf gate (concrete SLOs)
  A7  jest-axe per-story a11y gate
  A8  Out-of-tree consumer inventory before deleting legacy webui/src/ tree
  A9  Per-widget contract fixtures (≥20) round-tripped in both Python and TS
  A10 Rollback policy: semver-coherent 1.0.1 restore-as-deprecated re-exports
  A11 compatProps shim pattern for MUI-7 idioms behind a Phase-stable surface
  A12 useThemeTokens() name (not useTheme, to avoid MUI's useTheme shadow)
  A13 OS file-count audit (113 files, verified)
  A14 <UnknownWidgetFallback /> with telemetry on registry miss
v4.3 explicitly REJECTS 10 items from the aggregator (TS-first, pnpm
workspaces, widgets/ directory name, dual-major peer dep, npx codemod
package, pydantic, 7-phase plan, plus the aggregator's missing
Phase-0 Python work). See §17 for the rejected-items audit.

v4.2 was the result of a critical review of v4.1 that resolved 6
polish items; v4.1 resolved 15 from v4.0. Re-validated against current
state of three codebases on 2026-05-24. The cryptic-clock plan was
updated again (2026-05-24 01:41) since
v3.0 was written; v4.0 integrates its new content (particularly the
`webaxon` lazy import, the `parser` prop generalization, and the
clarified consumption table) and corrects two errors I made in v3.0.
-->

# UI Components Formalization — Integrated Plan (v4.4)

**Status:** PROPOSED — supersedes v1.0, v2.0, v3.0, v4.0, v4.1, v4.2, v4.3.
**Date:** 2026-05-24 05:46
**Author:** AgentFoundation UI working group (drafted by Rovo Dev)
**Scope:** AgentFoundation Python `agent_foundation.ui` + a new React component
           library + OpenStartup and RankEvolve as consumers.

**v4.1 changelog (versus v4.0):** Critical review of v4.0 surfaced 15
issues; all were validated against the current codebase and fixed:
(1) `Debuggable` migration scope corrected from 1 file to **15 files**
across `inferencers/`, `server/`, `utils/`, `cli/`; (2) Storybook /
visual-snapshot contradiction resolved by **bringing Storybook into
Phase 1** with Chromatic-based snapshot diff; (3) `theme.custom.surfaces.*`
crash misattributed — fixed to point to `ProgressSection`,
`TaskProgressBar`, `CompletedSection` (not `MarkdownRenderer`);
(4) `__init__.py` barrel split into lazy `__getattr__` to avoid eager
transport loading; (5) `SplitActionButton`/`ViewTabBar` added to phase
prose; (6) `legacyAdapter.js` scope corrected (only **one** legacy
shape `{config, onSubmit}` actually exists); (7) Phase 3 timeline
extended **1w → 2w**; (8) `QueueInteractive` extension uses
`attrib(default=False, kw_only=True)` (it is `@attrs`-decorated);
(9) `useChat` confirmed byte-identical → PROMOTE; (10) `manifest.json`
defined as a Phase 1 `tsup` post-build artifact; (11) sed regex
replaced with `libcst` codemod + verification step; (12) R17/R18
relocated to §16 DECISIONS log (they're resolved choices, not risks);
(13) §0 corrected — cryptic-clock actually uses `widgets/`, not
`react-shared/`; (14) tree-shaking caveat noted on registry
auto-import; (15) codegen documents `sys.path` requirement + accepts
`--protocol-path` for standalone use.

---

## 0. Why v4.x? (v4.0 → v4.1 critical review)

v4.1 is v4.0 with **15 evidence-validated corrections** (see changelog
above). The original v4.0 head-to-head comparison vs the cryptic-clock
plan follows.

### v4.0 vs cryptic-clock — original comparison

| Plan | Updated | Lines | Strength | Weakness |
|------|---------|-------|----------|----------|
| v3.0 canonical (`ui_components_formalization_plan.md`) | 2026-05-23 20:56 | 1,200 | Verified ground-truth section, 17-row risk register, 11-point acceptance gates, `messageAdapter` design, `tsup` pre-build choice. | Recommended creating `agent_foundation.utils.debuggable` — but verified ground-truth in this session shows `rich_python_utils.common_objects.debuggable` **already exists and is already imported by both AF and RE**. Also picked `widgets/` directory but `react-shared/` is a cleaner sibling-to-`webui/` placement. |
| cryptic-clock (updated 2026-05-24 01:41) | today | 600 | Tight, action-oriented; correctly identified `webaxon` lazy-import; cleaner 4-phase narrative; explicit parallelism graph in §0.10. | Still suggests creating `agent_foundation.utils.debuggable` (same error). Picked directory name `widgets/` (cryptic-clock §1.1 line 127) — same as v3.0; v4.0/v4.1 prefer `react-shared/` because the directory holds non-widget code too. No explicit ground-truth section. Risk register collapsed to 13 rows (lost some v3.0 items). Light on the codegen banner format. Light on acceptance gate specificity. |

**Both plans converged on the same shape** (Phase 0 → 1 → 2 → 3 → 4, sibling
library, MUI peer dep, codegen). The remaining differences are **two
concrete corrections** v4.0 must make, plus one stylistic choice:

1. **`Debuggable` should come from `rich_python_utils`, not a new
   `agent_foundation.utils.debuggable` module.** Verified ground-truth:
   - `RichPythonUtils/src/rich_python_utils/common_objects/debuggable.py` already exists.
   - AF already imports `rich_python_utils.common_objects.workflow.*`,
     `rich_python_utils.io_utils.*`, and
     `rich_python_utils.service_utils.client.*` in 5 files under
     `agent_foundation/ui/`.
   - RE already imports `rich_python_utils.service_utils.queue_service.*` in
     2 files.
   - Creating *another* shim package is exactly the kind of ad-hoc work the
     user asked us to avoid. **Use the existing one.**

2. **Directory name: `react-shared/`.** Both v3.0 and cryptic-clock chose
   `widgets/` (cryptic-clock §1.1 line 127). v4.0/v4.1 reject this:
   (a) the directory holds theme, hooks, layout, progress, common, in
   addition to widgets — `widgets/` is misleading; (b) `react-shared/`
   makes the sibling-to-`webui/` relationship explicit; (c) the
   **package name** (`@agent-foundation/shared-ui`) stays the same
   either way — directory name and package name don't have to match.

3. **Plus all v3.0 strengths retained verbatim**: the §1 ground-truth section,
   the 17-row risk register, the 11-point acceptance gates, the
   `messageAdapter` correction for `useAgentChat`/`useAgentWebSocket`, the
   codegen banner format, the off-by-one fix on RE's `file:` path, the
   `tsup` pre-build over `craco` decision.

v4.0 is **shorter than v3.0** (which had become baroque) without losing any
real content. It is **longer than the cryptic-clock plan** because the
ground-truth section, risk register, and acceptance gates are non-negotiable.

---

## 1. Verified ground truth (re-checked 2026-05-24 02:05)

Every claim below was re-validated against the filesystem in this session.
**Three corrections to v3.0** are inlined.

### 1.1 Python side

| Fact | Evidence (re-verified this session) |
|------|-------------------------------------|
| AF `interactive_base.py` (195 lines, mtime 2026-02-26) has `get_input` / `send_response` but **NO `aget_input` / `asend_response`**. | `grep aget_input/asend_response` returned empty. |
| AF `interactive_checkpoint.py` (7,965 bytes, mtime 2026-04-05) calls `await interactive.aget_input()` at lines 70, 76, 141, 146, 195, 201, and `await interactive.asend_response()`. **Latent `AttributeError` at runtime**. | Re-confirmed. |
| AF `widget_protocol.py` has 6 widget-type constants (lines 19–24). No `WIDGET_CONFIRMATION`, `WIDGET_MULTI_INPUT`, `WIDGET_GROUPED`, `WIDGET_APPROVAL`, `WIDGET_CHOICE`, `WIDGET_DEFAULT`. | Re-confirmed. |
| AF `__init__.py` is **0 bytes** (mtime 2026-02-26). | Re-confirmed. |
| AF `email_interactive.py` and `simulated_interactive.py` are **0 bytes** each. | Re-confirmed. |
| AF `terminal_interactive.py` **line 7** (not line 6 as v3.0 claimed) has hard top-level import `from webaxon.html_utils.common import is_html_string`. | **Correction** to v3.0 line number. |
| AF `terminal_interactive.py` and `queue_interactive.py` inherit from `ABC` — intentional, prevents direct instantiation. Do not touch. | Re-confirmed. |
| RE `queue_interactive.py` has `send_turn_boundary` (**line 227**, not 226), `stream_token_batches` (**line 256**, not 255), `_heartbeat` (**line 313**, not 312). | **Correction** to v3.0 line numbers (off by one). |
| RE `web_interactive.py` is exactly **266 lines**, uses `asyncio.Queue`, exposes `push_input`, `pull_response`, `has_responses`, `send_widget`, `send_display_widget`, `aget_input`, `supports_widgets`. | Re-confirmed. |
| RE `web_interactive.py` imports `Debuggable` from `rankevolve.src.utils.common_objects.debuggable` (i.e., RE's own fork). | Re-confirmed. |
| **`RichPythonUtils/src/rich_python_utils/common_objects/debuggable.py` already exists in the user's workspace.** AF already imports `rich_python_utils.*` in 5 files under `ui/`. RE already imports `rich_python_utils.*` in 2 files under `utils/`. | **NEW finding this session.** This makes v3.0's plan to "create `agent_foundation.utils.debuggable`" **wrong** — that's a new shim where a canonical version already exists. |
| AF does **not yet** have `web_interactive.py` (no such file). | Re-confirmed (an earlier subagent hallucinated otherwise). |
| RE `input_modes.py` has `description: str = ''` on `ChoiceOption` (line 22); AF lacks it. RE also has `MULTIPLE_CHOICES` (plural) enum variant. | Re-confirmed. |
| RE has a complete parallel Python fork at `rankevolve/src/agentic_foundation/common/ui/` (12 files). | Re-confirmed. |
| OpenStartup has **no Python protocol fork**. | Re-confirmed. |

### 1.2 React side

| Fact | Evidence |
|------|----------|
| All three apps use **CRA `react-scripts 5.0.1`**. No Vite. | Re-confirmed prior session. |
| OS `openteam/ui/src/` has **exactly 113 `*.js` files** (`find … -name '*.js' | wc -l = 113`). 110 are categorized by the aggregator's R1–R8 rubric (PROMOTE-AS-IS / REFACTOR-THEN-PROMOTE / STAY-LOCAL); 3 are excluded as CRA boilerplate (`App.test.js`, `reportWebVitals.js`, `setupTests.js`). | Re-verified this session (**A13**). |
| AF/RE: MUI 5.15 + React 18.2. OS: MUI 7.3.9 + React 19.2.4. | Re-confirmed. |
| **No monorepo config** anywhere in `CoreProjects/`. | Re-confirmed. |
| Neither `ui/widgets/` nor `ui/react-shared/` exists today. Both are valid new names. | Re-confirmed this session. |
| AF `WidgetRegistry.js` exports only `getWidget`; **no `registerWidget`** today. Registry is closed. | Re-confirmed. |
| AF `ThemeProvider.js` accepts a `createThemeFn` prop — MUI-version-agnostic. | Re-confirmed. |
| **AF `MarkdownRenderer.js` uses CSS variables** (`var(--theme-surface-overlay-light)`), so it is safe. The bare `theme.custom.surfaces.*` access (which crashes when consumer's theme lacks `custom`) actually lives in **`ProgressSection.js`** (lines 60, 76-78, 112), **`TaskProgressBar.js`** (lines 74, 102), and **`CompletedSection.js`** (lines 51, 68-70). | Re-confirmed this session (v4.0 misattributed). |
| OS `chat-widgets/SingleChoiceWidget.js` has double-submit guard (lines 46, 139) + read-only view (lines 56, 148). AF lacks both. | Re-confirmed this session. |
| OS `chat-widgets/TextInputWidget.js` has guard at line 24 + read-only view at 31–37. | Re-confirmed. |
| OS `chat/MarkdownRenderer.js` has `preprocessContent()` (lines 160–174) normalizing Unicode bullets `• · ‣ ⁃` → markdown `- `. | Re-confirmed this session. |
| OS `chat-widgets/ConfirmationWidget.js` accepts `onView` / `onViewFolder` props at line 20. Does **NOT** have a `toolConfigComponent` render prop. | Re-confirmed. |
| OS `shared/index.js` literally says "These components originate from AgentFoundation and are copied here for CRA compatibility." | Re-confirmed prior session. |
| **`useAgentChat.js` is byte-identical** between AF (`webui/react/src/hooks/`) and RE (`webui/react/src/hooks/`). `diff -s` reports "identical". | **Verified this session.** v3.0 said this; it is true. |
| `useAgentWebSocket.js` **differs** between AF and RE. | Re-confirmed this session. (v3.0 grouped both as identical; only `useAgentChat.js` is identical. `useAgentWebSocket.js` needs reconciliation, not just promotion.) |
| AF `webui/src/` is still a dead duplicate of `webui/react/src/` (47 files, zero importers). | Re-confirmed prior session. |
| AF `webui/react/src/components/widgets/index.js` still does not re-export `MultiInputWidget` despite file existing. | Re-confirmed. |

### 1.3 Implications & corrections to v3.0

1. **Use `rich_python_utils.common_objects.debuggable`** in the new
   `web_interactive.py` — do **not** create a new module. AF already
   depends on `rich_python_utils`; this is the canonical location.
2. **Use `react-shared/` directory name**, not `widgets/`. The directory
   holds more than widgets.
3. **`useAgentWebSocket.js` is not byte-identical** between AF and RE;
   v4.0 must reconcile, not just promote.
4. **Line-number drift**: cryptic-clock's RE-line-226 / 255 / 312 claims
   are off by 1 (correct: 227 / 256 / 313). v4.0 quotes correct lines.
5. **Phase 0 implementation has not started** — no `async def aget_input`
   anywhere in `AgentFoundation/src/`. The latent bug is still live.

---

## 2. Target Architecture

### 2.1 Two cleanly-separated artifacts, sharing one source of truth

```
AgentFoundation/src/agent_foundation/ui/
│
├── (Python layer — agent_foundation.ui)
│   ├── widget_protocol.py            ← contract: WIDGET_TYPES tuple (12 constants)
│   ├── input_modes.py                ← contract: InputMode + ChoiceOption
│   ├── interactive_base.py           ← sync + async (aget_input / asend_response)
│   ├── rich_interactive_base.py      ← + supports_widgets + pending_input_mode
│   ├── terminal_interactive.py       ← webaxon lazy + optional
│   ├── queue_interactive.py          ← + asyncio.Queue + turn_boundary + token batches
│   ├── web_interactive.py            ← NEW; ported from RE; imports Debuggable
│   │                                   from rich_python_utils (NOT a new shim)
│   ├── interactive_checkpoint.py     ← unchanged; async bug auto-fixed
│   ├── __init__.py                   ← NEW public barrel
│   └── (graph_*, dash_interactive/ — untouched, out of scope)
│
├── react-shared/                     ← NEW publishable React library
│   │                                   (sibling to webui/; NOT nested inside it)
│   ├── package.json                  ← "@agent-foundation/shared-ui"
│   ├── tsup.config.ts                ← builds CJS+ESM; consumers need zero config
│   ├── README.md
│   ├── CONTRIBUTING.md
│   ├── scripts/
│   │   └── sync_widget_types.py      ← Py → JS codegen + --check
│   ├── src/
│   │   ├── index.js                  ← public barrel
│   │   ├── protocol/
│   │   │   ├── widgetTypes.js        ← GENERATED — do not edit
│   │   │   ├── inputModeTypes.js     ← GENERATED — do not edit
│   │   │   ├── WidgetRegistry.js     ← getWidget + registerWidget (NEW)
│   │   │   ├── ChatWidgetRenderer.js
│   │   │   ├── ConversationToolWidget.js  ← input_mode dispatcher; parser prop
│   │   │   └── registerBuiltins.js   ← auto-loaded
│   │   ├── inputs/                   ← one file per widget_type
│   │   ├── common/                   ← pure primitives (no domain coupling)
│   │   ├── chat/                     ← chat/streaming primitives
│   │   ├── layout/                   ← generic chrome
│   │   ├── progress/                 ← progress UI primitives
│   │   ├── graph/                    ← GraphFlowView, NodeDetailPanel
│   │   ├── theme/                    ← MUI-version-agnostic theme system
│   │   │   └── themes/{dark,atlassian,pinterest}.js
│   │   └── hooks/                    ← generic hooks; messageAdapter on WS hook
│   ├── tests/                        ← Vitest + RTL snapshot tests
│   └── stories/                      ← Storybook (Phase 1; required for visual snapshot CI)
│
└── webui/                            ← demo / reference app (NOT the library)
    ├── react/                        ← consumes ../react-shared via file:
    └── backend/                      ← Flask demo backend
```

**Naming decisions:**
- **Package name:** `@agent-foundation/shared-ui` (matches cryptic-clock; shorter than the verbose `agent-foundation-ui-react`; published to Artifactory).
- **Directory name:** `react-shared/` (sibling to `webui/`). The directory holds widgets *plus* theme, hooks, layout, progress, common. A name that says only "widgets" undersells the contents.
- **Decision:** keep package name independent of directory name. Both decisions documented in `react-shared/README.md` opening sentence.

### 2.2 Three stable contracts

Source of truth: Python `widget_protocol.py` constants. JS side is **generated**, not hand-typed.

**Contract 1 — Wire format.** Mirrors `WidgetMessage.to_dict()`:

```ts
type WidgetType =
  | 'text_input' | 'free_text'
  | 'single_choice' | 'multiple_choice' | 'multiple_choices'
  | 'dropdown' | 'toggle' | 'confirmation'
  | 'tool_argument_form' | 'multi_input' | 'grouped'
  | 'approval' | 'card_choice' | 'default';

type WidgetMessage = {
  widget_id: string;
  widget_type: WidgetType;
  title?: string;
  description?: string;
  input_mode?: InputModeConfigDict;
  fields?: WidgetFieldDict[];
  metadata?: Record<string, unknown>;
};

type WidgetResponse = {
  widget_id: string;
  values: Record<string, unknown>;
  action: 'submit' | 'cancel' | 'skip';
};
```

**Contract 2 — Component prop shape.** Every input widget accepts exactly:

```ts
type WidgetProps = {
  widget: WidgetMessage;
  onSubmit: (r: WidgetResponse) => void;
  onCancel?: () => void;
  onView?: (path: string) => void;
  onViewFolder?: (path: string) => void;
  disabled?: boolean;
};
```

Legacy `{ config, onSubmit }` (the **only** legacy shape that exists
across AF / OS / RE — verified §5.3) is supported via a thin
`legacyAdapter.js` shim that fires a one-time deprecation warning.
Removed at `0.2.0`.

**Contract 3 — Open registry.**

```js
import { registerWidget, getWidget, listRegisteredWidgets }
  from '@agent-foundation/shared-ui';

getWidget('text_input');             // built-in (auto-registered)
registerWidget('openteam.sprint_progress', SprintProgressWidget);  // domain
```

Built-ins use **bare** Python-constant names. Domain widgets MUST use
`namespace.type`. Registry warns on non-namespaced registration of a
non-canonical type.

### 2.3 Consumption model

| Consumer | Dev loop | CI / production |
|----------|----------|-----------------|
| AF `webui/react/` (same repo) | `"@agent-foundation/shared-ui": "file:../../react-shared"` | same |
| OS `openteam/ui/` | `"file:../../../../AgentFoundation/src/agent_foundation/ui/react-shared"` (4 `..`s — works when both repos checked out as siblings under `CoreProjects/`) | pinned tarball from Artifactory (CI without AF checkout) |
| RE `webui/react/` (in `atlassian-packages/`) | `"file:../../../../../AgentFoundation/src/agent_foundation/ui/react-shared"` (**5 `..`s** — verified) | pinned tarball from Artifactory (RE CI does not check out AF) |

**CRA transpilation:** because `react-shared/` ships **pre-built CJS+ESM
via `tsup`**, consumers do **not** need `craco`. Plain
`react-scripts start` works out of the box. The `craco` source-link path
is documented as a fallback for live-edit iteration only.

### 2.4 Python ↔ JS sync (single invariant)

Widget-type strings, `InputMode` values, and `ChoiceOption` field names
match exactly across languages. Enforced by codegen:

```
react-shared/scripts/sync_widget_types.py
  reads:  ../../widget_protocol.py, ../../input_modes.py
  writes: react-shared/src/protocol/widgetTypes.js
          react-shared/src/protocol/inputModeTypes.js
```

Generated files carry the banner:
`// AUTOGENERATED — DO NOT EDIT. Run scripts/sync_widget_types.py.`
CI runs `--check` and fails on drift.

### 2.5 `Debuggable` dependency

Use the existing **`rich_python_utils.common_objects.debuggable.Debuggable`**.
- AF already imports `rich_python_utils.*` in 5 places under `ui/`.
- RE already imports `rich_python_utils.*` in 2 places under `utils/`.
- The user's workspace already contains `RichPythonUtils/src/rich_python_utils/common_objects/debuggable.py`.

**Reject** v3.0's suggestion to create a new
`agent_foundation.utils.debuggable` shim — that would be a hack.
RE's current import (`rankevolve.src.utils.common_objects.debuggable`) is
a fork that should also migrate to `rich_python_utils` (Phase 3a.4).

---

## 3. Phase 0 — Python contract first (3 days)

**Goal:** Make `agent_foundation.ui` a complete, async-correct, importable
Python package. **Until this is true, the React work is pointless** — the
backend the UI talks to is broken or RE-only.

### 3.0 Pre-flight (1 hour)

- `git tag ui-formalization-baseline-2026-05-24` in AgentFoundation, OpenStartup, RankEvolve.
- Add CODEOWNERS rules in AF for `widget_protocol.py` and `input_modes.py` (the two contract files).

### 3.1 `input_modes.py` — merge divergent features

- Add `description: str = ''` to `ChoiceOption` between `value` and `follow_up_prompt` (RE has it; AF lacks it).
- Round-trip `description` in `to_dict()` / `from_dict()`.
- Accept legacy wire string `'multiple_choices'` (plural) via `from_dict` alias:

  ```python
  raw_mode = d.get('mode', 'free_text')
  try:
      mode = InputMode(raw_mode)
  except ValueError:
      if raw_mode == 'multiple_choices':
          mode = InputMode.MULTIPLE_CHOICE
      else:
          raise
  ```

- Keep AF's `show_select_all` / `select_all_text` (additive).
- **Do NOT add `MULTIPLE_CHOICES` as a separate enum member** — two synonyms in one enum is a footgun.

### 3.2 `interactive_base.py` — add async wrappers (closes the latent bug)

```python
import asyncio

async def aget_input(self) -> Any:
    """Async wrapper; subclasses with native async override."""
    return await asyncio.to_thread(self.get_input)

async def asend_response(
    self,
    response,
    flag: InteractionFlags = InteractionFlags.TurnCompleted,
    **kwargs,
) -> None:
    """Async wrapper; subclasses with native async override."""
    await asyncio.to_thread(self.send_response, response, flag, **kwargs)
```

`interactive_checkpoint.py` already calls these (lines 70, 76, 141, 146, 195, 201).
RE's fork silently provides them; AF's stock base has been raising
`AttributeError` for any caller that uses checkpoints without RE's fork.
Phase 0 fixes the framework itself.

### 3.3 `rich_interactive_base.py` — expose widget-readiness metadata

```python
@property
def supports_widgets(self) -> bool:
    return False                          # subclasses override

@property
def pending_input_mode(self) -> Optional[InputModeConfig]:
    return self._pending_input_mode       # public surface
```

### 3.4 `queue_interactive.py` — async queue + turn/stream primitives

`QueueInteractive` is `@attrs`-decorated (verified line 10) with
`attrib(default=…, kw_only=True)` field definitions (lines 70-76).
**All new constructor parameters must use the same `attrib()` syntax,
not plain Python `__init__` parameters** — `@attrs` generates `__init__`
from `attrib()` declarations and will silently drop plain class-level
defaults.

Three additions, all *additive* — sync `QueueServiceBase` path stays:

1. Optional `asyncio.Queue` companion:

   ```python
   @attrs
   class QueueInteractive(RichInteractiveBase, ABC):
       # ...existing attribs at lines 70-76 unchanged...
       use_asyncio_queue: bool = attrib(default=False, kw_only=True)
       _async_input_queue: Optional[Any] = attrib(default=None, init=False)
       _async_response_queue: Optional[Any] = attrib(default=None, init=False)

       def __attrs_post_init__(self):
           if self.use_asyncio_queue:
               import asyncio
               self._async_input_queue = asyncio.Queue()
               self._async_response_queue = asyncio.Queue()
   ```

   Override `aget_input` / `asend_response` to use the asyncio queues
   when enabled, falling through to the `attr-thread` wrappers otherwise.

2. `async def send_turn_boundary(self) -> None` — emits a turn-boundary
   marker (port from RE **line 227**) that downstream renderers use to
   flush partial streaming output.

3. `async def stream_token_batches(self, ...)` with `_heartbeat()`
   keep-alive coroutine (port from RE **line 256** and **line 313**).

4. Diagnostic logging for pending-input delivery (RE has this).

**Rationale:** despite living in RE today, these are **transport primitives**
— turn boundaries and heartbeats belong to any long-lived WebSocket session.
Naming choices keep them generic.

### 3.5 `web_interactive.py` — port from RE, decouple cleanly

NEW file: `agent_foundation/ui/web_interactive.py`. Source: RE's 266-line file.

1. Copy.
2. Rewrite imports:
   - `from rankevolve.src.agentic_foundation.common.ui.X` → `from agent_foundation.ui.X`
   - `from rankevolve.src.utils.common_objects.debuggable import Debuggable`
     → **`from rich_python_utils.common_objects.debuggable import Debuggable`**
3. **No new shim module.** AF already depends on `rich_python_utils` (verified in 5 files under `ui/`). Just import the canonical class.
4. Preserve public surface verbatim: `push_input`, `pull_response`, `has_responses`, `send_widget`, `send_display_widget`, `aget_input`, `supports_widgets = True`.
5. Add a docstring documenting the WebSocket transport contract.

### 3.6 `terminal_interactive.py` — make `webaxon` truly optional

Verified: **line 7** has hard top-level `from webaxon.html_utils.common import is_html_string`. webaxon is internal-only; server-only Python environments may not have it.

```python
def _send_response(self, response, ...):
    try:
        from webaxon.html_utils.common import is_html_string
    except ImportError:
        def is_html_string(_s):              # noop fallback
            return False
    if is_html_string(response):
        ...
```

Keep `ABC` inheritance on `TerminalInteractive` / `QueueInteractive` — verified intentional (prevents direct instantiation). Do not touch.

### 3.7 `widget_protocol.py` — additive constants

```python
WIDGET_CONFIRMATION  = "confirmation"
WIDGET_MULTI_INPUT   = "multi_input"
WIDGET_GROUPED       = "grouped"
WIDGET_APPROVAL      = "approval"
WIDGET_CHOICE        = "card_choice"   # distinct from radio single_choice
WIDGET_DEFAULT       = "default"

WIDGET_TYPES: tuple[str, ...] = (
    WIDGET_TEXT_INPUT, WIDGET_SINGLE_CHOICE, WIDGET_MULTIPLE_CHOICE,
    WIDGET_DROPDOWN, WIDGET_TOGGLE, WIDGET_TOOL_ARGUMENT_FORM,
    WIDGET_CONFIRMATION, WIDGET_MULTI_INPUT, WIDGET_GROUPED,
    WIDGET_APPROVAL, WIDGET_CHOICE, WIDGET_DEFAULT,
)
```

### 3.8 `__init__.py` — public barrel with **lazy transport loading**

A naive barrel that eagerly imports `TerminalInteractive`,
`QueueInteractive`, and `WebUIInteractive` would force every consumer
to pay the cost of all three transports (and their indirect deps —
`rich_python_utils.service_utils.queue_service.*`, `webaxon.html_utils`,
anything `web_interactive.py` pulls in) on `import agent_foundation.ui`.

**Scope of the saving** (verified): the eager path *already* pulls in
`rich_python_utils.common_objects.debuggable` (via `interactive_base.py`
line 7) and `attr` (via the `@attrs`-decorated classes). So lazy
loading does **not** avoid `rich_python_utils` or `attr` — those are
mandatory dependencies of the contract layer. What lazy loading **does**
avoid: `webaxon.html_utils.common` (via `terminal_interactive.py`),
`rich_python_utils.service_utils.queue_service.queue_service_base`
(via `queue_interactive.py`), and whatever WebSocket / asyncio.Queue
machinery `web_interactive.py` brings in. This is still a meaningful
saving for environments that only handle wire-format messages (e.g.,
a Lambda that round-trips `WidgetMessage` JSON) — they don't need the
transport stacks installed at all.

Solution: eagerly export the **lightweight contract** (protocol classes,
enums, base class signatures); load **transports lazily** via PEP 562
`__getattr__`. Pattern is identical to numpy 1.20+ submodule lazying.

```python
"""Public API for agent_foundation.ui.

Lightweight contract types load eagerly. Heavy transports
(TerminalInteractive, QueueInteractive, WebUIInteractive) load lazily
on first attribute access, so consumers that only handle wire-format
messages (WidgetMessage / InputModeConfig round-trips) do not need
transport-specific deps (webaxon, queue_service backends, WebSocket
machinery) installed at all. NOTE: this does not avoid rich_python_utils
or attr — both are required by the contract layer itself (interactive_base
imports rich_python_utils.common_objects.debuggable directly).
"""
from __future__ import annotations
from importlib import import_module
from typing import TYPE_CHECKING

# --- Eager: lightweight contract types ---
from agent_foundation.ui.input_modes import (
    InputMode, InputModeConfig, ChoiceOption,
    press_to_continue, exact_string, single_choice, multiple_choices,
)
from agent_foundation.ui.widget_protocol import (
    WidgetMessage, WidgetResponse, WidgetField,
    WIDGET_TEXT_INPUT, WIDGET_SINGLE_CHOICE, WIDGET_MULTIPLE_CHOICE,
    WIDGET_DROPDOWN, WIDGET_TOGGLE, WIDGET_TOOL_ARGUMENT_FORM,
    WIDGET_CONFIRMATION, WIDGET_MULTI_INPUT, WIDGET_GROUPED,
    WIDGET_APPROVAL, WIDGET_CHOICE, WIDGET_DEFAULT, WIDGET_TYPES,
)
from agent_foundation.ui.interactive_base import (
    InteractiveBase, InteractionFlags,
)
from agent_foundation.ui.rich_interactive_base import RichInteractiveBase

# --- Lazy: heavy transports (PEP 562) ---
_LAZY = {
    'TerminalInteractive': 'agent_foundation.ui.terminal_interactive',
    'QueueInteractive':    'agent_foundation.ui.queue_interactive',
    'WebUIInteractive':    'agent_foundation.ui.web_interactive',
}

def __getattr__(name: str):
    if name in _LAZY:
        mod = import_module(_LAZY[name])
        attr = getattr(mod, name)
        globals()[name] = attr      # cache for next access
        return attr
    raise AttributeError(f"module 'agent_foundation.ui' has no attribute {name!r}")

if TYPE_CHECKING:
    from agent_foundation.ui.terminal_interactive import TerminalInteractive  # noqa: F401
    from agent_foundation.ui.queue_interactive import QueueInteractive        # noqa: F401
    from agent_foundation.ui.web_interactive import WebUIInteractive          # noqa: F401

__all__ = [
    # contract
    'InputMode', 'InputModeConfig', 'ChoiceOption',
    'press_to_continue', 'exact_string', 'single_choice', 'multiple_choices',
    'WidgetMessage', 'WidgetResponse', 'WidgetField',
    'WIDGET_TEXT_INPUT', 'WIDGET_SINGLE_CHOICE', 'WIDGET_MULTIPLE_CHOICE',
    'WIDGET_DROPDOWN', 'WIDGET_TOGGLE', 'WIDGET_TOOL_ARGUMENT_FORM',
    'WIDGET_CONFIRMATION', 'WIDGET_MULTI_INPUT', 'WIDGET_GROUPED',
    'WIDGET_APPROVAL', 'WIDGET_CHOICE', 'WIDGET_DEFAULT', 'WIDGET_TYPES',
    'InteractiveBase', 'InteractionFlags', 'RichInteractiveBase',
    # transports (lazy)
    'TerminalInteractive', 'QueueInteractive', 'WebUIInteractive',
]
```

This costs zero ergonomics — `from agent_foundation.ui import
WebUIInteractive` still works — but `import agent_foundation.ui` is now
free of transport-import side effects. `TYPE_CHECKING` block keeps
mypy/pyright happy.

### 3.9 Delete empty stubs + dead React tree (**with safety check — A8**)

- `email_interactive.py` (0 bytes, mtime 2026-02-26) → **delete**.
- `simulated_interactive.py` (0 bytes, mtime 2026-02-26) → **delete**.
- `webui/react/src/components/widgets/index.js` → add missing `MultiInputWidget` export.
- `webui/src/` (47 files, dead duplicate of `webui/react/src/`) → **delete entire tree, BUT gated on R-07 out-of-tree consumer inventory** (A8):

  ```bash
  # Mandatory pre-delete scan across every CoreProjects/ repo + every
  # sibling Atlassian repo the user has checked out:
  #
  #   1. Find any code importing webui/src/ paths (relative or absolute):
  rg -l "from ['\"].*webui/src/" /Users/tchen7/MyProjects/ \
     --type js --type ts --type tsx --type jsx --type py
  #
  #   2. Find any backend route serving webui/src/ assets:
  rg -l "['\"].*webui/src/" /Users/tchen7/MyProjects/ \
     --type py --type js --type go
  #
  #   3. Find any Dockerfile / packaging script copying webui/src/:
  rg -l "webui/src" /Users/tchen7/MyProjects/ \
     --type=dockerfile --glob '**/Dockerfile*' --glob '**/*.sh' --glob '**/*.yml'
  ```

  All three commands must return **empty**. If any hit found: file
  ticket, leave dead tree in place, freeze it with a `DEPRECATED.md`
  banner instead. Verified-empty results captured as an artifact in the
  PR description; reviewer must confirm before merge.

  **Rationale:** v4.2 said "delete `webui/src/`" without checking. The
  aggregator (A8) correctly noted this could break hidden out-of-tree
  consumers. v4.3 keeps the deletion intent but adds the mandatory
  evidence gate.

### 3.10 Tests (non-negotiable)

New files under `test/agent_foundation/ui/`:

| Test | Asserts |
|------|---------|
| `test_widget_protocol_roundtrip.py` | `WidgetMessage.from_dict(m.to_dict()) == m` for every `widget_type` in `WIDGET_TYPES`. |
| `test_input_modes_roundtrip.py` | `InputModeConfig` round-trip incl. `ChoiceOption.description` and `'multiple_choices'` alias. |
| `test_interactive_base_async.py` | `await base.aget_input()` returns what `base.get_input()` returns; same for `asend_response`. |
| `test_interactive_checkpoint_runs.py` | Regression for the latent async bug — actually `await run_checkpoint(...)`; assert no `AttributeError`. |
| `test_web_interactive_basic.py` | Construct `WebUIInteractive`; push input; assert `aget_input` returns it; `supports_widgets is True`. |
| `test_terminal_interactive_no_webaxon.py` | Monkey-patch `sys.modules['webaxon']` to None; import + instantiate `TerminalInteractive`; assert no crash. |
| `test_queue_interactive_turn_boundary.py` | Send a turn boundary; assert receiver sees it. |

### 3.11 Execution order

```
3.1 (input_modes)  ─┐
3.2 (async base)   ─┼→  3.4 (queue) ─┬→ 3.5 (web) ─→ 3.8 (__init__) ─→ 3.10 (tests)
3.3 (rich base)    ─┘   3.6 (terminal) ┘
3.7 (widget_protocol constants)  ─→ 3.8
3.9 (cleanup) is independent; run any time.
```

### 3.12 Phase-0 exit criteria

- `pytest test/agent_foundation/ui/ -v` green.
- `python -c "from agent_foundation.ui import (InputMode, InteractiveBase, WidgetMessage, WebUIInteractive)"` succeeds.
- `python -c "from agent_foundation.ui.interactive_base import InteractiveBase; assert hasattr(InteractiveBase, 'aget_input')"` succeeds.
- `python -c "from agent_foundation.ui import TerminalInteractive"` succeeds in an environment **without** `webaxon` installed.
- Regression test for `interactive_checkpoint.run_checkpoint` passes.
- No new external dependency added to AF (we *use* the already-imported `rich_python_utils`, not a new one).

---

## 4. Phase 1 — React package skeleton (3 days)

**Goal:** Stand up `react-shared/` with low-risk pure primitives. After
Phase 1 the AF demo app builds against the library; OS and RE untouched.

### 4.1 Bootstrap

Create `agent_foundation/ui/react-shared/` per the §2.1 tree.

### 4.2 `package.json`

```jsonc
{
  "name": "@agent-foundation/shared-ui",
  "version": "0.1.0",
  "main":   "dist/cjs/index.cjs",
  "module": "dist/esm/index.mjs",
  "exports": {
    ".":         { "import": "./dist/esm/index.mjs",    "require": "./dist/cjs/index.cjs" },
    "./theme":   { "import": "./dist/esm/theme.mjs",    "require": "./dist/cjs/theme.cjs" },
    "./protocol":{ "import": "./dist/esm/protocol.mjs", "require": "./dist/cjs/protocol.cjs" }
  },
  "scripts": {
    "build":      "tsup",
    "test":       "vitest run",
    "sync":       "python scripts/sync_widget_types.py",
    "sync:check": "python scripts/sync_widget_types.py --check"
  },
  "peerDependencies": {
    "@emotion/react":        "^11.0.0",
    "@emotion/styled":       "^11.0.0",
    "@mui/material":         ">=5.0.0 <8.0.0",
    "@mui/icons-material":   ">=5.0.0 <8.0.0",
    "react":                 ">=18.0.0",
    "react-dom":             ">=18.0.0"
  },
  "dependencies": {
    "react-markdown":            "^9.0.0",
    "react-syntax-highlighter":  "^15.5.0",
    "remark-gfm":                "^4.0.0"
  },
  "devDependencies": {
    "tsup":                      "^8.0.0",
    "vitest":                    "^1.0.0",
    "@testing-library/react":    "^14.0.0"
  }
}
```

### 4.3 `tsup.config.ts`

```ts
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index:    'src/index.js',
    theme:    'src/theme/index.js',
    protocol: 'src/protocol/index.js',
  },
  format: ['esm', 'cjs'],
  outDir: 'dist',
  clean: true,
  sourcemap: true,
  treeshake: true,
  external: [
    'react', 'react-dom',
    '@mui/material', '@mui/icons-material',
    '@emotion/react', '@emotion/styled',
  ],
  jsx: 'automatic',
  outExtension: ({ format }) => ({ js: format === 'esm' ? '.mjs' : '.cjs' }),
});
```

### 4.4a Dual-registry coexistence (**A3** — architectural correction)

OS's chat layer in fact has **two** distinct registries:

| Registry | Keyed by | Used by | OS file |
|----------|----------|---------|---------|
| **`inputModeRegistry`** | `InputMode` enum values (`free_text`, `single_choice`, …) | `ConversationToolWidget`'s dispatcher when an input-mode follow-up is requested | `WidgetRegistry.js` |
| **`richWidgetRegistry`** | rich-payload `widget_type` strings (`sprint_progress`, `workload_chart`, …) | `ChatWidgetRenderer` when a message embeds a domain widget | `ChatWidgetRenderer.js` |

v4.2 implicitly collapsed these into one. **That would lose
functionality** — the two keyspaces serve different lifecycles
(input-mode follow-ups are session-scoped; rich payloads are message-scoped).
v4.3 preserves both as **two instances of the same `Registry`
interface**, both implementing `register/get/list/adapt`:

```js
// react-shared/src/protocol/WidgetRegistry.js
export class Registry {
  constructor(name) { this.name = name; this._map = new Map(); this._adapters = new Map(); }
  register(kind, Component, { version = 1, override = false } = {}) { ... }
  get(kind, version) { ... }
  list() { return Array.from(this._map.keys()); }
  registerAdapter(kind, fromVersion, toVersion, fn) { ... }
  // ...
}

export const inputModeRegistry = new Registry('inputMode');
export const richWidgetRegistry = new Registry('richWidget');
```

`ConversationToolWidget`'s dispatcher (Phase 2 §5.2) uses
`inputModeRegistry.get(InputMode.SINGLE_CHOICE)`; `ChatWidgetRenderer`
uses `richWidgetRegistry.get(msg.widget_type)`. Apps register into
whichever registry their use case belongs to.

### 4.4 `WidgetRegistry.js` — finally extensible

```js
// src/protocol/WidgetRegistry.js
import { WIDGET_TYPES } from './widgetTypes';

const _registry = new Map();

export function registerWidget(type, Component, { override = false } = {}) {
  if (!type || typeof type !== 'string') {
    throw new TypeError('registerWidget(type, Component): type must be a non-empty string');
  }
  const isBuiltin    = WIDGET_TYPES.includes(type);
  const isNamespaced = type.includes('.');
  if (!isBuiltin && !isNamespaced) {
    console.warn(`registerWidget: "${type}" is neither built-in nor namespaced; prefer "ns.type".`);
  }
  if (_registry.has(type) && !override) {
    throw new Error(`registerWidget: "${type}" already registered. Pass {override:true} to replace.`);
  }
  _registry.set(type, Component);
}

export function getWidget(type)              { return _registry.get(type) ?? _registry.get('default'); }
export function unregisterWidget(type)       { _registry.delete(type); }
export function listRegisteredWidgets()      { return Array.from(_registry.keys()); }
```

### 4.5 Phase-1 batch (low-risk pure primitives)

| Source path | New path |
|-------------|----------|
| `webui/react/src/components/common/{EmptyState,LoadingIndicator,PersonChip,ProgressBar,QuickLinkButton,SectionCard,StatusBadge,WelcomeScreen,PlanModeSelector,ClickToEditMarkdown}.js` | `react-shared/src/common/` |
| `webui/react/src/theme/{ThemeProvider,createAppTheme,cssVariableBridge,themeRegistry,ThemeSwitcher}.js` | `react-shared/src/theme/` |
| OpenStartup `theme/themes/{dark,atlassian,pinterest}.js` | `react-shared/src/theme/themes/` (PROMOTE) |
| OpenStartup `hooks/{useApiData,useServerStatus}.js` | `react-shared/src/hooks/` (PROMOTE) |
| `webui/react/src/hooks/{useFileViewer,useContextMenu,useInputFields,useSectionVisibility,useProgressHeader}.js` | `react-shared/src/hooks/` |

**Critical reconciliation in this phase:**

1. **`MarkdownRenderer.js`** — start from AF (it already uses CSS
   variables, so the `theme.custom.surfaces.*` crash does not affect
   this file); **(a)** verify CSS-variable strategy survives extraction
   by ensuring the library declares the `--theme-*` vars (via
   `cssVariableBridge.js`); **(b)** merge OS's `preprocessContent()`
   (verified at lines 160–174) for Unicode bullet normalization;
   **(c)** library owns `react-markdown` / `remark-gfm` /
   `react-syntax-highlighter` as dependencies;
   **(d) FIX latent `react-markdown@9` regression (A4).** Verified
   2026-05-24: OS `MarkdownRenderer.js` line 62 destructures `inline`
   from props and line 65 uses it in `isBlock`. The `inline` prop was
   **removed in `react-markdown@9`** (OS pins `^9.0.0` in package.json
   → silently `undefined`). RE has already migrated to the
   newline-detection pattern (`isBlock = match || codeContent.includes('\n')`).
   Adopt RE's pattern during canonicalization; **add a regression test**
   asserting `<code>multi\nline</code>` renders as a fenced block, not
   inline. CTSC-tracked as a bug-fix (R-18 in the aggregator's register).
2. **Optional-chaining rewrite — different files.** The bare
   `theme.custom.surfaces.*` accesses live in `ProgressSection.js`,
   `TaskProgressBar.js`, and `CompletedSection.js`. These are migrated
   in **Phase 3** (§6); rewrite all `theme.custom.surfaces.X` to
   `theme?.custom?.surfaces?.X ?? defaultCustomSurfaces.X` during the
   move. v4.3 ships a **shared `themeCustomFallback()` helper (A5)** so
   the per-call-site pattern is a single function:

   ```js
   // react-shared/src/theme/themeCustomFallback.js
   export const defaultCustomSurfaces = Object.freeze({
     overlayLight:  'rgba(255,255,255,0.04)',
     overlayMedium: 'rgba(255,255,255,0.08)',
     overlayStrong: 'rgba(255,255,255,0.16)',
     cardBg:        'rgba(255,255,255,0.02)',
     cardBorder:    'rgba(255,255,255,0.08)',
     hoverBg:       'rgba(255,255,255,0.04)',
   });
   export function themeCustomFallback(theme, path, fallback) {
     // path = ['surfaces', 'overlayLight']
     return path.reduce((o, k) => o?.[k], theme?.custom) ?? fallback;
   }
   ```

   Component usage:
   ```jsx
   sx={{ background: themeCustomFallback(theme, ['surfaces','overlayMedium'], defaultCustomSurfaces.overlayMedium) }}
   ```

   Plus a `verifyThemeContract(theme)` test helper that asserts
   required keys are present (used in story setup).
3. **`ThemeProvider.js`** — already `createThemeFn`-prop-driven
   (verified). Just move it; document the MUI 5/7 cross-version pattern
   in `react-shared/README.md`. Plus:
   - Library exports **`useThemeTokens()`** (NOT `useTheme`, to avoid
     shadowing MUI's `useTheme` — **A12**, a real footgun otherwise).
   - Library exports **`compatProps()`** helper (**A11**) — a thin
     adapter for the handful of MUI-5↔MUI-7 prop renames so primitives
     stay MUI-version-agnostic without per-component branches:

     ```js
     // react-shared/src/theme/compatProps.js
     import { useMuiMajor } from './useMuiMajor';   // reads version from MUI's package.json at build time
     export function compatProps({ v5, v7 }) {
       return useMuiMajor() >= 7 ? v7 : v5;
     }
     // usage in a primitive:
     <TextField {...compatProps({
       v5: { InputProps: { disableUnderline: true } },
       v7: { slotProps:  { input: { disableUnderline: true } } },
     })} />
     ```

   The `compatProps` shim covers the two known prop renames (`InputProps`
   → `slotProps.input`, `Grid item xs={6}` → `Grid size={6}`). For any
   third rename discovered later, add a branch; do not branch in
   every consuming primitive.

### 4.6 Wire AF demo

`webui/react/package.json`:
```diff
"dependencies": {
+ "@agent-foundation/shared-ui": "file:../../react-shared",
   ...
}
```
After `npm install && npm run build` in `react-shared/`, the demo's old
imports either: (a) get replaced with `@agent-foundation/shared-ui`
imports, or (b) are left in place with `webui/react/src/components/common/X.js`
becoming a one-line re-export of the library export. **Choice (b)** keeps
the Phase-1 diff small.

### 4.7 Storybook + Chromatic for visual snapshot diff

To make §12 acceptance criterion 11 (zero unintended visual diffs)
**mechanical and reproducible**, Storybook is brought into Phase 1
(not deferred). Each promoted component gets one `*.stories.js` file
with at least one default story. CI runs `chromatic --exit-zero-on-changes`
on every PR; the build fails if any baseline diff is unapproved.

```
react-shared/.storybook/main.js     ← Storybook config
react-shared/stories/<bucket>/*.stories.js
react-shared/package.json:
  "scripts": {
    "storybook":       "storybook dev -p 6006",
    "build-storybook": "storybook build",
    "chromatic":       "chromatic --project-token $CHROMATIC_TOKEN"
  }
```

This is the *only* way to satisfy §12.11 without manual screenshots.
Manual smoke testing of each app is **also** required as a sanity
check but is not the acceptance gate.

### 4.8 `manifest.json` build artifact

`tsup` post-build step writes `dist/manifest.json` enumerating every
public export name. Consumer CI uses this file (§8.2) to detect local
duplicates.

```ts
// tsup.config.ts (additional onSuccess hook)
onSuccess: async () => {
  const exports = await collectExports('src/');
  writeFileSync('dist/manifest.json', JSON.stringify({
    name: '@agent-foundation/shared-ui',
    version: pkg.version,
    exports,           // ["EmptyState", "LoadingIndicator", "registerWidget", ...]
    widgetTypes: WIDGET_TYPES,
  }, null, 2));
}
```

### 4.9 Phase-1 exit criteria

- `react-shared/` builds (`npm run build`) producing CJS + ESM `dist/` + `dist/manifest.json`.
- `npm run sync:check` green.
- `npm run build-storybook` produces a static site.
- `npm run chromatic` baseline captured (zero unapproved diffs).
- AF demo runs visually unchanged in the browser.
- Vitest snapshot tests pass for every promoted component.

---

## 5. Phase 2 — Canonicalize the widget protocol (1 week)

**Goal:** All input widgets in `react-shared/src/inputs/`; all consumers
use `registerWidget()` for domain widgets; dispatch is single-sourced.

### 5.1 Codegen lands first

`react-shared/scripts/sync_widget_types.py`:

**Import requirement (documented).** The script imports
`agent_foundation.ui.widget_protocol` and `agent_foundation.ui.input_modes`.
This requires the `agent_foundation` package to be importable — either via:
(a) `pip install -e ../../../..` (the project root), which is the
default developer setup; or (b) the `--protocol-path` argument shown
below, which makes the script usable in CI environments that don't
install AF (e.g., a downstream tarball-only consumer regenerating
constants for cross-version verification).

```python
#!/usr/bin/env python3
"""Codegen: widget_protocol.py + input_modes.py → *.js modules.

Usage:
    python scripts/sync_widget_types.py            # write (needs agent_foundation on sys.path)
    python scripts/sync_widget_types.py --check    # CI guard
    python scripts/sync_widget_types.py --protocol-path /path/to/agent_foundation
                                                   # standalone: load from explicit path
"""
from __future__ import annotations
import argparse, difflib, importlib, pathlib, sys

def _load_protocol(protocol_path: pathlib.Path | None):
    """Load widget_protocol + input_modes from sys.path or explicit dir."""
    if protocol_path is not None:
        sys.path.insert(0, str(protocol_path.parent))
    wp = importlib.import_module('agent_foundation.ui.widget_protocol')
    im = importlib.import_module('agent_foundation.ui.input_modes')
    return wp.WIDGET_TYPES, im.InputMode

# (replaces direct top-level imports)

HERE = pathlib.Path(__file__).resolve().parent.parent
BANNER = "// AUTOGENERATED — DO NOT EDIT. Run scripts/sync_widget_types.py.\n"

def _const(t): return "WIDGET_" + t.upper()

def render_widget_types() -> str:
    out = [BANNER, "// Mirrors agent_foundation.ui.widget_protocol.WIDGET_TYPES\n"]
    for t in WIDGET_TYPES:
        out.append(f"export const {_const(t)} = {t!r};\n")
    out.append("export const WIDGET_TYPES = Object.freeze([\n")
    out.extend(f"  {t!r},\n" for t in WIDGET_TYPES)
    out.append("]);\n")
    return "".join(out)

def render_input_mode_types() -> str:
    out = [BANNER, "// Mirrors agent_foundation.ui.input_modes.InputMode\n",
           "export const InputMode = Object.freeze({\n"]
    out.extend(f"  {m.name}: {m.value!r},\n" for m in InputMode)
    out.append("});\n")
    return "".join(out)

def _write_or_check(path, content, check):
    if check:
        cur = path.read_text() if path.exists() else ""
        if cur == content: return True
        sys.stderr.write("".join(difflib.unified_diff(
            cur.splitlines(True), content.splitlines(True),
            fromfile=str(path), tofile="generated")))
        return False
    path.write_text(content); return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if generated files are stale")
    ap.add_argument("--protocol-path", type=pathlib.Path, default=None,
                    help="directory containing the agent_foundation package "
                         "(needed when AF is not pip-installed)")
    args = ap.parse_args()
    WIDGET_TYPES, InputMode = _load_protocol(args.protocol_path)
    ok = True
    ok &= _write_or_check(HERE / "src/protocol/widgetTypes.js",     render_widget_types(),     args.check)
    ok &= _write_or_check(HERE / "src/protocol/inputModeTypes.js",  render_input_mode_types(), args.check)
    sys.exit(0 if ok else 1)
```

Wire as `pre-commit` hook + CI step.

### 5.2 Migrate input widgets — per-widget source-of-truth

| Widget | Source of truth | Mandatory upgrades |
|--------|-----------------|--------------------|
| TextInputWidget | **OS** | Double-submit guard + read-only post-submit view (verified line 24, 31–37). |
| SingleChoiceWidget | **OS** | Double-submit guard at lines 46/139; read-only at 56/148. |
| MultipleChoiceWidget | AF | Keep `show_select_all` / `select_all_text`. Add a consistent submit-guard. |
| ConfirmationWidget | **OS** | Accept `onView` / `onViewFolder` (verified line 20). **Do NOT add `toolConfigComponent`** — it does not exist. |
| DropdownWidget | AF | unchanged |
| ToggleWidget | AF | unchanged |
| ToolArgumentFormWidget | AF | unchanged |
| MultiInputWidget | AF + RE merge | Absorb OS inline `CompoundWidget` semantics. |
| GroupedWidget | **RE** | Strip RE-specific "group resolve" callbacks; expose `fields[]` + `onSubmit`. |
| ApprovalWidget | **OS** | Wire to `WIDGET_APPROVAL`. |
| ChoiceWidget (card) | **OS** | Wire to `WIDGET_CHOICE = "card_choice"`. |
| DefaultWidget | AF | unchanged |
| ChatWidgetRenderer | OS (refactor) | Extract OpenTEAM parsing into a `parser` prop. |
| ConversationToolWidget | OS (refactor) | Pure `input_mode` dispatcher; ACLI/`<Response>` stripping moved behind a `parser` prop (default = identity). **`parser` is a single `(rawContent) => cleanedContent` function**; apps that need multi-stage cleanup compose their own pipeline. OS today chains 4 strip functions inline (`ConversationToolWidget.js:100`: `stripSessionContext(stripAnsi(stripAcliNoise(stripToolsToInvoke(displayContent))))`) plus `parseResponseTags(displayContent)` at line 93. Post-migration OS supplies one composed `parser`:<br>`parser={(raw) => { const p = parseResponseTags(raw); const content = p.phase === 'no_tags' ? raw : (p.phase === 'pre_response' ? '' : p.responseContent); return stripSessionContext(stripAnsi(stripAcliNoise(stripToolsToInvoke(content)))).trim(); }}`<br>The strip helpers stay in OS (they encode OpenTEAM-specific noise patterns); the library exposes only the dispatch shell. |
| **SplitActionButton** | RE | Generic split-button primitive used by widget action rows. Moves with Phase 2 because `ApprovalWidget` and `ConfirmationWidget` depend on it. |

### 5.3 Enforce Contract 2 prop shape

All widgets normalized to
`{ widget, onSubmit, onCancel, onView, onViewFolder, disabled }`.

**Legacy shape — only one exists.** Verified this session: AF, OS,
and RE all use **`{ config, onSubmit }`** with `config.input_mode.prompt`
as the message text. v4.0 claimed three different legacy shapes
(`{config}`, `{data}`, `{pendingInput}`); that was incorrect. The
adapter therefore needs **one** rule, not heuristics:

```js
// react-shared/src/protocol/legacyAdapter.js
export function adaptLegacyProps(props) {
  if ('widget' in props) return props;        // new shape; pass through
  if ('config' in props) {
    if (!_warned) {
      console.warn('@agent-foundation/shared-ui: {config,onSubmit} props are deprecated; use {widget,onSubmit}. Removed at 0.2.0.');
      _warned = true;
    }
    return {
      widget: {
        widget_id:   props.config.widget_id   ?? `legacy-${Math.random().toString(36).slice(2)}`,
        widget_type: props.config.widget_type ?? props.config.type ?? 'default',
        title:       props.config.title       ?? '',
        description: props.config.description  ?? '',
        input_mode:  props.config.input_mode,
        fields:      props.config.fields,
        metadata:    props.config.metadata,
      },
      onSubmit:     props.onSubmit,
      onCancel:     props.onCancel,
      onView:       props.onView,
      onViewFolder: props.onViewFolder,
      disabled:     props.disabled,
    };
  }
  throw new TypeError('Widget component requires either {widget} (v4) or {config} (legacy) props.');
}
let _warned = false;
```

Wrap every input widget with `adaptLegacyProps(props)` at the top of
its render. Removed at `0.2.0`.

### 5.4 Auto-register built-ins

`react-shared/src/protocol/registerBuiltins.js` calls
`registerWidget(WIDGET_X, Component)` once per built-in. Imported from
`react-shared/src/index.js` so the registry is populated on first import.

### 5.4a Per-widget version + `adapt()` migration hook (**A2 — DEFERRED**)

> **v4.4 status: DEFERRED to first external consumer (or v1.0,
> whichever comes first).** Critical-review evidence: v4.3's three
> initial consumers are all coordinatable within one sprint via
> library-level semver (§8.4); per-widget versioning forces every
> `registerWidget()` call to carry `{version: N}` forever for a benefit
> we don't yet have. **The simpler `UnknownWidgetFallback` (§5.4b)
> covers the "server sent an unknown widget kind" case** — which is
> the only real failure mode in the v0.1.0 → v1.0.0 window with three
> internal consumers. Re-evaluate when: (a) an external consumer
> appears, OR (b) a single widget's payload shape needs to evolve
> incompatibly while ≥ 2 consumers remain on the old shape, OR (c) the
> library hits v1.0.

Future design (kept for reference, not built in v0.1.0):

```js
// Registry registration would carry (kind, version, Component):
registerWidget('text_input',          TextInputWidgetV2, { version: 2 });
// Adapter for V1 → V2 messages:
registerAdapter('text_input', { from: 1, to: 2 }, (msgV1) => ({
  ...msgV1, version: 2, fields: msgV1.fields.map(f => ({...f, required: f.required ?? false})),
}));
```

**v0.1.0 reality:** `registerWidget(kind, Component)` — no version
parameter. The Python `WidgetMessage` may still carry an optional
`version: int = 1` field for forward-compat (cheap, ignored by
v0.1.0 clients), but the JS registry does not dispatch on it.

### 5.4b `<UnknownWidgetFallback />` with telemetry (**A14**)

A registry miss MUST NOT crash. Library ships a fallback component:

```jsx
function UnknownWidgetFallback({ widget }) {
  React.useEffect(() => {
    if (typeof window !== 'undefined' && window.__af_telemetry__) {
      window.__af_telemetry__('widget.unknown', {
        widget_type: widget.widget_type,
        version:     widget.version ?? 'unspecified',
        widget_id:   widget.widget_id,
      });
    }
    console.warn(`[shared-ui] Unknown widget: ${widget.widget_type}@${widget.version ?? '?'}`);
  }, [widget]);
  return (
    <Alert severity="warning">
      Unknown widget <code>{widget.widget_type}</code>
      {widget.version ? ` v${widget.version}` : ''}.
      The server may be ahead of this client.
    </Alert>
  );
}
```

Telemetry hook is **optional** — consumers wire `window.__af_telemetry__`
to whatever (Sentry, custom HTTP collector) they want. Missing hook =
no-op; only the console warning fires.

### 5.4c JSON Schema as the Py↔TS contract pivot (**A1 — DEFERRED to optional Phase 3+ upgrade**)

> **v4.4 status: DEFERRED.** Critical-review evidence: v4.3's existing
> `sync_widget_types.py` codegen (§5.1) is ~40 LoC, reads Python
> constants, writes JS constants, and is **sufficient for v0.1.0**.
> The JSON-Schema upgrade adds a 2-script chain (`widget_schema.py`
> + `generate_protocol_ts.mjs`) plus a new runtime dep (`ajv`) for
> marginal benefit given v4.3's other decisions:
>   - **D15: JS + JSDoc, NOT TypeScript-first** → the richer TS types
>     that JSON-Schema-to-TS produces have no JS+JSDoc consumer.
>   - **D10: no pydantic** → the pydantic round-trip benefit is moot.
>   - **3 internal consumers** → dev-mode runtime validation catches
>     only what `WidgetMessage.from_dict` already validates at the
>     Python boundary.
>
> **Re-evaluate when**: (a) v4.3 adopts TypeScript at v0.2.0, OR
> (b) ≥ 1 external consumer of `@agent-foundation/shared-ui` exists,
> OR (c) a wire-format incident demonstrates `sync_widget_types.py`
> + Python boundary validation are insufficient.

**v0.1.0 reality (retained from §5.1):** `sync_widget_types.py`
emits plain string constants + frozen `WIDGET_TYPES` array. CI runs
`--check` on every PR. The Python `WidgetMessage.from_dict` already
raises on unknown `widget_type` (verified) — that's the v0.1.0
validation gate.

**Future design** (~150 LoC total, kept for reference, not built in
v0.1.0):

```
agent_foundation/ui/widget_schema.py        # attrs introspection → JSON Schema
react-shared/scripts/generate_protocol_ts.mjs  # JSON Schema → widget.generated.ts
react-shared/schemas/widget-spec.schema.json    # GENERATED
react-shared/src/protocol/widget.generated.ts   # GENERATED
```

3-step drift gate `scripts/check-protocol-drift.sh`:
```bash
python -m agent_foundation.ui.widget_schema --out react-shared/schemas/widget-spec.schema.json
cd react-shared && node scripts/generate_protocol_ts.mjs
git diff --exit-code react-shared/schemas/ react-shared/src/protocol/widget.generated.ts
```

Dev-mode validation (opt-in via env var, no-op in prod):
```js
if (process.env.NODE_ENV !== 'production' && process.env.AF_UI_VALIDATE) {
  const validate = new Ajv().compile(schema);
  if (!validate(message)) console.error('[shared-ui] WidgetMessage invalid', validate.errors);
}
```

Importantly the future upgrade is **fully additive** — the v0.1.0
`sync_widget_types.py` constants stay as a stable thin layer that the
JSON-Schema path produces as a subset. No re-architecture needed when
we adopt it.

**Tree-shaking caveat (acknowledged):** the auto-registration call is
a side effect (it mutates the `Map`), so bundlers cannot tree-shake
unused widgets. All 12 built-ins ship in every consumer bundle. At
current scale this is ~50 KB gzipped — acceptable. If the catalog
grows past ~30 widgets, switch to **opt-in `registerCatalog()`**
helpers (one per widget bucket) so consumers can drop unused buckets.
Tracked as `OPEN-Q: catalog modularity` in §13.

### 5.5 Phase-2 tests

| Test | Asserts |
|------|---------|
| `registry_completeness.test.js` | Every `WIDGET_TYPES` member returns a real component from `getWidget`. |
| Per-widget snapshot tests | Render against canned `WidgetMessage` JSON. |
| `single_choice_double_submit.test.js` | Two consecutive submit clicks → `onSubmit` fired once. |
| `single_choice_post_submit_view.test.js` | After submit, component renders read-only. |
| `confirmation_view_callback.test.js` | `onView` invoked when `metadata.view` is set and the button is clicked. |
| `legacy_adapter.test.js` | Old `{config, onSubmit}` still mounts (this is the only legacy shape — see §5.3); `widget` shape passes through unchanged; deprecation warning fires exactly once across multiple mounts. |

### 5.5b Contract fixture suite (**A9**)

20+ canonical `WidgetMessage` JSON fixtures (one per built-in `kind`,
plus edge cases — empty fields, max-length strings, multi-choice with
0/1/N options, version-skew, all `InputMode` modes) live in:

```
react-shared/contract/fixtures/
  text_input_basic.json
  text_input_required.json
  single_choice_3_options.json
  multiple_choice_select_all.json
  confirmation_with_view.json
  ...
```

Both languages round-trip every fixture in CI:

```python
# test_contract_python.py
@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_widget_message_roundtrip(fixture):
    msg = WidgetMessage.from_dict(json.loads(fixture.read_text()))
    assert msg.to_dict() == json.loads(fixture.read_text())
```

```js
// contract.test.js  (v0.1.0 — no JSON Schema dep; see §5.4c)
import { getWidget, WIDGET_TYPES } from '../../src/protocol/WidgetRegistry';
import { render } from '@testing-library/react';

ALL_FIXTURES.forEach(({ name, data }) => {
  test(`${name} round-trips and renders`, () => {
    // Sanity: fixture's widget_type is a known constant.
    expect(WIDGET_TYPES).toContain(data.widget_type);
    // Smoke: the widget renders without throwing.
    const { Component } = getWidget(data.widget_type);
    expect(() => render(<Component widget={data} onSubmit={()=>{}}/>)).not.toThrow();
  });
});
// (JSON-Schema dev-mode validation can be opted into later — see §5.4c.)
```

Aggregated CI report written to `react-shared/contract/report.json`.

### 5.6 Phase-2 exit criteria

- All built-in widgets in `react-shared/src/inputs/`.
- `npm run sync:check` green (v4.4: `sync_widget_types.py` constant codegen; JSON Schema upgrade deferred per §5.4c).
- Registry completeness test green.
- **20+ contract fixtures round-trip in both Python and JS** (Python via `WidgetMessage.from_dict/to_dict`; JS via `render(<Component widget={fixture}/>)` smoke — no JSON-Schema validation in v0.1.0 per §5.4c).
- **`UnknownWidgetFallback` renders + emits telemetry** when registry miss is forced.
- Snapshot tests green.
- AF demo visually unchanged.

---

## 6. Phase 3 — Extended primitives & hooks (2 weeks)

**Goal:** Promote the second tier (chat, layout, progress, graph, hooks).

**Timeline rationale (v4.1 correction):** v4.0 estimated 1 week, but
this phase contains **7 LIFT-AND-GENERALIZE items**
(`AgentStatusBar` decomposition, `AgentStreamDrawer`/`Section`
extraction, `PendingReasonPopover` generalization, `BackendSelector`
→ `ModelSelector`, `useProgress`, `useWorkspace`, plus
`SplitActionButton`/`ViewTabBar` shells from §5.5). Each
LIFT-AND-GENERALIZE involves API design + at least one soak release,
not just a file move. Two weeks is realistic; one is not.

### 6.1 Components added in Phase 3

| Bucket | Items |
|--------|-------|
| `chat/` | `ChatInput`, **`ChatMessage`** (AF/RE; 58-line bare bubble — used for simple user/system text rendering), `StreamingMessage`, `CommandAutocomplete`, `Breadcrumb` (OS), `ThinkingFold` (OS, generalized), `PromptViewerDrawer` (OS), **`AgentMessageBubble`** (OS; 227-line rich agent renderer with thinking/response phases, session-context strip, prompt-viewer integration — used when displaying *agent* turns that may include `<Response>`-tagged content). **Relationship:** `AgentMessageBubble` does **not** subsume `ChatMessage`; they coexist. `ChatMessage` renders one chat bubble (any role); `AgentMessageBubble` composes a `ChatMessage` (or its own primitive) with `ThinkingFold` + `PromptViewerDrawer` for agent turns. Apps choose per message: simple user messages → `ChatMessage`; agent turns with phases → `AgentMessageBubble`. Documented in `react-shared/README.md` decision table. |
| `nav/` | **`ViewTabBar`** (RE — generic tab bar with active-tab indicator; promoted because OS/AF have copy-paste equivalents in `Sidebar.js`). |
| `layout/` | `AppHeader` (reconcile AF/RE), `FileViewer` (reconcile 3 forks), `ConnectionStatusBar` (OS, generalized), `FolderTree` (OS). |
| `progress/` | `ProgressSection`, `CompletedSection`, `TaskProgressBar`, `TaskProgressPanel` (reconcile AF/RE). |
| `graph/` | `GraphFlowView` (OS), `NodeDetailPanel` (OS). |
| `messages/` | `PreMessage` (reconcile AF/RE). |
| `queries/` | `QueryCard`, `EditableList` (renamed from `EditableQueryList`), `AddDropdown` (generic). `AddQueriesDropdown` stays in-app as a thin wrapper. |
| `actions/` | `SuggestedActions` (reconcile AF/RE). |
| `hooks/` | `useChat`, **`useAgentChat`** (verified byte-identical AF↔RE → PROMOTE as-is), **`useAgentWebSocket`** (verified **differs** → CANONICALIZE with `messageAdapter` prop for the small RE-specific filtering). |

### 6.2 LIFT-AND-GENERALIZE items

| Item | What changes |
|------|--------------|
| `useProgress` | RE-shape-specific. Extract `useProgress({ items, completedPredicate, sortFn })`. RE wraps it. |
| `useWorkspace` | RE-shape-specific. Extract `useWorkspace({ fetchTree, fetchFile })`. |
| `AgentStatusBar` | Decompose into `<ConnectionStatus>`, `<ModelSelector>`, `<TargetPathChip>`. Apps re-compose. |
| `AgentStreamDrawer` | Extract `<StreamDrawer>` shell. |
| `AgentStreamSection` | Extract `<CollapsibleStreamSection>` shell. |
| `PendingReasonPopover` (OS) | Generalize: `resolutions[]` + `onResolve(reason, value)` props (drop OpenTEAM resolution map). |
| `BackendSelector` (OS) | Becomes `<ModelSelector value options onChange />`. |

### 6.3 KEEP-IN-APP (domain coupling)

- **OS**: `SessionContextBar`, `Sidebar` (OS-flavored), `SettingsDrawer`,
  `TaskCard`, `TaskPanel`, `components/{employees,projects,tasks,teams,views}/*`,
  `useGraphState`, `useManagerChat`, `usePromptViewer`,
  `useServerBackends`, `useTaskTopologies`.
- **RE**: `components/hub/*`, `components/views/*`,
  `ProposalSelectionWidget`, `MultiViewTaskPanel`, `PipelineStatusBar`,
  RE-specific hooks (`useAutopilot`, `useComboOverrides`,
  `useProposalOverrides`, `useSubmissionTemplates`, `useViewWorkspace`,
  `useHypothesisImplementations`, `useAggregationSettings`,
  `useArchiveCount`).

### 6.4 Phase-3 exit criteria

- All Phase-3 components in `react-shared/src/`.
- AF demo uses no local copy of any promoted component.
- Snapshot tests green.

---

## 7. Phase 4 — Consumer migration (parallelizable; 1–2 weeks per consumer)

### 7a. RankEvolve — Python first, then React

RE is the most invasive consumer because of its parallel Python fork.

**7a.1 Replace `agentic_foundation.common.ui` imports — use libcst, not sed.**

Naive `sed` regexes miss multi-line imports such as

```python
from rankevolve.src.agentic_foundation.common.ui.widget_protocol import (
    WidgetMessage,
    WidgetResponse,
)
```

which is common in Python codebases. Use a `libcst` codemod instead
(stored at `scripts/codemods/rewrite_re_to_af_imports.py`):

```python
"""Rewrite `from rankevolve.src.agentic_foundation.common.ui...` to
`from agent_foundation.ui...`.

Uses libcst's get_full_name_for_node helper rather than walking the
Attribute tree manually — the Attribute structure is nested
(.value/.attr recursively), not a flat .children list, so naive
flattening misses leaves.
"""
import libcst as cst
from libcst.codemod import VisitorBasedCodemodCommand
from libcst.helpers import get_full_name_for_node_or_raise

OLD = "rankevolve.src.agentic_foundation.common.ui"
NEW = "agent_foundation.ui"

class RewriteImports(VisitorBasedCodemodCommand):
    DESCRIPTION = "Rewrite RE agentic_foundation.common.ui → agent_foundation.ui"

    def leave_ImportFrom(self, original_node, updated_node):
        if updated_node.module is None:
            return updated_node
        try:
            dotted = get_full_name_for_node_or_raise(updated_node.module)
        except Exception:
            return updated_node
        if dotted == OLD or dotted.startswith(OLD + "."):
            new_dotted = NEW + dotted[len(OLD):]
            return updated_node.with_changes(
                module=cst.parse_expression(new_dotted),
            )
        return updated_node
```

Run (`libcst` ≥ 1.0):
```bash
python -m libcst.tool codemod \
    scripts.codemods.rewrite_re_to_af_imports.RewriteImports \
    rankevolve/src/
```

The second codemod (`rewrite_re_debuggable_to_canonical.py` for §7a.4)
is structurally identical with `OLD = "rankevolve.src.utils.common_objects.debuggable"`
and `NEW = "rich_python_utils.common_objects.debuggable"`.

Verification step (mandatory before merge):
```bash
grep -rn "agentic_foundation\.common\.ui" rankevolve/src/   # must return 0
ruff check rankevolve/src/                                  # no new errors
pytest rankevolve/test/ -x                                  # green
```

**7a.2 `MULTIPLE_CHOICES` → `MULTIPLE_CHOICE`.** Same libcst codemod
(`RewriteEnumMember`) — a `grep | sed` would miss attribute access in
keyword arguments and conditional expressions. Wire-format alias from
§3.1 preserves backwards-compat for any message already in flight.

**7a.3 Delete RE Python fork** (8 files under
`rankevolve/src/agentic_foundation/common/ui/`):
`widget_protocol.py`, `input_modes.py`, `interactive_base.py`,
`rich_interactive_base.py`, `queue_interactive.py`,
`interactive_checkpoint.py`, `terminal_interactive.py`,
`web_interactive.py`. Keep `proposal_models.py`, `proposal_parser.py`,
`learnings_parser.py` (RE domain).

**7a.4 Adopt `rich_python_utils.common_objects.debuggable` — large scope.**

Verified this session: **15 files** across RE import from
`rankevolve.src.utils.common_objects.debuggable` (NOT just
`web_interactive.py`). They span:

| Area | Files |
|------|-------|
| `agentic_foundation/common/ui/` | `interactive_base.py` |
| `agentic_foundation/common/inferencers/` | `inferencer_base.py`, `flow_inferencers/{dual_inferencer.py, plan_then_implement_inferencer.py, reflective_inferencer.py, linear_workflow_inferencer.py}` (5 files) |
| `server/` | `research_propose_bridge.py`, `rankevolve_service/session_manager.py`, `dual_inferencer_bridge.py`, `tool_executor.py` (4 files) |
| `utils/service_utils/session_management/` | `session_base.py`, `session_manager.py` (2 files) |
| `utils/common_objects/workflow/` | `workgraph.py`, `common/worknode_base.py` (2 files) |
| `cli/chat_cli/` | `dual_inferencer_cli.py` (1 file) |

Strategy: a **single second libcst codemod**
(`scripts/codemods/rewrite_re_debuggable_to_canonical.py`) rewrites:

```
rankevolve.src.utils.common_objects.debuggable
  →
rich_python_utils.common_objects.debuggable
```

After codemod, `rankevolve/src/utils/common_objects/debuggable.py`
(the shim) becomes unreferenced. **Verify zero importers** with
`grep -rn "rankevolve\.src\.utils\.common_objects\.debuggable" rankevolve/src/`
then delete the shim file. **No new module in AF is created.**

**7a.4-API-compat check.** Before running this codemod, confirm
`rich_python_utils.common_objects.debuggable.Debuggable` exposes the
same public surface the RE fork relies on (class name, method
signatures of any methods called outside the constructor). If a method
is RE-specific, push it upstream into `rich_python_utils` as a
backward-compatible addition **before** the codemod runs. Acceptance:
all `pytest` suites in RE green after the codemod.

**7a.5 React migration.**
Add `"@agent-foundation/shared-ui"` to
`rankevolve/src/webui/react/package.json` as the Artifactory tarball.
Replace imports per §9 matrix. Keep RE-specific widgets locally and
register them at bootstrap:
```js
registerWidget('rankevolve.proposal_selection', ProposalSelectionWidget);
```

**7a.6 RE exit criteria.**
- `pytest` green in RE.
- `from agent_foundation.ui import WebUIInteractive` works inside RE.
- RE's `npm start` for the React app renders widgets; visual snapshot diff zero.
- `find rankevolve/src/agentic_foundation/common/ui/ -name '*.py' | grep -vE '(proposal_(models|parser)|learnings_parser)\.py$'` returns **0**.
- `grep -r 'rankevolve\.src\.utils\.common_objects\.debuggable' rankevolve/src/` returns **0**.

### 7b. OpenStartup — React only (no Python fork)

**7b.1 Add dependency** to `openteam/ui/package.json`:
```json
"@agent-foundation/shared-ui":
  "file:../../../../AgentFoundation/src/agent_foundation/ui/react-shared"
```
(4 `..`s; works under the typical `CoreProjects/{AgentFoundation,OpenStartup}/` layout).

**7b.2 Replace `shared/`** with re-export shim:
```js
// openteam/ui/src/shared/index.js
export {
  EmptyState, LoadingIndicator, PersonChip, ProgressBar,
  QuickLinkButton, SectionCard, StatusBadge,
} from '@agent-foundation/shared-ui';
export { default as PendingReasonPopover } from './PendingReasonPopover';
```
Delete the 7 individual `shared/*.js` files. `PendingReasonPopover`
stays local until §6.2 promotes it.

**7b.3 Replace chat-widgets** + register domain widgets:
```js
// openteam/ui/src/components/chat-widgets/index.js
export * from '@agent-foundation/shared-ui';
```
At app bootstrap (`openteam/ui/src/index.js`):
```js
import { registerWidget } from '@agent-foundation/shared-ui';
registerWidget('openteam.project_summary',  ProjectSummaryWidget);
registerWidget('openteam.sprint_progress',  SprintProgressWidget);
registerWidget('openteam.task_assignment',  TaskAssignmentWidget);
registerWidget('openteam.task_list',        TaskListWidget);
registerWidget('openteam.workload_chart',   WorkloadChartWidget);
```

**7b.4 Replace theme** with library re-exports; add OS-flavored theme via:
```js
import { registerTheme } from '@agent-foundation/shared-ui/theme';
registerTheme('openstartup', osTheme);
```

**7b.5 Replace chat/* generics**:
`chat/{MarkdownRenderer, StreamingMessage, Breadcrumb, CommandAutocomplete,
PromptViewerDrawer, ThinkingFold, AgentMessageBubble, GraphFlowView,
NodeDetailPanel}.js` → library re-exports. Keep
`SessionContextBar`, `TaskCard`, `TaskPanel`, `BackendSelector` local
(BackendSelector until §6.2 extracts ModelSelector).

**7b.6 OS exit criteria.**
- `npm install && npm run start` works.
- Screenshot diff baseline: zero unintended diffs.
- `grep -rE "from '.*shared/(EmptyState|LoadingIndicator|PersonChip|ProgressBar|QuickLinkButton|SectionCard|StatusBadge)'" openteam/ui/src/` returns **0** outside the re-export shim.
- `grep -rE "from '.*chat-widgets/(TextInput|SingleChoice|MultipleChoice|Confirmation|Approval|Choice)Widget'" openteam/ui/src/` returns **0** outside re-export shims.

### 7c. AgentFoundation demo app cleanup

- Reduce `webui/react/src/components/{widgets,common,chat,layout,progress,messages,queries,actions}/`
  to re-export shims **or delete entirely** if no external code imports
  from them.
- `webui/src/` (legacy 47-file duplicate) already removed in Phase 0 §3.9.
- `webui/backend/main.py` unchanged.

---

## 8. Phase 5 — Governance & anti-drift (ongoing)

### 8.1 CI guardrails (AgentFoundation repo)

| Check | Script | Fails when |
|-------|--------|------------|
| Py ↔ JS sync | `python react-shared/scripts/sync_widget_types.py --check` | `widget_protocol.py` or `input_modes.py` edited without regen. |
| Registry completeness | `vitest run tests/registry_completeness.test.js` | `WIDGET_TYPES` member has no registered component. |
| No duplicate components | `python scripts/check_no_duplicate_widgets.py` | Any file in `webui/react/src/components/{widgets,common}/` is not a one-line re-export of the same-named file in `react-shared/src/`. |
| Protocol round-trip | `pytest test/agent_foundation/ui/test_widget_protocol_roundtrip.py` | Wire format breaks. |
| `dist/` rebuilt | `npm run build && git diff --exit-code dist` | `react-shared/dist/` is stale. |
| CODEOWNERS on contract files | GitHub native | `widget_protocol.py`/`input_modes.py` edited without UI-team review. |

### 8.2 CI guardrails (OS + RE repos)

A `check_no_local_widget_duplicates.py` (~50 lines) walks each repo and
fails the build if any file in
`{shared,chat-widgets,components/widgets,components/common}/` duplicates
a name in the library's published manifest
(`@agent-foundation/shared-ui/manifest.json`).

The `manifest.json` is produced by the `tsup` post-build step defined
in §4.8 (Phase 1 deliverable). It enumerates every public export
name + the `WIDGET_TYPES` list. Consumer scripts download it via
`node -p "require('@agent-foundation/shared-ui/dist/manifest.json')"`
or by reading directly from the installed `node_modules/` tree.

### 8.3 `react-shared/CONTRIBUTING.md`

1. Every new widget MUST add a constant to `widget_protocol.py` first.
2. The PR MUST regenerate `widgetTypes.js` via `npm run sync`.
3. The PR MUST add the widget to `registerBuiltins.js`.
4. The PR MUST add a Vitest snapshot test.
5. The PR MUST conform to Contract 2 props.
6. Domain widgets MUST use `namespace.type` and live in the consuming app.

### 8.4 Versioning

Semver. `0.1.0` after Phase 2. `0.2.0` removes `legacyAdapter.js`.
`1.0.0` once OS & RE have shipped on `0.x` without compat shims.

### 8.5 Quarterly drift audit

Cron CI runs `diff -rq react-shared/src/ <consumer>/src/.../widgets/`
and opens an auto-generated Jira ticket on any unexpected file.

### 8.6 Performance & bundle-size SLOs (**A6**)

Concrete, machine-checkable performance gates (declared baseline:
GitHub Actions `ubuntu-latest`, 4-vCPU, 16 GB RAM; baseline snapshot
in `react-shared/bench/baseline.json`):

| SLO | Target | Tool | Fail when |
|-----|--------|------|-----------|
| Streaming markdown render | < 8 ms / 1 KB chunk | `vitest bench` | regression > 15% vs baseline |
| Registry lookup | < 1 µs | `vitest bench` | regression > 25% vs baseline |
| Bundle `index.mjs` | ≤ 75 KB gzipped | `size-limit` | regression > 5% vs baseline |
| Bundle `theme.mjs` | ≤ 12 KB gzipped | `size-limit` | regression > 5% |
| Bundle `protocol.mjs` | ≤ 8 KB gzipped | `size-limit` | regression > 5% |

`size-limit` config (`react-shared/.size-limit.json`):
```json
[
  { "path": "dist/esm/index.mjs",    "limit": "75 KB", "gzip": true },
  { "path": "dist/esm/theme.mjs",    "limit": "12 KB", "gzip": true },
  { "path": "dist/esm/protocol.mjs", "limit":  "8 KB", "gzip": true }
]
```

CI step:
```bash
npm run build && npx size-limit
npm run bench && node scripts/check_perf_regression.mjs bench/baseline.json bench/latest.json
```

### 8.7 Accessibility gate (**A7**)

Every Storybook story is wrapped in a `@storybook/addon-a11y` panel;
CI runs `@storybook/test-runner` with `jest-axe` per story:

```js
// .storybook/test-runner.js
const { injectAxe, checkA11y } = require('axe-playwright');
module.exports = {
  async preVisit(page) { await injectAxe(page); },
  async postVisit(page) {
    await checkA11y(page, '#storybook-root', {
      detailedReport: true,
      detailedReportOptions: { html: true },
      axeOptions: { runOnly: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'] },
    });
  },
};
```

CI fails on any **critical** or **serious** axe violation. `moderate`
and `minor` are surfaced as PR comments but non-blocking.

### 8.8 Rollback policy (**A10** — semver-coherent)

If a release `X.Y.Z` causes regressions in a consumer:

| Severity | Window | Action |
|----------|--------|--------|
| Broken at install (npm error) | < 4 h after publish | `npm unpublish` (allowed by registry within 72 h); pin consumers to last good. |
| Broken at runtime, < 1 consumer affected | < 24 h | Publish `X.Y.(Z+1)` reverting the offending change. |
| Broken at runtime, ≥ 2 consumers affected | < 24 h | Publish `X.Y.(Z+1)` reverting **and** open a post-mortem RFC. |
| Deletion in `1.0.0` causes broken consumer | any time | Publish **`1.0.1` restoring deleted paths as deprecated re-exports** — consumers pinning `^1.0` receive the restoration automatically. Re-attempt hard-delete only in `2.0.0` after a deprecation window with logged usage telemetry. |

The `1.0.1`-restore path is the v4.3 default rollback strategy for
post-major-bump regressions — **semver-coherent**, no need to
pre-release a `0.99.x` patch. (Aggregator A10.)

---

## 9. Component Classification Matrix

Legend: **PROMOTE** / **CANONICALIZE** / **LIFT-AND-GENERALIZE** / **KEEP-IN-APP** / **DELETE**.

### 9.1 Python

| File | Decision | Note |
|------|----------|------|
| AF `widget_protocol.py` | CANONICAL | +6 constants + `WIDGET_TYPES` (§3.7) |
| AF `input_modes.py` | CANONICAL | +`ChoiceOption.description`, `'multiple_choices'` alias (§3.1) |
| AF `interactive_base.py` | CANONICAL | +`aget_input`/`asend_response` (§3.2) |
| AF `rich_interactive_base.py` | CANONICAL | +`supports_widgets`/`pending_input_mode` (§3.3) |
| AF `queue_interactive.py` | CANONICAL | +asyncio.Queue + turn_boundary + token_batches (§3.4) |
| AF `terminal_interactive.py` | CANONICAL | webaxon lazy + optional (§3.6) |
| AF `web_interactive.py` | **NEW** | Ported from RE, Debuggable from rich_python_utils (§3.5) |
| AF `interactive_checkpoint.py` | unchanged | bug auto-fixed by §3.2 |
| AF `email_interactive.py`, `simulated_interactive.py` | DELETE | empty stubs |
| AF `__init__.py` | CANONICAL (rewrite) | public barrel (§3.8) |
| AF `dash_interactive/`, `graph_*.py` | KEEP | out of scope |
| RE Python fork (8 files) | DELETE | §7a.3 |
| RE `proposal_models.py`, `proposal_parser.py`, `learnings_parser.py` | KEEP-IN-APP | RE domain |
| RE `utils/common_objects/debuggable.py` | DELETE | use `rich_python_utils` (§7a.4) |

### 9.2 Input widgets

| Widget | Decision | Source |
|--------|----------|--------|
| TextInput, SingleChoice, Confirmation | CANONICALIZE | **OS** (submit-guard, view callbacks) |
| MultipleChoice, Dropdown, Toggle, ToolArgumentForm, MultiInput, Default | CANONICALIZE / PROMOTE | **AF** |
| Grouped | LIFT-AND-GENERALIZE | **RE** |
| Approval, CardChoice (`card_choice`) | PROMOTE | **OS** |
| ChatWidgetRenderer, ConversationToolWidget | PROMOTE | **OS** (with `parser` prop) |
| WidgetRegistry | CANONICALIZE | AF + add `registerWidget` (§4.4) |
| ToolConfigPanel | CANONICALIZE | AF |
| ProposalSelection, ProjectSummary, SprintProgress, TaskAssignment, TaskList, WorkloadChart | KEEP-IN-APP | RE/OS domain |

### 9.3 Pure UI primitives

| Component | Decision |
|-----------|----------|
| EmptyState, LoadingIndicator, PersonChip, ProgressBar, QuickLinkButton, SectionCard, StatusBadge, WelcomeScreen, PlanModeSelector, ClickToEditMarkdown | CANONICALIZE (Phase 1) |
| MarkdownRenderer | CANONICALIZE — merge OS preprocess + optional chaining |
| SplitActionButton (RE) | PROMOTE (Phase 2) |
| PendingReasonPopover (OS) | LIFT-AND-GENERALIZE (Phase 3) |
| Breadcrumb (OS), ViewTabBar (RE) | PROMOTE (Phase 3) |
| BackendSelector → ModelSelector | LIFT-AND-GENERALIZE (Phase 3) |
| FileViewer, AppHeader | CANONICALIZE (Phase 3) |
| FolderTree (OS) | PROMOTE (Phase 3) |
| ConnectionStatusBar (OS) | LIFT-AND-GENERALIZE (Phase 3) |
| GraphFlowView, NodeDetailPanel (OS) | PROMOTE (Phase 3) |
| AgentStatusBar, AgentStreamDrawer, AgentStreamSection | LIFT-AND-GENERALIZE (Phase 3) |
| SettingsDrawer, Sidebar (OS), TaskCard, TaskPanel, SessionContextBar, SessionSidebar, AgentChatPanel, MultiViewTaskPanel, PipelineStatusBar | KEEP-IN-APP |

### 9.4 Hooks

| Hook | Decision |
|------|----------|
| useApiData, useServerStatus (OS) | PROMOTE (Phase 1) |
| useFileViewer, useContextMenu, useInputFields, useSectionVisibility, useProgressHeader | CANONICALIZE |
| **useChat** | PROMOTE as-is (verified byte-identical AF↔RE this session) |
| **useAgentChat** | PROMOTE as-is (verified byte-identical AF↔RE) |
| **useAgentWebSocket** | CANONICALIZE with `messageAdapter` prop (verified differs between AF and RE) |
| useProgress, useWorkspace | LIFT-AND-GENERALIZE |
| useSessionApi, useSessionManager, useGraphState, useManagerChat, usePromptViewer, useServerBackends, useTaskTopologies, plus all RE-specific hooks | KEEP-IN-APP |

### 9.5 Theme

| File | Decision |
|------|----------|
| createAppTheme, cssVariableBridge, ThemeProvider, themeRegistry, ThemeSwitcher | CANONICALIZE |
| themes/{dark,atlassian,pinterest} | PROMOTE from OS |
| ThemeProvider `createThemeFn` prop (verified) | KEEP — bridges MUI 5 ↔ MUI 7 |

---

## 10. Risks & mitigations (evidence-backed)

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Async base-class change breaks RE override | LOW | HIGH | Phase-0 signatures match RE's; cross-test both paths. |
| R2 | `Debuggable` already-exists conflict | LOW | LOW | Use existing `rich_python_utils.common_objects.debuggable`; no new shim. |
| R3 | MUI 5 vs MUI 7 divergence | MEDIUM | MEDIUM | `peerDep >=5 <8`; verified `createThemeFn`; CI matrix tests both. |
| R4 | CRA does not transpile `node_modules` | HIGH (would-be) | HIGH | **`tsup` pre-build** of CJS+ESM; no `craco` needed. |
| R5 | `theme.custom.surfaces.*` bare access crashes in `ProgressSection`/`TaskProgressBar`/`CompletedSection` | HIGH (already happens) | MEDIUM | Phase 3 mandates optional-chaining rewrite + `verifyThemeContract(theme)` test helper. |
| R6 | OS submit-guards lost during promotion | MEDIUM | MEDIUM | Phase-2 names OS as source-of-truth + double-submit test. |
| R7 | Py↔JS widget-type drift returns | MEDIUM | HIGH | Codegen + CI `--check` (§5.1 + §8.1). |
| R8 | Big-bang migrations cause merge conflicts | MEDIUM | MEDIUM | Per-bucket PRs; each independently shippable. |
| R9 | React 18 (AF/RE) vs React 19 (OS) | LOW | MEDIUM | `peerDep react>=18`; CI tests both. |
| R10 | Artifactory publishing pipeline missing | MEDIUM | MEDIUM | Phase 1 sets up `tsup`; Phase 5 sets up publish job. Interim: `npm pack`. |
| R11 | ConversationToolWidget app-parsing leaks | HIGH | MEDIUM | Phase-2 extracts `parser` prop (default = identity). |
| R12 | RE's React fork sometimes ahead of AF | HIGH | LOW | §1 verified ground-truth dictates per-file source; `DECISIONS.md` logs each. |
| R13 | `interactive_checkpoint.py` async bug bites prod | LOW | HIGH | Phase-0 fixes first + regression test. |
| R14 | `MULTIPLE_CHOICES` plural alias confuses devs | LOW | LOW | Docstring + lint rule. |
| R15 | OS `PendingReasonPopover` resists generalization | MEDIUM | LOW | Lift only after one soak release. |
| R16 | webaxon missing in deploy env crashes terminal | MEDIUM | MEDIUM | Phase-0 §3.6 lazy + optional. |
| R17 | **Debuggable migration is 15 files, not 1** (v4.0 understatement) | HIGH | MEDIUM | Phase 4a.4 corrected this session; libcst codemod + API-compat check. |
| R18 | **`@attrs`-decorated `QueueInteractive` silently drops class-level defaults** if extension uses plain `__init__` syntax | MEDIUM | MEDIUM | Phase 0 §3.4 example uses `attrib(default=…, kw_only=True)` explicitly. |
| R19 | **Storybook deferral breaks visual-diff acceptance gate** | (was a v4.0 self-contradiction) | MEDIUM | Phase 1 §4.7 brings Storybook + Chromatic in (no longer deferred). |
| R20 | **`__init__.py` eagerly imports transports** → 200ms+ import latency for lightweight callers | MEDIUM | LOW | Phase 0 §3.8 uses PEP 562 `__getattr__` lazy loading. |
| R21 | **`react-markdown@9` `inline` prop removed; OS uses it; renders broken** | HIGH (live) | MEDIUM | Phase 1 §4.5(d) adopts RE's newline-detection + regression test. (**A4**) |
| R22 | **Server emits widget kind client doesn't know → crash** | LOW (3 controlled consumers) | HIGH if it happens | §5.4b `UnknownWidgetFallback` renders a graceful error + telemetry. v0.1.0 does **not** add per-widget versioning (A2 deferred per §5.4a); the same fallback covers version-skew at the kind level. |
| R23 | **Hidden out-of-tree consumer breaks when `webui/src/` is deleted** | LOW-MED | HIGH | §3.9 mandatory `rg` scan across all CoreProjects/ before deletion; reviewer-confirmed empty results. (**A8**) |
| R24 | **`useTheme` name collides with MUI's `useTheme` → silent shadowing** | LOW | MEDIUM | §4.5(3) library exports `useThemeTokens()` instead. (**A12**) |
| R25 | **MUI 5↔7 prop renames force per-component branches** | MEDIUM | LOW | §4.5(3) `compatProps({v5, v7})` centralizes branching. (**A11**) |
| R26 | **Wire-format drift between Py and JS escapes plain string-constant codegen** | LOW (3 controlled consumers + boundary validation) | MEDIUM | v0.1.0: `sync_widget_types.py --check` CI gate catches constant drift; `WidgetMessage.from_dict` raises on unknown payload at the Python boundary. JSON-Schema upgrade (§5.4c) tracked as a Phase-3+ enhancement, triggered on TS adoption / external consumer / wire-format incident. |
| R27 | **Performance / bundle-size regressions slip in unnoticed** | MEDIUM | LOW | §8.6 concrete SLOs + `size-limit` + `vitest bench` baseline. (**A6**) |
| R28 | **A11y regressions slip in unnoticed** | MEDIUM | LOW | §8.7 `jest-axe` per-story CI gate. (**A7**) |
| R29 | **1.0 deletion regret — need to restore deleted exports post-major** | MEDIUM | MEDIUM | §8.8 publish `1.0.1` re-exports; semver-coherent. (**A10**) |

**Resolved decisions (formerly R17/R18 in v4.0).** These are not active
risks; they are choices already made and recorded in §16 below.

---

## 11. Sequence & timeline

```
Week 1:    Phase 0 (3d) + Phase 1 (3d, starts after Phase 0 day 1)
Week 2:    Phase 2 — widget protocol canonicalization
Weeks 3–4: Phase 3 — extended primitives & hooks (2w; was 1w in v4.0)
Weeks 5–6: Phase 4a (RankEvolve)   ──┐
Weeks 5–6: Phase 4b (OpenStartup)   ──┴ parallelizable
Week 7:    Phase 4c (AF demo cleanup) + Phase 5 (governance setup)
Ongoing:   Phase 5 — drift audits, releases
```

Critical path: **Phase 0 → Phase 1 → Phase 2 → Phase 4a (RE Python +
15-file Debuggable migration)**. Phase 4a is the dominant cost in
Phase 4 because of the 15-file `Debuggable` codemod surface (§7a.4).

---

## 12. Deliverables & acceptance criteria

### 12.1 Files produced

- `_docs/_plan/ui_components_formalization/ui_components_formalization_plan.md` (this file, canonical)
- `_docs/_plan/ui_components_formalization/DECISIONS.md` (rolling log)
- `_docs/_plan/ui_components_formalization/MIGRATION_CHECKLIST.md` (ops tracker)
- `src/agent_foundation/ui/react-shared/` (new package)
- `src/agent_foundation/ui/web_interactive.py` (new file; uses existing `rich_python_utils`)
- `src/agent_foundation/ui/__init__.py` (rewritten)
- `test/agent_foundation/ui/test_*.py` (7 files per §3.10)
- `react-shared/scripts/sync_widget_types.py`
- `scripts/check_no_duplicate_widgets.py`

### 12.2 Objective acceptance gates

1. `pytest test/agent_foundation/ui/` green in AF.
2. `pytest` green in RE after migration.
3. `python -c "from agent_foundation.ui import WebUIInteractive"` works in both AF and RE envs.
4. `python -c "from agent_foundation.ui import TerminalInteractive"` works in an env WITHOUT webaxon installed.
5. `npm run build && npm run test && npm run sync:check` green in `react-shared/`.
6. All three apps run `npm install && npm start` clean.
7. `grep -rE "from '.*shared/(EmptyState|LoadingIndicator|PersonChip|ProgressBar|QuickLinkButton|SectionCard|StatusBadge)'" openteam/` returns **0** outside re-export shims.
8. `grep -rE "from '.*chat-widgets/(TextInput|SingleChoice|MultipleChoice|Confirmation|Approval|Choice)Widget'" openteam/` returns **0** outside shims.
9. `find rankevolve/src/agentic_foundation/common/ui/ -name '*.py' | grep -vE '(proposal_(models|parser)|learnings_parser)\.py$'` returns **0**.
10. `grep -r 'rankevolve\.src\.utils\.common_objects\.debuggable' rankevolve/src/` returns **0**.
11. Visual snapshot diff returns **0** unintended diffs.
12. CI guardrails green in all three repos.

---

## 13. Open questions

1. **Artifactory namespace.** `@agent-foundation/shared-ui` (recommended) vs `@atlassian/agent-foundation-shared-ui`?
2. **TypeScript.** Stay JS+JSDoc through `1.0.0` (recommended) vs move to TS at `0.2.0`?
3. **Storybook hosting.** GitHub Pages (recommended) vs Bitbucket Pipelines artifact?
4. **Compat-shim lifetime.** Remove `legacyAdapter.js` at `0.2.0` (~2 months, recommended) vs `0.3.0` (~4 months)?
5. **RE coordination window.** 1-week release window for Phase 4a vs async best-effort?
6. **`webaxon` long-term.** Lazy import now (Phase 0); should we also extract `is_html_string` into `rich_python_utils.html_utils`?
7. **`rich_python_utils` packaging.** Is it published as a wheel or vendored? If vendored, do AF and RE point at the same checkout, or are there forks?

---

## 14. Appendices

### Appendix A — Pre-flight verification commands (run in this session)

```bash
# Async bug:
grep -n "aget_input\|asend_response" \
  /Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/ui/interactive_base.py
#  → empty (bug confirmed)

# Async bug callers:
grep -n "await.*aget_input\|await.*asend_response" \
  /Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/ui/interactive_checkpoint.py
#  → lines 70, 76, 141, 146, 195, 201

# Webaxon hard import:
grep -n "webaxon" \
  /Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/ui/terminal_interactive.py
#  → line 7

# RE queue primitives:
grep -n "send_turn_boundary\|stream_token_batches\|_heartbeat" \
  /Users/tchen7/MyProjects/CoreProjects/atlassian-packages/rankevolve/src/agentic_foundation/common/ui/queue_interactive.py
#  → 227, 256, 313

# AF does NOT have web_interactive.py yet:
ls /Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/ui/web_interactive.py
#  → No such file or directory

# rich_python_utils canonical Debuggable:
ls /Users/tchen7/MyProjects/CoreProjects/RichPythonUtils/src/rich_python_utils/common_objects/debuggable.py
#  → exists

# AF already imports rich_python_utils:
grep -rE "from rich_python_utils" /Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/
#  → 5 hits under agent_foundation/ui/

# RE already imports rich_python_utils:
grep -rE "from rich_python_utils" /Users/tchen7/MyProjects/CoreProjects/atlassian-packages/rankevolve/src/
#  → 3 hits under rankevolve/src/utils/

# useChat byte-identical:
diff -s /Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/ui/webui/react/src/hooks/useChat.js \
        /Users/tchen7/MyProjects/CoreProjects/atlassian-packages/rankevolve/src/webui/react/src/hooks/useChat.js
#  → identical (verified v4.1)

# useAgentChat byte-identical:
diff -s /Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/ui/webui/react/src/hooks/useAgentChat.js \
        /Users/tchen7/MyProjects/CoreProjects/atlassian-packages/rankevolve/src/webui/react/src/hooks/useAgentChat.js
#  → identical

# useAgentWebSocket differs:
diff -q /Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/ui/webui/react/src/hooks/useAgentWebSocket.js \
        /Users/tchen7/MyProjects/CoreProjects/atlassian-packages/rankevolve/src/webui/react/src/hooks/useAgentWebSocket.js
#  → Files differ

# OS double-submit guard:
grep -n "submitted" \
  /Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/ui/src/components/chat-widgets/SingleChoiceWidget.js
#  → lines 46, 56, 139, 148
```

### Appendix B — Glossary

- **AF** — AgentFoundation. **OS** — OpenStartup / OpenTEAM. **RE** — RankEvolve.
- **Wire protocol** — JSON shape exchanged over WebSocket; defined by `WidgetMessage`.
- **Built-in widget** — one of the 12 types in `WIDGET_TYPES`; ships in `react-shared`.
- **Domain widget** — names a domain concept; stays in app; `ns.type` naming.
- **Turn boundary** — transport marker for renderers to flush partial streaming output (port from RE `queue_interactive.py:227`).

### Appendix C — File-by-file extraction map

See §9 for the full matrix. Phase ordering ensures each batch is a self-contained PR.

---

## 15. If we could ship only ONE plan — which one?

**The updated cryptic-clock plan (2026-05-24 01:41).**

Reasoning:

1. **Both plans have converged.** Both have Phase 0 → 1 → 2 → 3 → 4,
   verified `webaxon` lazy-import, verified `interactive_checkpoint`
   async bug, classification matrix, anti-drift CI, the `tsup` pre-build
   over `craco` analysis, the `useAgentChat`/`useAgentWebSocket`
   correction. There is no architectural disagreement left between them
   that materially affects outcomes.

2. **Cryptic-clock is now leaner and easier to execute.**
   600 lines of mostly action items, with an explicit parallelism
   graph in §0.10 and a tight 4-phase structure. v3.0 has the same
   content plus governance/risk/acceptance ceremony spread across more
   sections; the *signal* density is lower.

3. **On directory naming, neither plan got it right.** Both picked
   `widgets/`. v4.0/v4.1 pick `react-shared/` (sibling to `webui/`)
   because the directory holds non-widget code (theme, hooks, layout,
   progress, common). The **package name** stays
   `@agent-foundation/shared-ui` either way.

4. **Cryptic-clock's two remaining errors are localized and obvious.**
   It still proposes creating `agent_foundation.utils.debuggable`
   (a hack — `rich_python_utils.common_objects.debuggable` already
   exists). It lists RE queue lines off by 1. Both are 5-minute fixes
   on top of a sound plan.

5. **v3.0's strengths are bolt-on additions, not architectural.** The
   §1 ground-truth section, 17-row risk register, and 11-point
   acceptance gates are governance scaffolding that can be added to a
   correct technical plan in an afternoon. The reverse is not true:
   you cannot retro-fit a correctly-fixed `interactive_checkpoint.py`
   onto a UI library that has already shipped on top of a broken
   backend.

**Therefore: ship cryptic-clock if forced to one.** But the *ideal*
outcome is neither alone — it is v4.0 (this file), which takes
cryptic-clock's tight surgical structure, applies its two corrections
(use `rich_python_utils`, fix the off-by-one), and re-attaches v3.0's
governance scaffolding (verified ground-truth, risk register,
acceptance gates, `messageAdapter` for the WebSocket hook
divergence) in a way that does not bury the action items.

---

---

## 16. Decisions log (resolved choices, not risks)

These are choices already made by v4.0/v4.1. They are not tracked as
active risks because they have no failure mode left to mitigate.

| # | Choice | Alternative considered | Why this one |
|---|--------|------------------------|--------------|
| D1 | Use `rich_python_utils.common_objects.debuggable` | Create new `agent_foundation.utils.debuggable` shim (v3.0/cryptic-clock) | Canonical class already exists in workspace; AF imports `rich_python_utils.*` in 5 places under `ui/`; creating another shim is exactly the ad-hoc pattern the user asked to avoid. |
| D2 | Directory name `react-shared/` | `widgets/` (both v3.0 and cryptic-clock §1.1) | Directory holds theme, hooks, layout, progress, common, in addition to widgets — `widgets/` is misleading. Package name (`@agent-foundation/shared-ui`) is independent. |
| D3 | Pre-build CJS+ESM via `tsup` | `craco` source link in every consumer | Consumers shouldn't have to swap build tools. `tsup` keeps CRA setups untouched. |
| D4 | PEP 562 lazy `__init__.py` | Eager imports of all transports | Lightweight consumers (e.g., a script that only handles `WidgetMessage` round-trips) pay zero import cost for transports. |
| D5 | Storybook + Chromatic in Phase 1 | Defer Storybook to 0.2.0 (v4.0) | §12.11 requires "zero unintended visual diffs"; this is only mechanically verifiable with a snapshot tool. |
| D6 | `legacyAdapter.js` handles **one** legacy shape | Heuristic adapter for `{config}`/`{data}`/`{pendingInput}` (feedback claim) | Verified: all three apps use `{config, onSubmit}`. No heuristics needed. |
| D7 | `libcst` codemods over `sed` for Python rewrites | `git grep | xargs sed` | sed misses multi-line imports; libcst preserves comments and formatting. |
| D8 | Single `@attrs` `attrib(default=…, kw_only=True)` syntax for new `QueueInteractive` fields | Plain `field: T = default` class-level | `@attrs` ignores plain class-level defaults for `__init__`-generated fields. |
| D9 | Built-ins auto-register via `registerBuiltins.js` | Opt-in per consumer | 12 widgets, ~50 KB gzipped. Re-evaluate when catalog > 30. |
| D10 | **v0.1.0 keeps plain string-constant codegen** (`sync_widget_types.py`); JSON Schema upgrade is deferred Phase-3+ enhancement | Aggregator proposed pydantic-based JSON-Schema chain (rejected anyway as X6); v4.3 over-built it with attrs introspection | With JS+JSDoc (D15), no pydantic (still rejected), and 3 internal consumers, plain constants + Python boundary validation cover the v0.1.0 risk surface. JSON Schema added later when a real trigger appears (§5.4c). |
| D11 | **Dual-registry coexistence** (`inputModeRegistry` + `richWidgetRegistry`) | Single unified registry (v4.2 implicit) | Verified OS in fact uses two registries with different lifecycles; collapsing would lose functionality. |
| D12 | **No per-widget versioning in v0.1.0** (`registerWidget(kind, Component)` — no version param) | Per-widget version + `adapt()` migration (v4.3 initial) | 12 widgets, 3 internal consumers coordinatable within one sprint via library semver (§8.4). `UnknownWidgetFallback` (A14) covers the only real failure mode. Re-evaluate per §5.4a triggers. |
| D13 | **`useThemeTokens()` (not `useTheme`)** | Mirror MUI's `useTheme` name | Avoids shadow + import-aliasing footgun in consumer apps. |
| D14 | **`compatProps({v5, v7})` helper** | Per-component `if (muiMajor === 7)` branches | Single place to add new MUI prop renames; primitives stay version-agnostic. |
| D15 | **JS+JSDoc, NOT TypeScript-first** (rejected aggregator OQ-6) | TS-first source | All three consumers are JS; TS-first adds toolchain friction with no immediate payoff. `.d.ts` can still be hand-authored for the public surface in `0.2.0`. |
| D16 | **`react-shared/` directory name** (rejected aggregator `widgets/`) | `widgets/` (aggregator) | Directory holds theme/hooks/layout/progress in addition to widgets. |
| D17 | **`file:` deps + npm tarball** (rejected aggregator `pnpm-workspace.yaml`) | Workspace protocol | No monorepo config exists in `CoreProjects/`; `file:` deps work today without new tooling. |
| D18 | **Single peer-dep range `>=5 <8`** (rejected aggregator `^5 \|\| ^7`) | Dual-major range | Cross-major peer ranges are operationally painful; the `>=5 <8` form is functionally equivalent and idiomatic. |
| D19 | **5-phase plan (0–5)** (rejected aggregator 7-phase) | 7 phases including a "promote to 1.0" phase | Version bump is one PR, not a phase. Keeps cognitive load down. |

---

## 17. Rejected items from the aggregator (`output.md`) — audit

These items from the parallel aggregator deliverable were **considered
and rejected**. Each row documents *why*, so future readers don't
re-litigate.

| # | Aggregator proposal | Reason for rejection |
|---|---------------------|----------------------|
| X1 | **TypeScript-first source (OQ-6)** | All three consumers are JS today. Going TS-first means new toolchain (`tsc --noEmit`, `tsconfig.json` per consumer, `.d.ts` plumbing) and JSX-in-TSX migration for zero immediate gain. `.d.ts` can be hand-authored for the public surface in `0.2.0` if needed. (D15.) |
| X2 | **`pnpm-workspace.yaml` workspace protocol** | Verified in this session: no monorepo config exists in `CoreProjects/`. `file:` deps work today (D17). |
| X3 | **Directory name `widgets/`** | Directory holds non-widget code (theme, hooks, layout, progress). v4.3 uses `react-shared/` (D16). |
| X4 | **Dual-major peer-dep range `^5.14.0 \|\| ^7.0.0`** | Functionally equivalent to v4.3's `>=5.0.0 <8.0.0` but less idiomatic; npm warns more (D18). |
| X5 | **`npx @agent-foundation/ui-widgets-codemod`** as a published npm package | Premature packaging. v4.3 ships `scripts/codemods/*.py` (libcst) — simpler and runnable from a checkout. |
| X6 | **pydantic `BaseModel` subclasses in `widget_protocol.py`** | Heavy new dep. `attrs` introspection covers the schema-emission use case (D10). |
| X7 | **7-phase plan (Phase 6 = "promote to 1.0")** | Version bump is one PR, not a phase. v4.3 keeps 0–5 (D19). |
| X8 | **Bundle-as-pip-wheel** (OQ-2 alternative) | No current Python-only consumer renders React. Defer until a real need (e.g., a Dash consumer wanting to bundle JS). |
| X9 | **Promote `useApiData` / `useServerStatus` to library in Phase 1** | These are server-bound hooks; library would need an injectable `fetcher` and endpoint config. v4.3 keeps them in §9.4 CANONICALIZE bucket but defers to Phase 3 with explicit `fetcher`/endpoint props (matches aggregator's REFACTOR sub-label without front-loading). |
| X10 | **Storybook hosting decision (Chromatic vs Loki vs Percy)** (OQ-9) | v4.3 commits to **Chromatic** in §4.7. The aggregator left this open. Choosing now removes a Phase-1 blocker. |
| X11 | **Aggregator's missing Phase 0 (Python contract work)** | Aggregator has **zero Python-side ground-truth analysis** — no `webaxon`/`interactive_checkpoint`/`Debuggable`/`@attrs`/`PEP 562` insight. v4.3's entire Phase 0 (§3) is retained intact. |
| X12 | **Aggregator's CTSC-39558 Jira reference** | OS-specific epic; not portable across consumers. v4.3 leaves Jira epic creation as an OPEN action item rather than baking in a specific ticket ID. |

**Net (v4.3):** the aggregator added 14 high-value items (A1–A14), all
integrated; it also proposed 12 items that are either over-engineered
or contradict v4.2's verified ground-truth, all explicitly rejected
above.

**Net (v4.4):** A critical review of v4.3 itself surfaced that **A1
(JSON Schema chain) and A2 (per-widget versioning) were over-engineered
for v0.1.0** and have been **deferred** (kept in the plan as future
upgrades with explicit trigger conditions in §5.4a / §5.4c). The other
12 A-items (A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14)
remain in v4.4. The reviewer's objection to **A10 (rollback policy)**
was rejected — they claimed the user said "we go forward, no going
back", but that phrase never appears in the user's instructions, and
a rollback policy for a library with semver and external consumers
in `atlassian-packages` is exactly the kind of elegant engineering
the user demanded.

---

*End of v4.4 integrated plan.*




