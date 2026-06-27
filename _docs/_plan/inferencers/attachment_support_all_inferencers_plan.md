<!--
=========================================================================
PROVENANCE — how this plan was produced
=========================================================================
This plan is the verbatim deliverable of a REAL `task`-tool plan-only run
(not hand-authored), then independently spot-verified against source.

  - Command:   python -m agent_foundation.resources.tools.task "<request>" --plan
  - Topology:  default (PTI) -> propose -> BTA breakdown -> 3 parallel workers
               (worker_2 was itself an MFDual with parallel flows)
  - Workspace: _runtime/tasks/task/task_20260626_163910_6f4c940e
  - Result:    rc=0, 485-line unified deliverable (aggregator-consolidated)
  - Run time:  ~2026-06-26 16:39 -> 16:43 PDT

Independent verification performed by the saving agent (Rovo Dev), confirmed
against live source on branch dev_xinli_2601:
  [verified] All 3 AG inferencers expose set_messages(messages) + pass
    prompt_or_messages= to the AI-Gateway backend — the clean, non-hacky
    injection seam for multimodal content blocks (no transport change).
  [verified] Codex CLI already accepts a dict inference_input with a "prompt"
    key (codex_cli_inferencer.py:174) — natural carrier for an "attachments" key.
  [verified] AgentAttachment already exists (agents/agent_attachment.py:33) — so
    the plan's choice of name InferencerAttachment avoids a real collision.
  [verified] The shared module common/inferencers/attachments.py does NOT yet
    exist — Step 1 (create it) is genuinely new work, not a duplicate.

The body below is the task tool's own output, unmodified.
=========================================================================
-->

# Unified Plan: Image & File Attachment Support Across All 8 Inferencers

> **Scope.** Consolidate the three upstream subtask plans into one coherent, executable plan that adds image and file attachment support to:
> - **2 Claude Code external inferencers** (CLI + SDK) under `src/.../agentic_inferencers/external/claude_code/`
> - **2 Codex external inferencers** (CLI + SDK) under `src/.../agentic_inferencers/external/codex/`
> - **3 AG (AI‑Gateway) API inferencers** (Claude / Gemini / OpenAI) under `src/.../api_inferencers/ag/`
>
> Plus one new shared attachment type and helper module hoisted to `common/inferencers/` so all groups share a single attachment surface (per aggregator guidance).

---

## 0. Aggregation Provenance & Reconciliation Map

This artifact integrates three upstream worker outputs:

| Upstream | Path (final_deliverables/output.md) | Scope |
|---|---|---|
| Worker 0 | `worker_0/.../final_deliverables/output.md` | Claude Code CLI + SDK |
| Worker 1 | `worker_1/.../final_deliverables/output.md` | Codex CLI + SDK (already Round‑01 consolidated; spine of that branch) |
| Worker 2 | `worker_2/.../final_deliverables/output.md` | AG Claude + Gemini + OpenAI |

All three independently converged on the **same architecture**: a single neutral `Attachment` value type at the inferencer boundary plus per‑backend adapters that translate to the native wire form (CLI flag, SDK content‑part, or provider HTTP message block). They differ only on (a) **where** the shared type lives and (b) minor field naming.

### 0.1 Cross‑worker naming & location reconciliation (resolved)

| Concern | Worker 0 (Claude Code) | Worker 1 (Codex) | Worker 2 (AG) | **Final, unified decision** |
|---|---|---|---|---|
| Type name | `Attachment` | `InferencerAttachment` | `Attachment` + `AttachmentIntent` | **`InferencerAttachment`** (unambiguous vs. `AgentAttachment`); legacy alias `Attachment = InferencerAttachment` re‑exported from each sub‑package for ergonomics. |
| Type location | `external/claude_code/common.py` | `external/sdk_types.py` | `api_inferencers/ag/attachments.py` | **Hoist to `src/.../common/inferencers/attachments.py`** — single source of truth. Each of the three sub‑packages re‑exports it via `__init__.py`. |
| Kind/intent field | `kind: Literal["image","file","auto"]` | `kind: Literal[...]` | `intent: AttachmentIntent` enum (`IMAGE/DOCUMENT/FILE`) | **`kind: Literal["image","document","file","auto"]`** (string, IDE‑friendly, extensible without enum imports; `document` added so AG PDF case is first‑class). `intent` accepted as deprecated alias for one release. |
| Source carrier | `path \| url` | `path \| url` | `source: Union[str, bytes, Path]` | **All three: `path: str\|None`, `url: str\|None`, `data: bytes\|None`** (exactly one set). Bytes path is needed for AG (in‑memory PNG), not for CLI/SDK; centralizing keeps adapters uniform. |
| MIME hint | `mime_type` | `mime_type` | `mime_type` | **`mime_type: Optional[str]`** (consensus). |
| Logical name | n/a | `name` (for MentionInput) | `name` (for OpenAI filename / Anthropic title) | **`name: Optional[str]`**. |
| Role targeting | n/a | n/a | `role: Literal["user","system"]` | **`role: Literal["user","system"] = "user"`** — only AG honors it today; CLI/SDK ignore (always implicit user turn). |
| MIME detection | extension only | extension + `mime_type` hint | extension + caller MIME + **magic‑byte sniff** (PNG/JPEG/GIF/WebP/PDF) | **Adopt the magic‑byte sniff from Worker 2** as the third fallback after caller‑MIME and `mimetypes.guess_type`. |
| Caller carriers | dict‑prompt key AND `attachments=` kwarg | dict‑prompt key AND `attachments=` kwarg, merged | `attachments=` kwarg + `set_attachments()` setter | **All three**: dict key, kwarg, and (AG only) setter. Merge order: dict‑key first, then kwarg appended, then setter (AG) one‑shot. |
| Unsupported‑kind handling | warn + skip | hard fail for SDK URL+file (`ValueError`); CLI `<attachments>` trailer for non‑image | `strict_unsupported_attachments` flag, default degrade to text stub `[Attached file: <name>]` | **Per backend**: Codex SDK `ValueError` for URL+file (no analog); CLI `<attachments>` trailer for non‑image; AG `strict_unsupported_attachments=False` default with text‑stub degrade. Documented per inferencer. |
| URL fetch policy | n/a | leave to backend | `prefer_base64_for_anthropic=True` default — fetch+encode for Bedrock route | **Keep `prefer_base64_for_anthropic=True`** for AG Claude only. |

### 0.2 Unique vs overlapping content (provenance)

- **Overlapping (all three)**: single neutral type, additive kwarg + dict‑key carriers, zero text‑only regression, lazy backend imports, per‑backend adapter table.
- **Unique to Worker 0 (Claude Code)**: native attachment research (Claude Code CLI `@path` mention syntax + SDK content blocks). See `worker_0/.../final_deliverables/output.md` §0.3.
- **Unique to Worker 1 (Codex)**: live verification of `codex exec --help` (`-i, --image <FILE>`, repeatable); live `inspect` of `AsyncThread.turn` showing `RunInput = list[Item] | Item | str` and the `TextInput / LocalImageInput / ImageInput / MentionInput / SkillInput` item types; **R7 defensive single‑text SDK downgrade** that preserves byte‑identical text‑only fast path even when `attachments=[]` is passed. See `worker_1/.../final_deliverables/output.md` §1.4–§1.5 and §0.1 row 5.
- **Unique to Worker 2 (AG)**: endpoint reality check — OpenAI route is **Chat Completions** (not Responses), Gemini route is **OpenAI‑compatible** (not native `:generateContent`); magic‑byte MIME sniff; `prefer_base64_for_anthropic` for the Bedrock route; `attachment_style="openai_compat"|"gemini_native"` opt‑in; hard size check (`AttachmentTooLargeError` at 20 MB raw). See `worker_2/.../final_deliverables/output.md` §1.2 and R3/R4/R5.

---

## 1. Evidence‑Based Investigation Summary (verified)

### 1.1 Inferencer surface area (all 8, repo paths)

| Group | File | LoC | Base class | Prompt entry |
|---|---|---|---|---|
| Claude Code | `external/claude_code/claude_code_cli_inferencer.py` | ~915 | `TerminalSessionTemplatedInferencerBase` | `construct_command()` L233‑318 — dict/`str` split |
| Claude Code | `external/claude_code/claude_code_sdk_inferencer.py` | ~497 | `StreamingInferencerBase, TemplatedInferencerBase` | `_extract_prompt(...)` |
| Claude Code | `external/claude_code/common.py` | ~112 | — | shared helpers |
| Codex | `external/codex/codex_cli_inferencer.py` | 603 | `TerminalSessionTemplatedInferencerBase` | `construct_command()` ~L166 — dict/`str` split |
| Codex | `external/codex/codex_sdk_inferencer.py` | 398 | `StreamingInferencerBase, TemplatedInferencerBase` | `_ainfer_streaming(prompt, **kwargs)` → `thread.turn(prompt)` ~L265 |
| AG | `api_inferencers/ag/ag_claude_api_inferencer.py` | 237 | `StreamingInferencerBase` | `_infer`/`_ainfer`/`_ainfer_streaming` + `set_messages()` |
| AG | `api_inferencers/ag/ag_gemini_api_inferencer.py` | 166 | `StreamingInferencerBase` | same |
| AG | `api_inferencers/ag/ag_openai_api_inferencer.py` | 189 | `StreamingInferencerBase` | same |

### 1.2 Native attachment mechanisms per backend (verified)

| Backend | Verified mechanism | Source |
|---|---|---|
| Claude Code CLI | `@path/to/file` mentions inside prompt text; SDK `content` blocks for images | worker_0 §0.3 |
| Claude Code SDK | Multimodal content blocks via SDK message API | worker_0 §0.3 |
| Codex CLI | `codex exec -i, --image <FILE>` (repeatable, variadic, FILE‑only — not URL) | worker_1 §1.4, live `codex exec --help` |
| Codex SDK | `AsyncThread.turn(input: RunInput)`; `RunInput = list[Item] \| Item \| str`; items: `TextInput`, `LocalImageInput(path)`, `ImageInput(url)`, `MentionInput(name,path)`, `SkillInput(...)` | worker_1 §1.5, live `inspect` |
| AG Claude | Anthropic native blocks at `/v1/messages`: `{type:"image", source:{type:"base64"\|"url", media_type, data\|url}}` and `{type:"document", source, title}` | worker_2 §1.2 |
| AG Gemini | **OpenAI‑compatible** Chat Completions at `/v1/google/.../chat/completions`: `{type:"image_url", image_url:{url}}` — native `inline_data`/`file_data` NOT routed today | worker_2 §1.2, R2 |
| AG OpenAI | **Chat Completions** at `/v1/openai/v1/chat/completions` (NOT Responses): `{type:"image_url", image_url:{url}}` and `{type:"file", file:{file_data\|file_url, filename}}` | worker_2 §1.2, R1 |

### 1.3 Constraints carried forward from each upstream

- **Module load without backend installed** (Codex SDK soft‑dep) — keep all `openai_codex` symbol imports lazy inside `_build_run_input` (worker_1 §2.1).
- **Tier‑3 isolation** — attachments must flow through call‑local `inference_input`/`kwargs`, never instance state (worker_1 R11).
- **Zero text‑only regression** — first‑class acceptance criterion in all three plans (worker_0 §5, worker_1 §5.4, worker_2 §5.2 case a).
- **AG transport untouched** — `ai_gateway_*_llm.py` already accept list‑of‑dicts on `prompt_or_messages`; only the three inferencers + new shared module are edited (worker_2 §1.3).

---

## 2. Unified Architecture

### 2.1 The single shared type (hoisted per aggregator guidance)

**Location.** `src/agent_foundation/common/inferencers/attachments.py` — a *new* module hoisted out of `claude_code/common.py`, `external/sdk_types.py`, and `ag/attachments.py`. This is the resolution to the aggregator guidance "if multiple subtasks introduced their own Attachment type, hoist a single shared definition into a common module."

```python
# Shape only (not source). See §2.2 for full contract.
@attrs.frozen
class InferencerAttachment:
    path: Optional[str] = None       # absolute or cwd-relative local path
    url:  Optional[str] = None       # http(s) URL (currently best for images)
    data: Optional[bytes] = None     # raw in-memory bytes (AG only today)
    kind: Literal["image","document","file","auto"] = "auto"
    name: Optional[str] = None       # logical filename
    mime_type: Optional[str] = None  # caller hint; else inferred
    role: Literal["user","system"] = "user"  # honored by AG only
```

Validator: exactly one of `{path, url, data}` must be set. `__attrs_post_init__` raises `ValueError` otherwise.

**Module helpers** (same file):
- `normalize_attachments(value, cwd=None) -> list[InferencerAttachment]` — accepts `None | str | os.PathLike | InferencerAttachment | Mapping | Iterable[...]`. Bare path string → `InferencerAttachment(path=<str>, kind="auto")`. Does **not** auto‑resolve relative paths to absolute; emits `WARNING` instead (worker_1 R9 — auto‑resolving risks wrong‑dir bugs across CLI `--cd` and SDK `cwd=`).
- `_resolve_kind(att) -> Literal["image","document","file"]` — uses `att.kind` if explicit; else infers from extension or `mime_type`. Image exts: `{".png",".jpg",".jpeg",".gif",".webp",".bmp"}`. Document exts: `{".pdf"}`. Else `file`.
- `infer_mime(att) -> str` — precedence: caller `mime_type` → `mimetypes.guess_type` → magic‑byte sniff for `data` (PNG `\x89PNG`, JPEG `\xff\xd8\xff`, GIF `GIF8`, WebP `RIFF…WEBP`, PDF `%PDF`) → `application/octet-stream` for documents; **raises `ValueError` for `image` intent without resolvable MIME** (vendor APIs reject unknown image media types).
- `to_base64(att) -> str` — reads `path` or returns `data` encoded; raises on URL source.
- `is_url(att) -> bool`.
- `AttachmentTooLargeError(Exception)` — raised when raw bytes exceed `MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024` (~27 MB base64). Warning logged over 4 MB (worker_2 R4).
- Legacy alias: `Attachment = InferencerAttachment` re‑exported at sub‑package level (worker_0 used `Attachment`; back‑compat).

### 2.2 Sub‑package re‑exports

Each of the three sub‑package `__init__.py` files imports from the shared module:

```python
# src/.../external/claude_code/__init__.py
# src/.../external/codex/__init__.py
# src/.../api_inferencers/ag/__init__.py
from agent_foundation.common.inferencers.attachments import (
    InferencerAttachment,
    Attachment,           # alias
    normalize_attachments,
    AttachmentTooLargeError,
)
__all__ = [..., "InferencerAttachment", "Attachment", "normalize_attachments",
           "AttachmentTooLargeError"]
```

The AG sub‑package additionally exposes its provider serializers and the `_augment_messages_with_attachments` helper (see §2.4).

### 2.3 Per‑backend adapter table (canonical)

| Source kind | Claude Code CLI | Claude Code SDK | Codex CLI | Codex SDK | AG Claude | AG OpenAI / Gemini |
|---|---|---|---|---|---|---|
| Local image (`path`) | `@<abs_path>` mention in prompt (CLI native) | `image` content block from file bytes | `-i "<abs_path>"` (one flag per image; before final prompt positional) | `LocalImageInput(path=<abs_path>)` item in `RunInput` | `{type:"image",source:{type:"base64",media_type,data:<b64>}}` block | `{type:"image_url",image_url:{url:"data:<mime>;base64,<b64>"}}` part |
| Image URL | `<attachments>` trailer (no CLI flag for URLs) | `image` block with URL source if SDK supports; else fetch+encode | `<attachments>` trailer (CLI `-i` is FILE‑only) + warn | `ImageInput(url=<url>)` item | `{type:"image",source:{type:"url",url}}` (fetch+encode if `prefer_base64_for_anthropic=True`) | `{type:"image_url",image_url:{url:<u>}}` |
| Local file/doc (`path`) | `@<abs_path>` mention | content block w/ file ref where supported, else `@mention` | `<attachments>` trailer with `- file: <abs_path>` (Codex reads via tools) | `MentionInput(name=<name or basename>, path=<abs_path>)` | `{type:"document",source:{type:"base64",media_type:"application/pdf",data:<b64>},title}` for PDF; else `<attachments>` trailer or error | `{type:"file",file:{file_data:"data:...;base64,<b64>",filename}}` |
| Remote URL + file kind | `<attachments>` trailer | as above | `<attachments>` trailer | **`ValueError`** (no Codex SDK analog) | URL source for `document` block | `{type:"file",file:{file_url:<u>,filename}}` |
| In‑memory `data: bytes` | unsupported (write to temp file or error) | content block from bytes if SDK supports | unsupported (write temp file or `ValueError`) | unsupported (`ValueError` — SDK requires path/url) | base64 block | base64 data URL |

The CLI `<attachments>` trailer block is an isolated, documented graceful degrade; the helper that emits it is swappable for a native flag when Codex / Claude Code ship one. Format:

```
<attachments>
- file: /abs/path/to/spec.md
- image: https://example.com/diagram.png
</attachments>
```

### 2.4 AG: shared message‑rewrite helper + per‑provider serializers

Lives in `api_inferencers/ag/attachment_serializers.py` (kept AG‑local because the wire shapes are provider‑specific):

- `to_anthropic_blocks(atts) -> list[dict]` — produces Anthropic `image` / `document` blocks per §1.2.
- `to_openai_parts(atts) -> list[dict]` — Chat Completions `image_url` / `file` parts per §1.2.
- `to_gemini_parts(atts, style="openai_compat") -> list[dict]` — default mirrors OpenAI parts (because AG Gemini routes through `/chat/completions`); `style="gemini_native"` emits `inline_data` / `file_data` parts for future `:generateContent` routing — unit‑tested but not the default (worker_2 R2).
- `_augment_messages_with_attachments(prompt_or_messages, atts, serializer, text_wrapper) -> list[dict]` — normalises bare string to a messages list; locates the target message by `role` (default last `user`); promotes its `content` to a parts list (prepending a text block); appends serialized attachment parts. Wrapped in `try/finally` to clear `_attachments_override` (worker_2 R7). Precedence: explicit messages (`_messages_override` or list prompt) win and attachments append to the last user message of that list rather than constructing a new turn (worker_2 R6).

### 2.5 Per‑call input contract (additive, back‑compat)

All inferencers accept attachments through one of three equivalent channels (merged in this order):

1. **Dict prompt**: `inference_input = {"prompt": "...", "attachments": [...]}` (CLI / SDK / AG).
2. **Per‑call kwarg**: `inferencer(prompt, attachments=[...])` (all 8).
3. **AG setter**: `ag_inferencer.set_attachments([...])` — one‑shot, cleared in `try/finally`, mirrors existing `set_messages()`.

Pure string input remains byte‑identical to today.

---

## 3. Unified Implementation Plan (Dependency‑Ordered)

> Steps are numbered globally so the executor can run them in order across all three groups. Per‑step acceptance criteria are explicit so reviewers can verify in isolation. Cross‑references point to upstream sections for full detail.

### Step 1 — Create shared module `common/inferencers/attachments.py`
**File (new):** `src/agent_foundation/common/inferencers/attachments.py`
- Implement `InferencerAttachment` (per §2.1), `normalize_attachments`, `_resolve_kind`, `infer_mime` (with magic‑byte sniff), `to_base64`, `is_url`, `AttachmentTooLargeError`.
- Module docstring documents the contract for all 8 inferencers + the three carrier channels (§2.5).
- Legacy alias `Attachment = InferencerAttachment` defined here.

**Acceptance**
- `normalize_attachments(None)` → `[]`.
- `normalize_attachments("foo.png")` → `[InferencerAttachment(path=<str>, kind="auto")]`, `_resolve_kind` → `"image"`.
- `infer_mime` for PNG bytes (`\x89PNG\r\n\x1a\n…`) returns `image/png` even without a path.
- `InferencerAttachment(path=None, url=None, data=None)` raises `ValueError`.
- `InferencerAttachment(path="x", url="y")` raises `ValueError`.
- File > 20 MB raises `AttachmentTooLargeError` from `to_base64`.

### Step 2 — Sub‑package `__init__.py` re‑exports
**Files (modified):**
- `src/.../external/claude_code/__init__.py`
- `src/.../external/codex/__init__.py`
- `src/.../api_inferencers/ag/__init__.py` (currently empty)

Each adds the imports from §2.2 and extends `__all__`.

**Acceptance**
- `from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import InferencerAttachment` resolves.
- Same for `codex` and `ag` sub‑packages.

### Step 3 — Wire `CodexCliInferencer` (worker_1 §3 Step 3)
**File (modified):** `src/.../external/codex/codex_cli_inferencer.py`
- Extract attachments at dict + kwarg boundary in `construct_command()`.
- Argv assembly: append `-i "<abs_path>"` per image, **immediately before** the trailing prompt positional / `parts.append("-")` block (worker_1 R5).
- For URL‑only image OR `kind=file` attachments: build `_augment_prompt_with_attachments(prompt, fallback_atts)` trailer — appears exactly once.
- Resume subcommand: pass `-i` on `exec resume` as well; failures surface via existing stderr handling (worker_1 R1).
- Debug‑log count/kinds; never log absolute paths at info level.

**Acceptance** — see worker_1 §3 Step 3 acceptance list + test matrix rows 1–7.

### Step 4 — Wire `CodexSdkInferencer` (worker_1 §3 Step 4)
**File (modified):** `src/.../external/codex/codex_sdk_inferencer.py`
- Add `_extract_attachments(inference_input, kwargs)` helper.
- Add `_build_run_input(prompt, attachments) -> Union[str, list]` with **lazy** SDK imports (`TextInput`, `LocalImageInput`, `ImageInput`, `MentionInput`) inside the helper.
- **R7 defensive single‑text downgrade**: if `attachments=[]` is passed explicitly, return bare `str` so `thread.turn(<str>)` fast path is exercised exactly (preserves byte‑identical text‑only behavior).
- URL + file kind → synchronous `ValueError` before any SDK call.
- No changes to `aconnect`, notification loop, tool‑use counters, idle‑timer, Tier‑3 fan‑out.

**Acceptance** — see worker_1 §3 Step 4 acceptance list + matrix rows 8–14.

### Step 5 — Wire `ClaudeCodeCliInferencer` (worker_0 §2.2)
**File (modified):** `src/.../external/claude_code/claude_code_cli_inferencer.py`
- In `construct_command()` (L233‑318) replace dict‑extraction block to also pop `attachments` (dict key + kwarg merged).
- For each attachment: append `@<abs_path>` mention to prompt (CLI native attachment syntax), preserving original prompt text first; for URLs use `<attachments>` trailer.
- Update `_extract_prompt(...)` to keep templated string rendering intact.

**Acceptance**
- String prompt without attachments → byte‑identical command vs. baseline.
- Single image path → prompt contains `@<abs_path>` exactly once at the position chosen by the trailer convention.
- Two image paths → two distinct mentions in input order.

### Step 6 — Wire `ClaudeCodeSdkInferencer` (worker_0 §2.3)
**File (modified):** `src/.../external/claude_code/claude_code_sdk_inferencer.py`
- Continue calling `self._extract_prompt(...)` for templated rendering.
- Translate `InferencerAttachment` list to SDK content blocks (image / document) where the SDK supports them; fall back to `@mention` prompt augmentation for unsupported kinds.
- Preserve streaming/agentic semantics, session capture, idle‑timer.

**Acceptance** — see worker_0 §2.3 and §5.1 layered verification.

### Step 7 — Update `claude_code/common.py` (worker_0 §2.1)
**File (modified):** `src/.../external/claude_code/common.py`
- Replace any locally proposed `Attachment` class with re‑export from the hoisted shared module.
- Add shared helpers needed by both CLI + SDK paths: `_render_attachment_mention(att) -> str`, `_build_attachments_trailer(atts) -> str`.

### Step 8 — Create AG shared modules
**Files (new):**
- `src/.../api_inferencers/ag/attachments.py` — thin re‑export of the shared module + AG‑specific `prefer_base64_for_anthropic`, `attachment_style`, `strict_unsupported_attachments` flag types (`Literal[...]` aliases). **No duplicate `InferencerAttachment` definition.**
- `src/.../api_inferencers/ag/attachment_serializers.py` — `to_anthropic_blocks`, `to_openai_parts`, `to_gemini_parts(..., style=...)`, `_augment_messages_with_attachments` per §2.4.

**Acceptance** — snapshot unit tests assert exact dict shape per §2.3 for each provider × source × kind cell.

### Step 9 — Wire 3 AG inferencers (worker_2 §3 Phase 3)
**Files (modified):**
- `src/.../api_inferencers/ag/ag_claude_api_inferencer.py`
- `src/.../api_inferencers/ag/ag_openai_api_inferencer.py`
- `src/.../api_inferencers/ag/ag_gemini_api_inferencer.py`

Each adds:
- `_attachments_override: Optional[list[InferencerAttachment]] = attrib(default=None, init=False)`.
- `set_attachments(attachments)` setter (normalises single → list).
- `attachments` capture in `_apply_defaults` (consumes kwarg, stores override, strips from forwarded kwargs).
- In `_infer` / `_ainfer` / `_ainfer_streaming`, just before calling the AG gateway: build messages via `_augment_messages_with_attachments(...)` with the provider‑specific serializer; pass `prompt_or_messages=<list[dict]>`; clear `_attachments_override` in `try/finally`.

Provider specifics:
- **AG Claude**: `prefer_base64_for_anthropic: bool = True` instance attribute; fetches and base64‑encodes URL images when set (Bedrock route compatibility — worker_2 R3).
- **AG Gemini**: `attachment_style: Literal["openai_compat","gemini_native"] = "openai_compat"` (worker_2 R2).
- **AG OpenAI**: documents Chat Completions endpoint reality in module docstring (worker_2 R1).

**Acceptance** — see worker_2 §3 Phase 3 acceptance + §5.2 inferencer test cases (a)–(i).

### Step 10 — Examples + brief docstring/README usage notes (per aggregator guidance #3)
**Files (new / modified):**
- `examples/.../external/codex/example_codex_cli_streaming.py` — opt‑in commented attachments demo.
- `examples/.../external/codex/example_codex_sdk_streaming.py` — same.
- `examples/.../external/claude_code/example_claude_code_*.py` — same.
- `examples/agent_foundation/common/inferencers/api_inferencers/ag/example_ag_attachments.py` — one snippet per AG inferencer attaching a PNG.
- `src/agent_foundation/common/inferencers/attachments.py` — module docstring is the unified usage note: shows the **one** canonical way to pass an attachment to any of the 8 inferencers using `InferencerAttachment(path=..., kind=...)` and the three carrier channels in §2.5.
- `src/agent_foundation/apis/ag/README.md` — append "Attachments" subsection (~30 lines) per provider.

### Step 11 — Tests
See §5 below. Tests are part of each preceding step's acceptance and should be committed alongside the code they verify.

---

## 4. Files to Create / Modify (consolidated, all 3 groups)

### 4.1 Create
| Path | Purpose |
|---|---|
| `src/agent_foundation/common/inferencers/attachments.py` | **Single shared `InferencerAttachment` + helpers** (hoisted; per aggregator guidance). |
| `src/.../api_inferencers/ag/attachment_serializers.py` | AG‑local provider serializers + `_augment_messages_with_attachments`. |
| `src/.../api_inferencers/ag/attachments.py` | Thin AG‑side facade (re‑exports + AG flag aliases). |
| `examples/.../api_inferencers/ag/example_ag_attachments.py` | One snippet per AG inferencer. |
| `test/.../common/inferencers/test_attachments.py` | `InferencerAttachment`, `normalize_attachments`, MIME helper, magic‑byte sniff, size cap. |
| `test/.../external/claude_code/test_claude_code_attachments.py` | CLI mention/trailer + SDK content‑block cases. |
| `test/.../external/codex/test_codex_cli_attachments.py` | argv matrix rows 1–7 + R5 snapshot + hostile path escape. |
| `test/.../external/codex/test_codex_sdk_attachments.py` | SDK item‑type assertions + R7 downgrade + lazy‑import isolation. |
| `test/.../api_inferencers/ag/test_attachment_serializers.py` | Snapshot dict shapes per §2.3. |
| `test/.../api_inferencers/ag/test_ag_claude_attachments.py` | AG Claude with stubbed gateway (cases a–i). |
| `test/.../api_inferencers/ag/test_ag_openai_attachments.py` | Same for OpenAI. |
| `test/.../api_inferencers/ag/test_ag_gemini_attachments.py` | Same for Gemini. |
| `test/.../api_inferencers/ag/integration/test_ag_attachments_live.py` | Env‑gated live smoke (`RUN_AG_LIVE_TESTS=1`). |

### 4.2 Modify
| Path | Change |
|---|---|
| `src/.../external/claude_code/common.py` | Re‑export `InferencerAttachment`; add `_render_attachment_mention`, `_build_attachments_trailer`. |
| `src/.../external/claude_code/claude_code_cli_inferencer.py` | Dict + kwarg extraction; `@mention` emission; `<attachments>` trailer fallback. |
| `src/.../external/claude_code/claude_code_sdk_inferencer.py` | Translate to SDK content blocks; fallback to mentions. |
| `src/.../external/claude_code/__init__.py` | Re‑export shared types. |
| `src/.../external/codex/codex_cli_inferencer.py` | `-i` flag emission; `<attachments>` trailer; resume‑aware. |
| `src/.../external/codex/codex_sdk_inferencer.py` | `_build_run_input` with lazy SDK imports; R7 downgrade. |
| `src/.../external/codex/__init__.py` | Re‑export shared types. |
| `src/.../external/sdk_types.py` | **No new type**; if Worker 1's plan had placed `InferencerAttachment` here, it now imports from the hoisted shared module instead (single source). |
| `src/.../api_inferencers/ag/__init__.py` | Re‑export `InferencerAttachment`, `Attachment`, serializers. |
| `src/.../api_inferencers/ag/ag_claude_api_inferencer.py` | `_attachments_override`, `set_attachments`, `prefer_base64_for_anthropic`, merge in `_infer`/`_ainfer`/`_ainfer_streaming`. |
| `src/.../api_inferencers/ag/ag_openai_api_inferencer.py` | Same scaffolding + Chat Completions docstring. |
| `src/.../api_inferencers/ag/ag_gemini_api_inferencer.py` | Same scaffolding + `attachment_style`. |
| `src/agent_foundation/apis/ag/README.md` | "Attachments" subsection per provider. |
| `src/agent_foundation/agents/agent_attachment.py` | Docstring note pointing to multimodal `InferencerAttachment` to prevent confusion. **No behavioural change.** |

### 4.3 Explicitly NOT modified (reviewed, no edits)
- `streaming_inferencer_base.py`, `terminal_session_inferencer_base.py` — kwargs flow through transparently (worker_0 §2.6, worker_1 §4).
- `src/agent_foundation/apis/ag/ai_gateway_claude_llm.py`, `ai_gateway_openai_llm.py`, `ai_gateway_gemini_llm.py` — all already accept list‑of‑dicts on `prompt_or_messages` (worker_2 §1.3).

---

## 5. Validation & Testing Strategy

### 5.1 Static
- `mypy` / `pyright` (project‑standard) over the new shared module, the three modified inferencer groups, and the AG serializers.
- `grep` confirms no `Attachment` class definition remains outside `common/inferencers/attachments.py` (single source enforced).

### 5.2 Unit test matrix (consolidated across all 3 groups)

**Shared module** (one test row per behavior):
| # | Case | Expected |
|---|---|---|
| S1 | `normalize_attachments(None)` | `[]` |
| S2 | `normalize_attachments("foo.png")` | one `InferencerAttachment` with kind→`image` |
| S3 | `normalize_attachments([dict, InferencerAttachment, str])` round‑trip | mixed list preserved |
| S4 | `InferencerAttachment(path=None,url=None,data=None)` | `ValueError` |
| S5 | `InferencerAttachment(path="x",url="y")` | `ValueError` |
| S6 | `infer_mime` magic‑byte sniff on PNG/JPEG/GIF/WebP/PDF | correct MIME for each |
| S7 | `to_base64` on > 20 MB file | `AttachmentTooLargeError` |

**Codex CLI** (worker_1 §5.2 rows 1–7): no attachments / 1 image kwarg / 2 image dict / image + stdin / image + resume / URL‑only / kind=file mixed.

**Codex SDK** (worker_1 §5.2 rows 8–13): no attachments / 1 local image / 1 URL image / 1 local file / URL+file `ValueError` / explicit `attachments=[]` R7 downgrade.

**Claude Code CLI/SDK** (worker_0 §5.1): byte‑identical baseline / 1 image / 1 file / mixed / streaming preserved / session capture preserved.

**AG (per provider)** (worker_2 §5.2 cases a–i): text only / text+image path / text+image URL / text+PDF / caller MIME wins / kwarg via `__call__` / override cleared after call / streaming variant / mid‑call exception cleanup.

**AG serializers** snapshot dict shape per §2.3 cell.

### 5.3 Cross‑cutting tests
- **Snapshot parity:** `construct_command("hello")` for both CLI inferencers, and `prompt_or_messages` for AG inferencers, byte‑identical to pre‑change baseline captured on a known commit.
- **Hostile path escaping:** `/tmp/a b/x.png`, `path with"quote".png` round‑trip through `_escape_for_shell` + `shlex.split` (worker_1 R2).
- **Lazy‑import isolation:** with `openai_codex` patched to fail, Codex SDK inferencer module still imports (worker_1 R8).
- **Cross‑inferencer parity (NEW — verifying aggregator guidance #1):** dispatch the same normalized `InferencerAttachment` list to all 8 inferencers via a parametrized harness; assert per §2.3 adapter table that each produces the expected backend payload shape. Confirms the unified API is truly consistent.
- **Session capture parity:** `turn/started` thread‑id capture still updates `active_session_id` for attachment‑bearing turns (worker_1 §5.2 cross‑cutting).

### 5.4 Integration smoke (env‑gated)
- **Codex** (`RUN_CODEX_E2E=1` + `codex login status`): tiny PNG + "Describe this image." → streamed `agent_message` references image; `thread_id` captured.
- **Claude Code** (analogous gating): same.
- **AG** (`RUN_AG_LIVE_TESTS=1` or `AG_INTEGRATION=1`): per provider, tiny 8×8 (or 1×1) PNG + "What color is this pixel?" → HTTP 200, non‑empty text; for PDF‑capable providers, 1‑page PDF. Write outcomes to `test/_artifacts/ag_attachments_smoke_<ts>.json` (never stdout — per project convention).

### 5.5 Regression guards
- Re‑run the **entire existing inferencer test suite** — text‑only flows remain green with no expected‑output changes. This is the smoke step explicitly called out in aggregator guidance #4.
- Lint new public types per project `attrs`/typing conventions (match `SDKInferencerResponse` decorator usage exactly).
- Re‑run any existing AG / Claude Code / Codex examples in text‑only mode; outputs and timings equivalent (worker_2 §5.4).

### 5.6 Self‑validation checklist (executor must tick before merge)
- [ ] Single `InferencerAttachment` definition exists at `common/inferencers/attachments.py`; `grep -R "class InferencerAttachment\|class Attachment\b" src/` returns exactly that file.
- [ ] All 3 sub‑packages re‑export it (Step 2).
- [ ] All 8 inferencers accept `attachments=` kwarg (Steps 3–9).
- [ ] Text‑only baseline byte‑identical for all 8 (snapshot tests in §5.3).
- [ ] R7 SDK single‑text downgrade verified (Codex SDK test).
- [ ] Magic‑byte MIME sniff verified on at least PNG/JPEG/GIF/WebP/PDF (S6).
- [ ] AG mid‑call exception still clears `_attachments_override` (case i).
- [ ] No edits to `streaming_inferencer_base.py`, `terminal_session_inferencer_base.py`, or any `ai_gateway_*_llm.py` (verified by `git diff`).
- [ ] Cross‑inferencer parity harness passes (§5.3).
- [ ] README + attachments module docstring document the **one** unified usage (per aggregator guidance #3).

---

## 6. Risk Register (consolidated across all 3 groups)

| # | Sev | Risk | Failure mode | Mitigation | Source |
|---|---|---|---|---|---|
| R1 | 🔴 High | AG OpenAI route is Chat Completions, not Responses API | Emitting `input_image`/`input_file` parts would 400 | Implement Chat Completions `image_url`/`file` parts; document in docstring; smoke‑test (§5.4) | worker_2 R1 |
| R2 | 🔴 High | AG Gemini route is OpenAI‑compatible | Native `inline_data`/`file_data` would be rejected | Default `to_gemini_parts(style="openai_compat")`; native opt‑in is unit‑tested only | worker_2 R2 |
| R3 | 🔴 High | Codex SDK `RunInput` signature mismatch in some SDK builds | Passing list where single item expected fails the call | Verified live: `RunInput = list \| Item \| str`; **R7 defensive downgrade** in `_build_run_input` | worker_1 R7 |
| R4 | 🟡 Med | Codex CLI rejects `-i` on `exec resume` in some versions | Non‑zero exit; attachments silently ignored if swallowed | Pass `-i` on resume; surface stderr verbatim; targeted unit‑test fixture | worker_1 R1 |
| R5 | 🟡 Med | Path quoting on Windows / paths with spaces, quotes, `$`, backticks | argv mangled; injection in test harnesses | Reuse existing `_escape_for_shell`; explicit hostile‑path tests | worker_1 R2 |
| R6 | 🟡 Med | No native CLI flag for non‑image files (Codex CLI) | `kind=file` callers expect attachment; CLI has no `--file` | Documented limitation; `<attachments>` trailer fallback delivers absolute path so Codex/Claude Code can read via their tools | worker_1 R3 |
| R7 | 🟡 Med | `openai-codex` SDK API drift (`LocalImageInput` rename, etc.) | Import‑time or call‑time `AttributeError` | Lazy import keeps text‑only paths working; one focused unit test pins each item type by name; pin minimum SDK version in `pyproject.toml` extras | worker_1 R4 |
| R8 | 🟡 Med | Streaming regression — `-i` insertion moves arg order | Misplaced `-i` would treat next token as stdin sentinel | Insertion strictly BEFORE the `parts.append("-")` block; snapshot test (matrix row 5) | worker_1 R5 |
| R9 | 🟡 Med | Some Claude / Bedrock routes do not accept `image` URL `source` variant | 400 from Bedrock on URL‑source images | `prefer_base64_for_anthropic=True` default: fetch + base64‑encode URL images before serialization | worker_2 R3 |
| R10 | 🟡 Med | Large file base64 inflates payload > 1 MB → 413/timeout | Timeout / 413 | Hard size check (20 MB raw → ~27 MB base64); `AttachmentTooLargeError`; warning over 4 MB | worker_2 R4 |
| R11 | 🟡 Med | MIME guessing wrong for raw `bytes` without filename | Provider rejects "unsupported media type" or uses `application/octet-stream` | Magic‑byte sniff (PNG/JPEG/GIF/WebP/PDF); raise clear `ValueError` for IMAGE intent if unresolvable | worker_2 R5 |
| R12 | 🟡 Med | `set_messages` / `attachments=` precedence ambiguity (AG) | Caller using both could double‑set | Defined precedence: explicit messages win; attachments append to last user message of that list; unit test guards | worker_2 R6 |
| R13 | 🟡 Med | Relative vs absolute paths drift across CLI `--cd` and SDK `cwd=` | Wrong‑directory resolution | `normalize_attachments` does NOT auto‑resolve; emits WARNING for relative paths without explicit `cwd=` | worker_1 R9 |
| R14 | 🟡 Med | Remote URL image not fetched by Codex CLI | URL silently treated as a file path | Validate URL vs path in `normalize_attachments`; URL+image+CLI falls back to `<attachments>` trailer with warn | worker_1 R14 |
| R15 | 🟡 Med | Prompt‑embedded fallback semantics drift over time | Future model ignores `<attachments>` block | Unambiguous tag block; helper isolated and swappable for native flag when one ships | worker_1 R10 |
| R16 | 🟢 Low | Cache‑key drift for cached resume calls | Identical text prompts with new images erroneously hit cache | Documented in `InferencerAttachment` docstring; future cache‑key extension out of scope | worker_1 R6 |
| R17 | 🟢 Low | Soft‑dep import regression (Codex SDK) | Importing `openai_codex` at module load breaks env without SDK | Lazy import inside `_build_run_input` (existing pattern preserved) | worker_1 R8 |
| R18 | 🟢 Low | Tier‑3 isolation under fan‑out leaks attachments across branches | Wrong attachments delivered on a parallel branch | Attachments flow through call‑local input/kwargs only — never instance state | worker_1 R11 |
| R19 | 🟢 Low | `LargeInputMode.STDIN` interaction with augmented prompt | Stdin contains augmented prompt + argv `-` (correct, but easy to regress) | Snapshot test covers stdin payload + argv simultaneously (matrix row 4) | worker_1 R12 |
| R20 | 🟢 Low | Breaking the existing `inference_input` dict contract | Downstream callers passing unknown keys raise errors | Tolerant `inference_input.get("attachments")`; regression test asserts today's `{"prompt": "..."}` byte‑identical | worker_1 R13 |
| R21 | 🟢 Low | `_attachments_override` not cleared on exception (AG) | Stale attachments leak to next call | `try/finally` clears slot regardless of success (mirrors `_messages_override`) | worker_2 R7 |
| R22 | 🟢 Low | Naming collision with existing `AgentAttachment` | IDE confusion / import mix‑ups | `InferencerAttachment` is unambiguous; "See also" docstring notes in both `agents/agent_attachment.py` and the new shared module | worker_2 R8 |
| R23 | 🟢 Low | Tests requiring network / SLAuth fail in CI | Failed CI runs | All new unit tests stub gateway client; live tests gated by `RUN_AG_LIVE_TESTS=1` / `RUN_CODEX_E2E=1` | worker_2 R9 |
| R24 | 🟢 Low | Streaming + attachments timing surprises | UX surprise on first token | No change to streaming protocol — only request body changes; stub test asserts | worker_2 R10 |
| R25 | 🟢 Low | Cross‑inferencer parity drift (one inferencer lags behind shared API) | Two inferencers expose different attachment APIs | Single shared module (§2.1); cross‑inferencer parity harness in §5.3; CI snapshots catch drift | worker_1 R15 + worker_2 R11 |
| R26 | 🟢 Low | New public types not re‑exported, breaking discoverability | `from ... import InferencerAttachment` fails | Step 2 explicitly updates all 3 `__init__.py`; tests import via sub‑package paths | worker_1 R16 |

---

## 7. Open Questions & Future Work (non‑blocking)
1. **Native Gemini route.** If AG ever exposes `:generateContent`, swap default `attachment_style` to `"gemini_native"` — serializer already implemented and unit‑tested.
2. **Adapter from `AgentAttachment` → `InferencerAttachment`.** Would let legacy XML‑attachment call sites opt into multimodal automatically. Out of scope; intentional duplication noted.
3. **Larger‑than‑5 MB inputs.** Consider a helper that pre‑uploads via AG's signed‑URL service (if/when exposed) and returns an `InferencerAttachment(url=<https-url>)`. Current plan accommodates transparently because all serializers prefer URL form when source is URL.
4. **OpenAI `file` vs `input_file` reconciliation.** §5.4 smoke determines whether the AG OpenAI tenant accepts Chat Completions `file` part. If it requires the Responses‑API `input_file` shape, file (non‑image) attachments deferred to a separate ticket.
5. **Cache‑key extension** to include attachment hashes (worker_1 R6 — out of scope here).

---

## 8. Coverage Verification (mental checklist per upstream)

| Upstream key topic | Represented in this artifact? |
|---|---|
| Worker 0 §0.3 Claude Code CLI `@path` mention + SDK content blocks | ✅ §1.2, §2.3, Steps 5–7 |
| Worker 0 §1.3 input contract / additive carriers | ✅ §2.5 |
| Worker 0 §2.5 test plan for Claude Code attachments | ✅ §5.2 Claude Code section |
| Worker 0 §7.4 winner pick / ranking | ✅ §9 below |
| Worker 1 §0.1 six reconciled decisions | ✅ §0.1 (folded into broader naming reconciliation) |
| Worker 1 §1.4–§1.5 live verification of Codex CLI + SDK | ✅ §1.2 |
| Worker 1 §2.3 per‑backend mapping | ✅ §2.3 |
| Worker 1 §3 Steps 1–6 dependency‑ordered implementation | ✅ Steps 3, 4 (+ all others) |
| Worker 1 §5.2 14‑row test matrix | ✅ §5.2 Codex sections |
| Worker 1 §6 R1–R16 risk register | ✅ §6 (folded into R3–R8, R13–R20, R25, R26) |
| Worker 2 §1.2 endpoint reality (Chat Completions / OpenAI‑compat Gemini) | ✅ §1.2, §6 R1/R2 |
| Worker 2 §2.1 split into `attachments.py` + `attachment_serializers.py` | ✅ §2.1, §2.4, Step 8 |
| Worker 2 §2.3 per‑provider serializer contracts | ✅ §2.4 (referenced) + §2.3 adapter table |
| Worker 2 §4 R1–R11 risk register | ✅ §6 (folded into R1, R2, R9, R10, R11, R12, R21, R22, R23, R24, R25) |
| Worker 2 §5 test strategy (unit + inferencer + smoke + regression) | ✅ §5.2/§5.4/§5.5 |
| Worker 2 §7 Open Questions | ✅ §7 |

All key topics from each upstream are represented.

---

## 9. Aggregation Judgment

### 9.1 Why the hoisted shared module
The aggregator guidance explicitly directed: *"if multiple subtasks introduced their own Attachment type, hoist a single shared definition into a common module (e.g., common/inferencers/) and refactor each group to import it."* All three workers did indeed each introduce a local definition (Worker 0: `claude_code/common.py`; Worker 1: `external/sdk_types.py`; Worker 2: new `ag/attachments.py`). Hoisting to `src/agent_foundation/common/inferencers/attachments.py` satisfies the guidance, gives a single source of truth, and lets the cross‑inferencer parity harness in §5.3 exist at all. The sub‑package re‑exports (§2.2) preserve the per‑sub‑package import ergonomics each worker designed for.

### 9.2 API consistency verification (aggregator guidance #1)
The public attachments API is now uniform across all 8 inferencers:
- **Parameter name**: `attachments` (kwarg, dict key, and `set_attachments` setter).
- **Element type**: `InferencerAttachment` (with `Attachment` alias for back‑compat with Worker 0's naming).
- **Source carriers**: exactly one of `path` / `url` / `data` (validator enforced).
- **Supported kinds**: `image`, `document`, `file`, `auto`.
- **MIME handling**: centralized in `infer_mime` (caller hint → `mimetypes` → magic‑byte sniff → provider‑specific fallback).

### 9.3 Notable design choices made by this aggregation
- **Naming**: `InferencerAttachment` over `Attachment` (worker_0) or split `Attachment`+`AttachmentIntent` enum (worker_2) — chosen for unambiguity vs. `AgentAttachment` and IDE‑friendly `Literal["image","document","file","auto"]` over enum imports.
- **Adding `data: bytes` as a third source carrier** — required for AG's in‑memory case (worker_2's `source: Union[str,bytes,Path]`) but expressed as a separate field for clarity; unsupported for Codex/Claude Code SDK paths today (would require temp‑file write — flagged in §2.3).
- **Document as a first‑class kind** — promoted from worker_2's `AttachmentIntent.DOCUMENT` because AG Claude has a distinct `document` block separate from `image`; worker_1's binary `image/file` split was insufficient for AG PDF support.

### 9.4 Forced‑pick judgment (the question worker_0 §7.4 raised)
If forced to pick exactly one upstream as the standalone winner for its own group:
- **Worker 0 (Claude Code)**: the artifact's own aggregator judgment is the winner (already a Round‑01 consolidation).
- **Worker 1 (Codex)**: Upstream 0 (already explicitly identified as "the spine"; carries six substantive reconciliations including R7 single‑text downgrade and the 14‑row matrix).
- **Worker 2 (AG)**: Upstream 2 / `flow_1` (worker_2's own §8 — strictly dominates; magic‑byte MIME sniff, serializer‑module split, `attachment_style` opt‑in, artifact‑JSON discipline).

The final consolidated artifact (this document) is essentially: Worker 1's spine for cross‑inferencer architecture conventions + Worker 0's Claude Code group details + Worker 2's AG depth and risk catalogue + the hoisting move that the aggregator guidance required.


