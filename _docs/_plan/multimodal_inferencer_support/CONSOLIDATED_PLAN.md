# Consolidated Plan: Multimodal (Image & File) Support for Inferencers

**Date:** 2026-06-26
**Scope:** `agent_foundation.common.inferencers` — 7 inferencer classes across 5 inferencer families

> **Provenance.** This plan consolidates three independent investigation flows:
>
> - **Flow 1** (`_docs/_plan/multimodal_inferencer_support/PLAN.md`) — strongest on API format specs, attrs model detail, convenience API patterns, and native CLI ideas that needed verification.
> - **Flow 0** (`_docs/_plan/inferencer_architecture/multimodal_external_and_ag_inferencers_plan.md`) — most thorough on security (`§7 Redaction and Cache Safety`, `§Risk Register`), streaming cache risk, cautious capability probing, and the "do not reuse `AgentAttachment`" warning.
> - **Codex-focused upstream plan** (`_docs/_plan/inferencer_architecture/codex/multimodal_attachment_support_external_and_ag_inferencers_plan.md`) — strongest on shared normalization/staging, preserving attachments across `StreamingInferencerBase._extract_prompt()` and template rendering, Codex CLI native `--image`, and task-tool regression notes.
> - **Flow 2 summary** (provided in the orchestration prompt; claimed `outputs/final_deliverables/plan.md`, which is not present in this workspace) — strongest on 7-class scope, CLI strategy ladder, Codex SDK typed dataclasses, explicit reconciliation table, env-gated E2E JSON output, and "not modified" guard list.
>
> Where flows disagree, this plan records the resolution and rationale.

---

## 1. Target Inferencers

| # | Inferencer | Source Path | Backend |
|---|---|---|---|
| 1 | `ClaudeCodeCliInferencer` | `src/.../claude_code/claude_code_cli_inferencer.py` | Claude Code CLI subprocess |
| 2 | `ClaudeCodeSdkInferencer` | `src/.../claude_code/claude_code_sdk_inferencer.py` | `claude_agent_sdk.ClaudeSDKClient.query()` |
| 3 | `CodexCliInferencer` | `src/.../codex/codex_cli_inferencer.py` | `codex exec` CLI subprocess |
| 4 | `CodexSdkInferencer` | `src/.../codex/codex_sdk_inferencer.py` | `openai_codex.AsyncCodex` thread turn |
| 5 | `AgClaudeApiInferencer` | `src/.../api_inferencers/ag/ag_claude_api_inferencer.py` | AI Gateway → Anthropic Messages API (Bedrock) |
| 6 | `AgOpenAIApiInferencer` | `src/.../api_inferencers/ag/ag_openai_api_inferencer.py` | AI Gateway → OpenAI Chat Completions |
| 7 | `AgGeminiApiInferencer` | `src/.../api_inferencers/ag/ag_gemini_api_inferencer.py` | AI Gateway → Gemini (OpenAI-compat) |

All 7 accept **text-only** prompts (`str` or `{"prompt": "..."}` dict). None have first-class multimodal support.

### Non-Goals (First Implementation)

- Do not redesign conversational inbox/server schemas (text-oriented: `inbox.py:21`, `schema.py:65`).
- Do not repurpose `AgentAttachment` (`agent_attachment.py:33`) — it is a prompt-rendering abstraction with `id`, `description`, `content`, `formatter`; no MIME, path, URL, byte payload, or base64 semantics. *(Flow 0 unique finding.)*
- Do not promise "any binary file works everywhere" — provider/gateway support differs by MIME type and route.
- Do not modify inferencers outside the 7 named targets.
- Do not inline arbitrary large binaries in prompts or logs.

---

## 2. Evidence-Based Findings

### 2.1 Current Input Handling (All Text-First)

**Claude Code CLI** (`claude_code_cli_inferencer.py`):
- Class at line 34; `construct_command()` at line 233; `_ainfer_streaming()` at line 450; `_infer()` at line 612.
- `large_input_mode: LargeInputMode = STDIN` (line 98) — avoids `ARG_MAX` by piping prompt via stdin.
- `has_local_access = True` (line 90). No `--image`/`--file` flag referenced anywhere.
- `extra_cli_args` escape hatch exists for forward-compat.

**Claude Code SDK** (`claude_code_sdk_inferencer.py`):
- Class at line 33; `_ainfer_streaming()` at line 265; calls `client.query(prompt)` at line 303; `_infer()` at line 363.
- `has_local_access = True` (line 151). Default `allowed_tools` includes `["Read", "Write", "Bash"]` (line 157).
- `claude-agent-sdk`'s `ClaudeSDKClient.query()` may accept either a string or a structured message list per Anthropic SDK conventions — but **must confirm in Phase 0 spike** since SDK import is not available in current shell.

**Codex CLI** (`codex_cli_inferencer.py`):
- Class at line 53; `construct_command()` at line 166; `_ainfer_streaming()` at line 345; `_infer()` at line 480.
- `large_input_mode == STDIN` (line 359). `has_local_access = True` (line 68).
- **CLI flag finding:** local `codex exec --help` on 2026-06-26 exposes `-i, --image <FILE>...` for the initial prompt. Local `codex exec resume --help` also exposes `-i, --image <FILE>` for the prompt sent after resuming. **Resolution:** use native `--image` for image path attachments when the installed command exposes it; keep staged path/manifest fallback for generic files, unsupported image sources, and future CLI drift.
- Codex CLI also supports path mentions / workspace references for file context in the prompt body. Use this for non-image files and as a fallback when native image flags are unavailable. *(Flow 2 summary + codex-focused plan.)*

**Codex SDK** (`codex_sdk_inferencer.py`):
- Class at line 71; `_ainfer_streaming()` at line 254; calls `thread.turn(prompt)` at line 276; `_infer()` at line 332.
- `has_local_access = True` (line 83).
- Local SDK check on 2026-06-26: `openai_codex 0.1.0b3` is importable. It exposes typed dataclasses `TextInput(text)`, `ImageInput(url)`, `LocalImageInput(path)`, and `MentionInput(name, path)`. `openai_codex.api.AsyncThread.turn(self, input: RunInput, ...)` accepts `RunInput`, defined by the package as `str | Input`. **Resolution:** keep `thread.turn(prompt)` unchanged for text-only calls; when attachments are present, pass a typed input list built from `TextInput`, `LocalImageInput` / `ImageInput`, and `MentionInput`.
- **Note:** `openai-codex` is NOT in `src/BUCK` even though the local Python environment has it. Absence from BUCK is an implementation packaging gap to resolve, not proof the SDK is unavailable. *(Flow 0 finding.)*

**AG API Inferencers** (shared pattern):
- `AgClaudeApiInferencer`: class at `ag_claude_api_inferencer.py:111`; `_messages_override` at line 154; `set_messages()` at line 168; `_infer()` at line 191, `_ainfer()` at line 204, `_ainfer_streaming()` at line 221.
- `AgOpenAIApiInferencer`: class at `ag_openai_api_inferencer.py:69`; `_messages_override` at line 108; `set_messages()` at line 122; `_infer()` at line 145, `_ainfer()` at line 157, `_ainfer_streaming()` at line 173.
- `AgGeminiApiInferencer`: class at `ag_gemini_api_inferencer.py:61`; `_messages_override` at line 95; `set_messages()` at line 108; `_infer()` at line 125, `_ainfer()` at line 136, `_ainfer_streaming()` at line 151.
- **Critical nuance:** `_messages_override` is only consulted by `_ainfer_streaming()`. Sync `_infer()` and non-streaming `_ainfer()` pass `inference_input` directly to backend helpers. This means `set_messages()` alone is insufficient — all three modes need explicit multimodal handling.

### 2.2 AG Backend Pass-Through Behavior

The AG backend `_get_messages()` functions **already pass through structured content blocks**:

- **Claude** (`ai_gateway_claude_llm.py:155-207`): when input is `List[Dict]`, normalizes only `content: str` → `content: [{type: "text", text: ...}]`; content arrays with image/document blocks pass through untouched (lines 195-204, verified by code read).
- **OpenAI** (`ai_gateway_openai_llm.py:105-140`): same pattern; dict/list messages preserved.
- **Gemini** (`ai_gateway_gemini_llm.py:93-118`): mirrors OpenAI-style normalization.

**AG backend change decision:** Flow 2 said "zero backend changes needed" because of this pass-through. Flow 0 proposed adding `attachments` parameter to all backend public functions. **Resolution:** Minimal backend changes — add `MultimodalInput` / `attachments` acceptance to `_get_messages()` in each backend (small branch that calls the provider serializer), while keeping the raw dict/list pass-through unchanged. This is needed because: (a) sync `_infer()` and non-streaming `_ainfer()` don't use `set_messages()`, and (b) centralizing serialization in backend helpers avoids duplicating it in each wrapper method.

### 2.3 Legacy File-Path-as-Prompt Behavior

AG backends treat a string input that is a local file path as UTF-8 prompt text:
- Claude: `path.exists()` at `ai_gateway_claude_llm.py:160-163`
- OpenAI: `path.isfile()` at `ai_gateway_openai_llm.py:117-123`
- Gemini: `path.isfile()` at `ai_gateway_gemini_llm.py:97-103`

This is semantically different from attaching an image/file. **Must preserve** for string inputs. Only explicit `attachments=` or `MultimodalInput.attachments` triggers attachment semantics. *(Flow 0 §6.)*

### 2.4 Shared Infrastructure & Cache Risk

- `InferencerBase` at `inferencer_base.py:69`; `has_local_access` at line 161; `effective_cwd` at line 377 (priority: `target_path > workspace.root > os.getcwd()`).
- `StreamingInferencerBase` at `streaming_inferencer_base.py:119`; `_extract_prompt()` at line 564 handles only strings and dicts with `prompt` key.
- **Cache risk** *(Flow 0 unique finding):* Streaming cache key depends only on prompt extraction. Two requests sharing the same prompt but different attachments would collide. Attachment fingerprint must be included in cache identity, or caching disabled for multimodal requests.
- `AgentAttachment` at `agent_attachment.py:33` has `id`, `description`, `content`, `formatter` — renders XML-ish prompt text via `.text`/`.full_text`. No MIME type, source path, URL, byte payload, base64 handling, or provider serialization. **Do not reuse** as the new media model. *(Flow 0 unique finding.)*

### 2.5 Capability Matrix

| Inferencer | Image Mechanism | File Mechanism | Key Constraint |
|---|---|---|---|
| Claude Code CLI | Stage to disk + `@<path>` mention (agent reads via Read tool) | Same — path reference | No native multimodal CLI flag verified; `extra_cli_args` available for future flags |
| Claude Code SDK | `client.query([{role:"user", content:[{type:"image", source:{...}}]}])` *(verify in Phase 0)* | `{type:"document", source:{...}}` for PDFs | Falls back to path+manifest if SDK rejects structured input |
| Codex CLI | Native `-i/--image` for image paths on local `exec` and `exec resume`; staged path/manifest fallback when unavailable | Path reference / manifest via Read tool | Probe in CI/dev before relying on native flag because CLI versions drift |
| Codex SDK | `openai_codex.LocalImageInput(path)` / `ImageInput(url)` *(typed dataclasses)* | `MentionInput(name, path)` for context files; no generic binary `FileInput` exposed locally | `thread.turn(str)` remains the text-only path; typed list only when attachments exist |
| AG Claude | `{type:"image", source:{type:"base64", media_type:..., data:...}}` | `{type:"document", source:{type:"base64", media_type:"application/pdf", data:...}}` | Route-dependent size limits (direct: 10MB; Bedrock may differ) |
| AG OpenAI | `{type:"image_url", image_url:{url:"data:image/...;base64,...", detail:"auto"}}` | `{type:"file", file:{filename:..., file_data:...}}` | Chat Completions format; do NOT mix Responses API shapes |
| AG Gemini | Same as OpenAI (OpenAI-compat layer) | Same as OpenAI | Gate behind capability test; gateway may reject |

### 2.6 Test Coverage Status

- **Claude Code:** dedicated tests at `test/agent_foundation/common/inferencers/external/claude_code/` (CLI flag, SDK, cache, integration).
- **Codex:** **no** test folder at `test/.../external/codex/` — must create from scratch.
- **AG gateway:** tests at `test/agent_foundation/apis/ag/`.
- **No existing multimodal tests** for any of the 7 inferencers (verified by grep).
- Large-input tests (`test_large_input_mode.py`, `test_large_arg_offload.py`) are text-delivery only.

### 2.7 Provider Documentation (Checked 2026-06-26)

*Provider docs are volatile; re-check before implementation.*

- **Anthropic Vision:** `{type:"image", source:{type:"base64", media_type:..., data:...}}`. PDF: `{type:"document", source:{type:"base64", media_type:"application/pdf", data:...}}`. *(Refs: `platform.claude.com/docs/en/build-with-claude/vision`, `.../pdf-support`)*
- **OpenAI Vision:** `{type:"image_url", image_url:{url:"data:image/...;base64,...", detail:"auto"}}`. Files: `{type:"file", file:{filename:..., file_data:...}}`. *(Refs: `developers.openai.com/api/docs/guides/images-vision`, `.../file-inputs`)*
- **Codex CLI:** local `codex exec --help` confirms `--image, -i <FILE>...` for fresh `exec`; local `codex exec resume --help` confirms `--image, -i <FILE>` for resumed turns. Keep command-construction tests because flag shape and placement can drift by CLI version. *(Local check 2026-06-26; external reference: `developers.openai.com/codex/cli/reference`)*
- **Gemini:** OpenAI-compatible chat/streaming; image block acceptance needs gateway-level testing. *(Ref: `ai.google.dev/gemini-api/docs/openai`)*
- **Claude Code CLI:** `claude -p` supports stdin prompt, `--input-format stream-json`. No native image flag found. *(Ref: `code.claude.com/docs/en/cli-reference`)*
- **Claude Agent SDK:** `query(prompt=...)` examples; agents use Read/Write/Edit/Bash tools. *(Ref: `code.claude.com/docs/en/agent-sdk/overview`)*

### 2.8 Dependencies

- `src/BUCK` declares: `attrs`, `httpx`, `pydantic`, `requests`, `pyyaml`, `claude-agent-sdk`.
- `src/BUCK` does NOT list: `openai-codex`, Pillow, `python-magic`, `filetype`.
- **First implementation uses stdlib only:** `mimetypes`, `base64`, `hashlib`, `pathlib`. No new third-party deps.

---

## 3. Architecture

### 3.1 Design Principles

1. **Shared value object, backend-specific adapters.** Single `MultimodalInput` carries images/files through the framework. Pure-function adapters convert to provider-native format at the serialization boundary. Each inferencer wires exactly one adapter.

2. **Strictly additive public API.** `str` and `dict` inputs take unchanged code paths. Multimodal is opt-in via `MultimodalInput`, dict-shortcut with `images`/`files` keys, or explicit `attachments=` kwarg.

3. **AG inferencers: `set_messages()` for streaming + explicit handling for sync/async.** The existing `set_messages()` → `_messages_override` flow works for streaming, but sync `_infer()` and non-streaming `_ainfer()` need explicit multimodal handling because they bypass `_messages_override`.

4. **CLI inferencers: staged path references.** Attachments materialized to disk under `effective_cwd`, referenced via `@<absolute_path>` mentions. Both Claude Code and Codex CLIs recognize this syntax. *(Flow 2 architecture.)*

5. **SDK inferencers: typed content payloads.** SDKs accept structured input directly — no filesystem materialization needed for in-memory attachments. Falls back to path+manifest if SDK rejects structured input.

6. **Files by capability, not by label.** *(Flow 0 principle.)*
   - **Images:** provider-native blocks for API; staged paths for CLI; typed content for SDK.
   - **PDFs:** Anthropic `document` block; OpenAI `file` block; Codex SDK file context via `MentionInput` unless a future SDK exposes a dedicated file input; path reference for CLI.
   - **Text files:** inline text (up to size cap) for API; path reference for agentic.
   - **Other binary:** path reference for local agentic only; clear rejection for API unless provider explicitly supports.

7. **No over-engineering.** No new abstract base class hierarchy. No full multimodal pipeline abstraction. Content model + adapter module + per-inferencer wiring.

### 3.2 Module Structure

**Conflict resolution:** Flow 1 co-located serializers in the model module. Flow 0 placed serializers in `src/agent_foundation/apis/ag/multimodal.py`. Flow 2 separated model + adapters. **Decision:** Follow Flow 2 — separate `multimodal.py` (model + normalization) and `multimodal_adapters.py` (pure converters), both under `src/agent_foundation/common/inferencers/`. This keeps provider serialization co-located with the inferencer layer (since adapters serve CLI/SDK too, not just AG).

```
src/agent_foundation/common/inferencers/
├── multimodal.py              # MultimodalInput, ImageAttachment, FileAttachment + helpers
├── multimodal_adapters.py     # Pure converters: to_anthropic, to_openai_chat, to_codex_sdk_typed_input, to_cli_mentions
├── ...existing files unchanged...
```

### 3.3 Data Model

**Naming resolution:** Flow 1 used `MultimodalContent`/`ImageContent`/`FileContent`; Flows 0/2 used `MultimodalInput`/`Attachment` variants. **Decision:** `MultimodalInput` (describes the role at the inferencer boundary), `ImageAttachment`/`FileAttachment` (distinguishes from existing `AgentAttachment`).

```python
# src/agent_foundation/common/inferencers/multimodal.py

@attrs(slots=True)
class ImageAttachment:
    source: str = attrib()              # File path, URL, or "inline"
    data: Optional[bytes] = attrib(default=None, repr=False)  # Raw bytes (repr=False for safety)
    media_type: Optional[str] = attrib(default=None)   # e.g. "image/png", "image/jpeg"
    detail: Optional[str] = attrib(default=None)       # OpenAI vision detail: "auto"/"low"/"high"
    name: Optional[str] = attrib(default=None)          # Display name

@attrs(slots=True)
class FileAttachment:
    path: str = attrib()
    content: Optional[str] = attrib(default=None)       # Pre-read text content
    media_type: Optional[str] = attrib(default=None)    # MIME type
    name: Optional[str] = attrib(default=None)          # Display name

@attrs(slots=True)
class MultimodalInput:
    prompt: str = attrib()                              # Text prompt (required)
    images: List[ImageAttachment] = attrib(factory=list)
    files: List[FileAttachment] = attrib(factory=list)
    messages: Optional[List[dict]] = attrib(default=None)  # Escape hatch for pre-built messages
```

**Why attrs over dataclass?** Consistency — every inferencer and data model in the codebase uses `@attrs`. *(All three flows agree.)*

**Utility functions in `multimodal.py`:**
- `MultimodalInput.normalize(x: Any) -> MultimodalInput` — accepts `str | dict | MultimodalInput`; dict with `images`/`files`/`attachments` keys auto-constructs; plain `str` wraps as `MultimodalInput(prompt=x)`.
- `sniff_media_type(path_or_name: str) -> str` — `mimetypes` + hard-coded extension table; accepts explicit override.
- `read_attachment_bytes(att, max_bytes: int) -> bytes` — reads from `source` path; rejects if size exceeds `max_bytes`.
- `encode_image_base64(image: ImageAttachment) -> Tuple[str, str]` — returns `(base64_data, media_type)`. Reads from `source` path if `data` is None; auto-detects `media_type` if not provided. *(Flow 1 detail.)*
- `compute_fingerprint(att) -> str` — SHA-256 for cache identity. *(Flow 0 requirement.)*
- `is_multimodal(x: Any) -> bool` — type check helper.

**Standard library only:** `mimetypes`, `base64`, `hashlib`, `pathlib`. No Pillow/`python-magic`/`filetype` — absent from `src/BUCK` and not needed. *(All three flows agree.)*

### 3.4 Adapter Functions

```python
# src/agent_foundation/common/inferencers/multimodal_adapters.py

def to_anthropic_content_blocks(mm: MultimodalInput) -> List[dict]:
    """Anthropic Messages API content blocks.

    Images:   {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    PDFs:     {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "..."}}
    Text:     {"type": "text", "text": "..."}
    """

def to_openai_chat_content(mm: MultimodalInput) -> List[dict]:
    """OpenAI Chat Completions content parts.

    Images:  {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "auto"}}
    Files:   {"type": "file", "file": {"filename": "...", "file_data": "data:...;base64,..."}}
    Text:    {"type": "text", "text": "..."}
    """

def to_codex_sdk_typed_input(mm: MultimodalInput) -> 'str | list[RunInput]':
    """Convert to Codex SDK typed dataclasses when available.

    Text-only: return mm.prompt.
    With attachments: use openai_codex.TextInput, LocalImageInput(path),
    ImageInput(url), and MentionInput(name, path).
    """

def to_cli_attachment_paths(mm: MultimodalInput, materialize_root: Path) -> List[Path]:
    """Materialize non-local attachments to disk; return absolute paths.

    Path-source: return as-is (absolute). Bytes/URL/base64-source: write to
    materialize_root/<sha256_prefix>/<name> and return the materialized path.
    """

def format_prompt_with_mentions(prompt: str, paths: List[Path]) -> str:
    """Append @<absolute_path> mention lines to prompt for CLI inferencers."""

def fingerprint_for_cache(mm: MultimodalInput) -> str:
    """Deterministic fingerprint of all attachments for cache key inclusion."""
```

### 3.5 Public API Surface (Backward Compatible)

```python
# Existing (unchanged)
result = inferencer("Describe this scene")
result = inferencer({"prompt": "Describe this scene"})

# New: MultimodalInput (full control)
mm = MultimodalInput(
    prompt="Describe this image",
    images=[ImageAttachment(source="/path/to/img.png")],
)
result = inferencer(mm)

# New: dict shortcut (convenience)
result = inferencer({"prompt": "Describe this image", "images": [{"source": "/path/to/img.png"}]})

# New: per-inferencer images attribute (applied to all calls)
inferencer = AgClaudeApiInferencer(images=["/path/to/img.png"])
result = inferencer("Describe this image")
```

All existing `str` / `dict-with-prompt` call sites use the identical code path. Multimodal branch gated by `isinstance(x, MultimodalInput)` or "dict with `images`/`files` keys".

---

## 4. Implementation Phases (Dependency-Ordered)

### Phase 0: Spike — Confirm Runtime Capabilities (≤ 0.5 day)

**Why:** The three upstream plans disagree on CLI flag support and SDK input signatures. Initial local probing has resolved the Codex side; Claude SDK and AG route behavior still need implementation-environment confirmation. *(Flow 2 originated this phase; Flow 0 required cautious capability probing.)*

**Tasks:**
1. Inspect installed CLIs and record exact versions/help excerpts:
   - `claude --help` / `claude -p --help` — check for image/file flags. Local 2026-06-26 result: stream-json exists; no local image flag; `--file` is file-resource startup syntax (`file_id:relative_path`), not general local attachment upload.
   - `codex exec --help` / `codex exec resume --help` — local 2026-06-26 result: both expose `-i/--image`. Implementation still needs command-construction tests for fresh and resume placement.
2. Inspect installed Python SDKs:
   - `claude_agent_sdk`: does `ClaudeSDKClient.query()` accept `List[dict]` (message list) or only `str`? Local `python3` import check on 2026-06-26 returned no module, despite `src/BUCK` declaring `claude-agent-sdk`.
   - `openai_codex`: local 2026-06-26 result: version `0.1.0b3`; `AsyncThread.turn(input: RunInput, ...)`; dataclasses `TextInput(text)`, `ImageInput(url)`, `LocalImageInput(path)`, `MentionInput(name, path)`.
3. Run one live test per AG backend with a 1×1 PNG:
   - AG Claude direct mode
   - AG OpenAI
   - AG Gemini (does gateway accept `image_url` content blocks?)
4. Record results in `artifacts/spike-notes.md`:
   - Per surface: image support, PDF support, text file support, binary behavior, size limits
   - Final transport decision per backend

**Acceptance Criteria:**
- Spike notes exist with concrete SDK signatures observed from installed packages.
- No code path depends on undocumented CLI flags without local probe.
- `openai-codex` packaging status clarified (build/BUCK updates needed?).
- Transport decision per backend locked (structured query vs `@<path>` mention vs native flag).

### Phase 1: Shared Value Object & Adapters

**Create:**
- `src/agent_foundation/common/inferencers/multimodal.py`
- `src/agent_foundation/common/inferencers/multimodal_adapters.py`
- `test/agent_foundation/common/inferencers/test_multimodal.py`

**Implement:**
- `attrs` data model (§3.3), normalization helpers, MIME inference, size validation, fingerprinting, redacted display
- All adapter functions (§3.4)
- Validation: path exists, regular file, readable, under configured size cap (default 10 MB image, 32 MB file)
- Provider-specific allow-lists: Anthropic `image/jpeg|png|gif|webp`; OpenAI chat file blocks only where AG route accepts them; Codex files via `MentionInput` unless the installed SDK adds a dedicated file input
- **Redacted display:** `repr`/debug rendering never includes raw bytes or base64 *(Flow 0 §7)*

**Acceptance Criteria:**
- Unit tests cover: media-type sniffing, all source-form normalization (str, dict, `MultimodalInput`, keyword), size-limit rejection, non-existent/directory/unreadable paths, explicit MIME override, URL/data-URI forms
- Golden payload tests: adapter output for 1×1 PNG matches Anthropic/OpenAI specs verbatim
- `MultimodalInput.normalize("plain string")` produces same normalized prompt as existing `str` flow
- Sanitized `repr` never includes base64 or raw bytes

### Phase 2: AG API Inferencers (Lowest Risk, Highest Leverage)

**Modify AG backends (minimal changes):**
- `ai_gateway_claude_llm.py` — add `MultimodalInput` branch to `_get_messages()`: calls `to_anthropic_content_blocks()`
- `ai_gateway_openai_llm.py` — add `MultimodalInput` branch to `_get_messages()`: calls `to_openai_chat_content()`
- `ai_gateway_gemini_llm.py` — add `MultimodalInput` branch to `_get_messages()`: calls `to_openai_chat_content()` (Gemini uses OpenAI-compat)

**Modify AG wrappers:**
- `ag_claude_api_inferencer.py`, `ag_openai_api_inferencer.py`, `ag_gemini_api_inferencer.py`
- Add private `_resolve_inference_input(self, inference_input)` helper:
  - `str | dict-without-images-or-files` → behave as today (zero-change path)
  - Else normalize → for streaming: build content blocks + `set_messages()`; for sync/async: pass `MultimodalInput` through to backend `_get_messages()`
- Call from `_infer`, `_ainfer`, `_ainfer_streaming` — one entry point per method
- Add `supports_images: bool`, `supports_files: bool` class-level flags. For Gemini, gate by Phase 0 probe.

**Compatibility:**
- `set_messages()` + explicit attachments → raise clear error (ambiguous merge)
- Streaming resets `_messages_override` after use (existing single-shot semantics preserved)
- Raw dict/list pass-through unchanged
- Legacy path-as-prompt unchanged for string inputs

**Acceptance Criteria:**
- Mocked transport tests assert:
  - String input: identical call args to today
  - `MultimodalInput(prompt="...", images=[...])`: correct provider-specific content blocks
  - Oversized image: rejected with `ValueError`
  - All three modes (sync, async, streaming): serialize through same helper
  - `set_messages()` raw override unchanged
  - `set_messages()` + attachments: clear error
  - Legacy path-as-prompt: text reading unchanged

### Phase 3: SDK Inferencers (Claude Code SDK + Codex SDK)

**Modify:**
- `claude_code_sdk_inferencer.py`
- `codex_sdk_inferencer.py`

**Claude Code SDK:**
- In `_ainfer_streaming()`, before `await client.query(prompt)`:
  - `mm = MultimodalInput.normalize(inference_input)`
  - If attachments: `payload = [{"role":"user", "content": to_anthropic_content_blocks(mm)}]` → `await client.query(payload)`
  - Else: `payload = mm.prompt` (string, as today)
- Jinja templating (`_render_prompt()`) applies only to `mm.prompt`; attachments untouched
- **Fallback:** if installed SDK rejects structured content (Phase 0 finding), write images to workspace temp files + reference paths in prompt

**Codex SDK:**
- Same pattern: detect multimodal → build typed input via `to_codex_sdk_typed_input(mm)`:
  - `TextInput(mm.prompt)` for the text prompt
  - `LocalImageInput(path)` for local images
  - `ImageInput(url)` for URL images
  - `MentionInput(name, path)` for context files and non-native file references
- Pass the typed list to the existing `thread.turn(items)` call site. Preserve `thread.turn(prompt)` for text-only calls.
- Non-PDF / generic files: use `MentionInput` or materialize to disk + path mention; do not invent an unsupported `FileInput`.

**Acceptance Criteria:**
- Mocked `ClaudeSDKClient` / `AsyncCodex.thread` assert:
  - `query()` / `turn()` called with typed content when attachments present
  - Called with plain string when no attachments
  - Sync `infer()` works without event loop issues (regression)
  - Non-PDF files for Codex SDK: correct fallback to manifest

### Phase 4: CLI Inferencers (Claude Code CLI + Codex CLI)

**Modify:**
- `claude_code_cli_inferencer.py`
- `codex_cli_inferencer.py`

**Shared strategy (both CLIs):**
1. In `construct_command()`, normalize via `MultimodalInput.normalize(inference_input)`
2. If attachments:
   - Create per-call staging dir: `Path(self.effective_cwd) / ".agent_attachments" / <uuid4>`
   - `paths = to_cli_attachment_paths(mm, materialize_root)` to stage non-local attachments
   - `prompt = format_prompt_with_mentions(mm.prompt, paths)` to build augmented prompt for path-reference transports
   - Use augmented prompt in existing stdin/arg flow
3. If no attachments: `prompt = mm.prompt` (identical to today)
4. Cleanup in `_ainfer` / `_infer` via `try/finally` removing staging directory

**Transport by CLI:**
- **Claude Code CLI:** default to staged path references / manifest. Do not use Claude `--file` unless the attachment source is explicitly a remote file resource (`file_id:relative_path`). Stream-json image input remains a spike-gated enhancement, not the default path.
- **Codex CLI:** use native `-i/--image` for local/staged image path attachments when help probing confirms support (confirmed locally on 2026-06-26 for both fresh `exec` and `exec resume`). Use path manifest / `@<path>` references for generic files and as fallback if a future CLI lacks the flag. Test flag placement for both command shapes. *(Flow 1 native-flag idea + Flow 0 probe requirement + Flow 2 strategy ladder.)*

**Path safety** *(Flow 0 requirements):*
- Resolve real paths via `pathlib.Path.resolve()`
- Validate regular files only (reject symlinks escaping allowed roots)
- Files must be under `effective_cwd` or in `additional_allowed_paths`
- Outside-cwd-but-allowed files: copy into staging directory
- Outside-cwd disallowed files: reject with clear error

**Read tool injection:** Ensure `allowed_tools` includes `"Read"` when attachments present (auto-inject if missing; log at info). Critical for `enable_shell=False` callers. *(Flow 2 unique requirement.)*

**Staging hygiene:**
- `.agent_attachments/` → `.gitignore` entry on first creation (idempotent) *(Flow 2)*
- Content-addressed: `<sha256_prefix>/<filename>` *(Flow 0)*
- Always cleaned up in `finally` block

**Acceptance Criteria:**
- Mocked subprocess tests verify:
  - No attachments → command identical to today
  - Attachments → prompt contains `@/abs/path/to/file.png` lines; bytes written to staging; staging gone after call
  - `allowed_tools` augmented with `Read` when missing
  - Under-cwd files: referenced by path (not staged)
  - Outside-cwd allowed: staged
  - Outside-cwd disallowed: clear error
  - Prompt has manifest metadata, not base64
  - `large_input_mode=STDIN` still sends final prompt via stdin

### Phase 5: Pipeline Integration — Template Rendering & Caching

**Template rendering:**
- Each multimodal-capable inferencer's `_resolve_inference_input()` / `_ainfer()` extracts `mm.prompt` for Jinja2 rendering, then re-wraps with original attachments. Localized per-inferencer, low-risk.
- Template variables apply to text only. Attachments never template-rendered.

**Streaming cache** *(Flow 0 unique requirement):*
- Add cache key override: when attachments present, include `fingerprint_for_cache(mm)` in key
- Alternative: disable streaming cache for multimodal requests until hooks updated
- Same prompt + different fingerprints → no cache reuse
- Same prompt + same fingerprints → may reuse

**Logging & redaction:**
- Raw bytes/base64 never in logs, cache keys, checkpoints, exception messages
- Debug logs: filename, MIME type, byte size, SHA-256 prefix, staged relative path only

**Acceptance Criteria:**
- Template variables apply to text; images preserved
- Cache identity changes with attachment content
- Logs contain no raw base64
- No leftover `.agent_attachments/` after test suite

### Phase 6: Documentation & Type Hints

- Each target inferencer's docstring: add `Multimodal usage` section
- `__init__.py` files: re-export `MultimodalInput`, `ImageAttachment`, `FileAttachment`
- `src/agent_foundation/apis/ag/README.md`: document multimodal convenience layer
- Type checker: `pyre` (per `# pyre-strict` headers) passes with zero new errors

### Phase 7: End-to-End Smoke Tests (Env-Gated)

- `test/agent_foundation/common/inferencers/test_multimodal_e2e.py`
- Parameterized over all 7 inferencers
- Skips by default; activates with `RUN_LIVE_MULTIMODAL_E2E=1` + required auth/CLI/SDK
- Programmatically-generated fixtures: 1×1 PNG, 1-page PDF
- **Results → `artifacts/multimodal_e2e_results.json` (NOT stdout)** — per-backend pass/fail + latency + token usage *(Flow 2 requirement)*

### Phase Dependency Graph

```
Phase 0: Spike (0 dependencies)
    ↓
Phase 1: Shared value object & adapters (depends on Phase 0 decisions)
    ↓
Phase 2: AG API inferencers ─────────────┐
Phase 3: SDK inferencers (Claude/Codex) ──┤ All depend on Phase 1; independent of each other
Phase 4: CLI inferencers (Claude/Codex) ──┘
    ↓
Phase 5: Pipeline integration (depends on Phases 2-4)
    ↓
Phase 6: Documentation (depends on Phase 5)
    ↓
Phase 7: E2E smoke tests (depends on all)
```

Phases 2, 3, and 4 can proceed in parallel after Phase 1 is complete.

---

## 5. Risk Register

*Merged from the upstream flows and local repo plan artifacts (Flow 0: 14 risks, Flow 1: 8 risks, Flow 2: 10 risks, plus Codex-focused staging/manifest risks). Deduplicated, severity-harmonized, best mitigations kept.*

| ID | Severity | Risk | Failure Mode | Mitigation |
|----|----------|------|--------------|------------|
| R1 | 🔴 High | Gateway route incompatibility | AG direct/Bedrock/proximity/Gemini-compat routes reject content blocks provider docs support elsewhere. | Route-aware capability gates in Phase 0. Fail locally with clear unsupported-mode errors before sending. *(Flows 0, 1.)* |
| R2 | 🔴 High | Base64 leakage | Raw image/PDF bytes appear in logs, cache keys, checkpoints, test snapshots, or exception messages. | `repr=False` on data fields; centralize sanitized display in `multimodal.py`; unit test redaction. *(Flow 0 §7.)* |
| R3 | 🔴 High | Path safety / symlink escapes | Staging reads files outside allowed roots or follows symlinks unexpectedly. | Resolve real paths, validate regular files, require allowed roots, copy into controlled staging, test symlink escape. *(Flow 0 §5.)* |
| R4 | 🔴 High | Misleading "file support" | Arbitrary binaries accepted but model can't inspect them → false confidence. | Define support by MIME × provider route. Images/text/PDFs first. Reject unknown binary for API unless provider explicitly supports. *(Flow 0 §3.)* |
| R5 | 🟡 Med | Claude Code CLI multimodal ambiguity | Stream-json image input rejected or silently ignored by CLI. | Default to staged path + `@<path>` mention. Only add stream-json image input after empirical Phase 0 proof. *(Flows 0, 2.)* |
| R6 | 🟡 Med | Codex CLI flag drift | Local 2026-06-26 CLI supports `--image`, but future/CI versions may differ or require different placement. | Probe `codex exec --help`; test flag placement for both `exec` and `exec resume`; keep manifest fallback. *(Flows 0, 1, 2.)* |
| R7 | 🟡 Med | SDK capability uncertainty | `claude-agent-sdk` `query()` may not accept message list; `openai-codex` `turn()` shape may differ across versions. | Phase 0 spike pins installed version + signature. Feature-detection fallback: downgrade to string + `@<path>`. *(Flow 2 R2.)* |
| R8 | 🟡 Med | Streaming cache collisions | Same prompt + different attachments reuses old cached output. | Include attachment fingerprint in cache key, or disable cache for multimodal requests. *(Flow 0 unique.)* |
| R9 | 🟡 Med | Raw message override regression | Existing `set_messages()` callers lose structured pass-through. | Keep override unchanged. Error if both `set_messages()` and attachments provided. Regression tests. *(Flow 2 R7.)* |
| R10 | 🟡 Med | Token / payload bloat | Text files or PDFs inlined too aggressively exceed limits. | Size caps (10 MB image, 32 MB file). Prefer path references for agentic. Explicit too-large errors. |
| R11 | 🟡 Med | Sync/async/streaming drift | Attachments work in one AG mode but silently ignored in another. | Route all modes through shared `_resolve_inference_input()`; test all three per provider. *(Flow 0.)* |
| R12 | 🟡 Med | Legacy path-as-prompt ambiguity | String path becomes attachment unexpectedly. | Preserve legacy for string input. Only explicit `attachments=` / `MultimodalInput` triggers attachment semantics. *(Flow 0 §6.)* |
| R13 | 🟡 Med | Staging directory pollution | `.agent_attachments/` in `git status` or leftover after crashes. | UUID subdir; `.gitignore` on first creation; `finally`-block cleanup; content-addressed. *(Flow 2 R5.)* |
| R14 | 🟢 Low | `_messages_override` single-shot semantics | Cleared after streaming; multimodal callers using sync `_infer` may be surprised. | Document semantics (existing contract). Unit test. *(Flow 2 R7.)* |
| R15 | 🟢 Low | Jinja rendering of attachments | Template variables in attachments not rendered (unexpected?). | Contract: only `.prompt` rendered. Documented, tested. *(Flow 2 R8.)* |
| R16 | 🟢 Low | MIME guessing inaccuracies | `mimetypes` returns `application/octet-stream` for valid files. | Explicit `mime_type` override; clear error messages. No new deps. *(Flow 0.)* |
| R17 | 🟢 Low | Provider allow-list drift | Future-incompatible rejections. | Allow-lists in `multimodal_adapters.py` constants — single update point. *(Flow 2 R9.)* |
| R18 | 🟢 Low | Scope creep to conversational/server APIs | Text-only schema boundaries tempt broad refactors. | Keep to 7 target inferencers. Document future expansion separately. *(Flow 0.)* |

---

## 6. Files to Create / Modify

### New Files

| Path | Purpose |
|------|---------|
| `src/agent_foundation/common/inferencers/multimodal.py` | `MultimodalInput`, `ImageAttachment`, `FileAttachment` + normalization, validation, MIME inference, fingerprinting, redacted display |
| `src/agent_foundation/common/inferencers/multimodal_adapters.py` | Pure converters: `to_anthropic_content_blocks`, `to_openai_chat_content`, `to_codex_sdk_typed_input`, `to_cli_attachment_paths`, `format_prompt_with_mentions`, `fingerprint_for_cache` |
| `test/agent_foundation/common/inferencers/test_multimodal.py` | Value object + adapter unit tests (Phase 1) |
| `test/agent_foundation/common/inferencers/api_inferencers/ag/test_ag_multimodal.py` | AG inferencer multimodal tests, mocked transport (Phase 2) |
| `test/agent_foundation/common/inferencers/agentic_inferencers/external/test_sdk_multimodal.py` | SDK inferencer tests, mocked SDK clients (Phase 3) |
| `test/agent_foundation/common/inferencers/agentic_inferencers/external/test_cli_multimodal.py` | CLI inferencer tests, mocked subprocess (Phase 4) |
| `test/agent_foundation/common/inferencers/test_multimodal_e2e.py` | Env-gated live E2E → `artifacts/multimodal_e2e_results.json` (Phase 7) |
| `test/agent_foundation/common/inferencers/external/codex/` | New Codex test directory (currently missing) |
| `test/fixtures/test_image.png` | Programmatically-generated 1×1 PNG |
| `test/fixtures/test_image.jpg` | Same image, JPEG format |
| `artifacts/spike-notes.md` | Phase 0 deliverable: probed SDK/CLI signatures + transport decisions |

### Modified Files

| Path | Changes |
|------|---------|
| `src/.../ag/ai_gateway_claude_llm.py` | `_get_messages()`: add `MultimodalInput` branch → `to_anthropic_content_blocks()` |
| `src/.../ag/ai_gateway_openai_llm.py` | `_get_messages()`: add `MultimodalInput` branch → `to_openai_chat_content()` |
| `src/.../ag/ai_gateway_gemini_llm.py` | `_get_messages()`: add `MultimodalInput` branch → `to_openai_chat_content()` |
| `src/.../ag/ag_claude_api_inferencer.py` | Add `_resolve_inference_input()`; call from `_infer`, `_ainfer`, `_ainfer_streaming`; capability flags |
| `src/.../ag/ag_openai_api_inferencer.py` | Same pattern |
| `src/.../ag/ag_gemini_api_inferencer.py` | Same pattern; capability flags gated by Phase 0 |
| `src/.../claude_code/claude_code_cli_inferencer.py` | Multimodal `construct_command()`; staging + `@<path>` mentions; `Read` tool injection; cleanup |
| `src/.../claude_code/claude_code_sdk_inferencer.py` | Multimodal `_ainfer_streaming()` + `_ainfer()`; typed content or path+manifest fallback |
| `src/.../codex/codex_cli_inferencer.py` | Multimodal `construct_command()`; native `-i/--image` for images when probed; manifest/path fallback |
| `src/.../codex/codex_sdk_inferencer.py` | Multimodal `_ainfer_streaming()`; `LocalImageInput`/`ImageInput`/`MentionInput` or manifest |
| `src/.../claude_code/__init__.py` | Re-export multimodal types |
| `src/.../codex/__init__.py` | Re-export multimodal types |
| `src/.../api_inferencers/ag/__init__.py` | Re-export multimodal types |
| `src/agent_foundation/apis/ag/README.md` | Document multimodal convenience layer |

### Files Explicitly NOT Modified *(Flow 2 guard list, extended)*

| Path | Reason |
|------|--------|
| `src/agent_foundation/agents/agent_attachment.py` | Do not repurpose; optional future adapter only |
| `src/agent_foundation/common/inferencers/inferencer_base.py` | Base class stays `Any`-typed for `inference_input` |
| `src/agent_foundation/common/inferencers/streaming_inferencer_base.py` | Narrow cache hook only if per-inferencer overrides insufficient (Phase 5 decision) |
| `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/inbox.py` | Out of scope — text-oriented conversational schema |
| `src/agent_foundation/server/schema.py` | Out of scope — text-oriented server schema |
| `src/agent_foundation/common/configs/registered_targets.py` | No alias changes needed |

---

## 7. Validation & Testing Strategy

### 7.1 Test Pyramid

**Layer 1 — Unit (fast, hermetic, every CI):**
- `test_multimodal.py`: value object + adapter coverage. Media-type sniffing, source-form normalization, size rejection, path validation, fingerprinting, redacted display, golden payload output.
- `test_ag_multimodal.py`: monkeypatch AG backends; assert `_messages_override` / `_get_messages()` populated with provider-correct blocks. Text-only snapshot comparison.
- `test_sdk_multimodal.py`: monkeypatch `ClaudeSDKClient` / `AsyncCodex` (autospec); assert `query()`/`turn()` arguments per input shape.
- `test_cli_multimodal.py`: stub subprocess; capture command + materialized attachment dir; assert cleanup.

**Layer 2 — Contract/Regression (CI on-demand):**
- Re-run: `test_inferencer_axes_contract.py`, `test_inferencers_minimal.py`, `test_dual_inferencer/` suite
- Snapshot: command-string output with no attachments matches pre-change golden output
- Large input: `test_large_input_mode.py`, `test_large_arg_offload.py`

**Layer 3 — Live E2E (env-gated):**
- `test_multimodal_e2e.py` parameterized over all 7 inferencers
- `RUN_LIVE_MULTIMODAL_E2E=1` + auth env
- Results → `artifacts/multimodal_e2e_results.json`

### 7.2 Security & Regression Tests

- Symlink escape out of allowed roots → rejected
- Outside-cwd allowed file → staged
- Outside-cwd disallowed → clear error
- Reuse staged path for identical fingerprint
- Cache key changes with attachment content
- Logs/checkpoints contain no raw base64
- No leftover `.agent_attachments/` after suite

### 7.3 Test Commands

```bash
# Phase 1
python -m pytest test/agent_foundation/common/inferencers/test_multimodal.py
# Phase 2
python -m pytest test/agent_foundation/common/inferencers/api_inferencers/ag/test_ag_multimodal.py
# Phase 3
python -m pytest test/agent_foundation/common/inferencers/agentic_inferencers/external/test_sdk_multimodal.py
# Phase 4
python -m pytest test/agent_foundation/common/inferencers/agentic_inferencers/external/test_cli_multimodal.py
# Regression
python -m pytest test/agent_foundation/common/inferencers
python -m pytest test/agent_foundation/apis/ag
# E2E
RUN_LIVE_MULTIMODAL_E2E=1 python -m pytest test/agent_foundation/common/inferencers/test_multimodal_e2e.py
```

### 7.4 Quality Gates Before Merge

1. All Layer 1 + Layer 2 tests green locally
2. `pyre` reports zero new errors on changed files
3. At least one Layer 3 E2E pass per backend in `artifacts/multimodal_e2e_results.json`, linked in PR
4. Docstrings on all 7 inferencer classes show multimodal usage pattern
5. `git status` after unit suite shows no leftover `.agent_attachments/`

### 7.5 Task-Tool Regression Note

The Codex-focused upstream plan recorded a separate task-tool/runtime issue: a plan-only smoke test under `_runtime/tasks/attachment_support_plan_test/attachment_support_plan_test_20260626_163754_266eed24` produced an empty `children/propose/outputs/output.md`, and the RovoDev breakdown process was killed after 120 seconds with no output. This is not part of multimodal runtime support, but it is worth preserving as a regression check for the planning pipeline:

- Re-run the same plan-only task after the task-tool/RovoDev empty-output issue is fixed or avoided.
- Acceptance: `children/propose/outputs/output.md` is non-empty and the task exits without manual interruption.

---

## 8. Source Coverage and Reconciliation

### 8.1 Sources Audited

| Source | Coverage Preserved Here | Drill-In |
|---|---|---|
| Flow 1 plan | API payload examples, attrs model preference, convenience constructor/kwargs patterns, Codex native `--image` proposal, template-rendering survival concern | `_docs/_plan/multimodal_inferencer_support/PLAN.md` `§1 High-Level Approach`, `§2 Key Implementation Steps`, `§6 Detailed Design Notes` |
| Flow 0 plan | 7-target scope, security requirements, `AgentAttachment` non-reuse, legacy path-as-prompt behavior, `StreamingInferencerBase._extract_prompt()` cache risk, provider docs dated 2026-06-26, comprehensive validation strategy | `_docs/_plan/inferencer_architecture/multimodal_external_and_ag_inferencers_plan.md` `§Evidence-Based Findings`, `§High-Level Approach`, `§Risk Register`, `§Validation and Testing Strategy` |
| Codex-focused upstream plan | Shared envelope/staging idea, Codex CLI native `--image`, Claude `--file` caveat, `attachment_staging` helper rationale, task-tool zero-byte regression note | `_docs/_plan/inferencer_architecture/codex/multimodal_attachment_support_external_and_ag_inferencers_plan.md` `§Current Findings`, `§Implementation Plan`, `§Testing Strategy`, `§Success Criteria` |
| Flow 2 summary | Standardized 7 inferencer classes, `MultimodalInput` naming, CLI strategy ladder, Codex SDK typed `RunInput` dataclasses, `Read` tool auto-injection, env-gated E2E JSON output, "not modified" guard list | Prompt-provided summary for Flow 2; claimed `outputs/final_deliverables/plan.md`, but that path is absent in this workspace |
| Direct local verification | Corrected stale CLI/SDK assumptions: Codex CLI has `-i/--image` for fresh and resumed `exec`; Claude CLI has stream-json but no general local image flag; `claude_agent_sdk` not importable in this shell; `openai_codex 0.1.0b3` exposes `TextInput`, `ImageInput`, `LocalImageInput`, `MentionInput` | Commands run 2026-06-26: `codex exec --help`, `codex exec resume --help`, `claude --help`, `python3 -c ...` |

### 8.2 Contradiction Resolutions

| Topic | Upstream Disagreement | Final Resolution |
|---|---|---|
| Codex CLI image transport | Flow 1/codex plan preferred native `--image`; Flow 0/Flow 2 required probing and fallback | Use native `-i/--image` for image paths when help probing confirms it. Local CLI confirms support for fresh and resumed `exec`; keep staged path/manifest fallback for generic files and CLI drift. |
| Claude Code CLI image transport | Flow 1 proposed stream-json image blocks; Flow 0/codex plan warned no general local image flag; Flow 2 preferred path mentions | Default to staged local paths plus manifest / `@<path>` references. Treat stream-json image blocks as spike-gated because local help exposes stream-json but no image flag. Do not misuse Claude `--file` except for explicit remote file-resource IDs. |
| AG backend changes | Flow 2 said raw message pass-through means no backend changes; Flow 0 proposed broader backend changes | Add only a small `MultimodalInput` branch in each AG `_get_messages()` helper. Keep raw dict/list pass-through unchanged. This covers sync, async, and streaming because `set_messages()` is only consulted in streaming wrappers. |
| AG `set_messages()` | Some flows treated `set_messages()` as sufficient | Preserve it as a raw provider-message escape hatch, but do not rely on it for normalized attachment support. Raise a clear error when callers combine raw override messages with neutral attachments. |
| Codex SDK input shape | Some upstream text described generic OpenAI Responses-style dicts; Flow 2/local checks found SDK dataclasses | Prefer the local typed surface: `TextInput`, `ImageInput`, `LocalImageInput`, `MentionInput`, passed to `thread.turn(input: RunInput)`. Avoid over-specifying internal import paths. |
| Shared model naming | Flow 1 used `MultimodalContent` / `ImageContent`; Flow 0/Flow 2 used `MultimodalInput` / attachment names | Use `MultimodalInput`, `ImageAttachment`, and `FileAttachment` to avoid confusion with provider `content` blocks and existing `AgentAttachment`. |
| Shared base-class edits | Codex-focused plan proposed broad normalization through streaming/template bases; Flow 0/Flow 2 preferred lower blast radius | Keep first implementation localized to the 7 inferencers and backend helpers. Add a narrow cache-key hook only if per-inferencer cache overrides cannot safely include attachment fingerprints. |
| Output location | Flow 2 summary claimed `outputs/final_deliverables/plan.md`; existing repo consolidation uses `_docs/_plan/...` | Keep the durable repo artifact at `_docs/_plan/multimodal_inferencer_support/CONSOLIDATED_PLAN.md`; `outputs/` does not exist in this workspace. |

### 8.3 Coverage Check

- Flow 1 unique details are preserved in the provider payload examples, attrs rationale, convenience API section, and template-rendering phase, but corrected where native CLI assumptions were too broad.
- Flow 0 unique details are preserved in path safety, symlink/allowed-root validation, base64 redaction, cache fingerprinting, `AgentAttachment` non-reuse, legacy path-as-prompt preservation, and the risk register.
- The Codex-focused plan's unique details are preserved through native Codex `--image`, staging/manifest helper behavior, Claude `--file` caveat, and task-tool regression coverage in testing notes.
- Flow 2 unique details are preserved in the 7-class target matrix, `MultimodalInput` naming, strategy ladder, Codex SDK typed dataclasses, `Read` tool injection, env-gated E2E JSON artifact, and explicit guard list.
- Direct local verification corrected the consolidated plan's stale conclusions around Codex CLI and Codex SDK. This is the main value added by this integration pass.

### 8.4 Integration-Value Judgment

The integration added meaningful value. The upstream inputs were not mostly identical: they disagreed on Codex CLI image support, Claude CLI stream-json viability, AG backend changes, and Codex SDK payload shape. Consolidation resolved those disagreements against local code and installed-tool evidence while preserving unique security, cache, staging, and validation details.

```json iteration_judgment
{
  "decision": "continue",
  "reason": "Integration surfaced implementation-critical signal beyond any single upstream input: local Codex CLI and SDK capabilities, the AG sync/async set_messages gap, and a corrected CLI strategy ladder. The consolidated artifact is internally consistent, but the upstream divergence was substantive rather than duplicative."
}
```

---

## 9. Self-Validation Checklist

- [x] All 7 inferencers addressed (2 Claude Code + 2 Codex + 3 AG API)
- [x] Evidence section cites real file paths and line ranges from codebase reads
- [x] AG backend pass-through verified: `_get_messages()` already handles structured content
- [x] AG sync/async gap identified: `_messages_override` only consulted by streaming
- [x] Architecture: shared value object + adapter module; minimal AG backend changes
- [x] CLI strategy: Claude path/manifest fallback; Codex native `-i/--image` when probed, with path/manifest fallback
- [x] SDK strategy: Codex typed dataclasses verified locally; Claude SDK structured media remains spike-gated with path+manifest fallback
- [x] Backward compatibility: string inputs take unchanged code paths
- [x] Phased with dependency ordering and per-phase acceptance criteria
- [x] Risk register: 18 items (4🔴/9🟡/5🟢) covering security, compatibility, SDK uncertainty
- [x] File list: new + modified + explicit "not modified" guard
- [x] Testing: unit/contract/E2E layers + security regression
- [x] Distinguishes image, text file, PDF, and unsupported binary per provider route
- [x] No over-engineering: no new base class hierarchy; no full pipeline abstraction
- [x] Provider documentation dated and flagged as volatile
- [x] Upstream flow disagreements resolved with explicit rationale
- [x] Coverage verified against upstream inputs, repo plan artifacts, and direct local capability checks
