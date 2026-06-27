# Plan: Multimodal (Image & File) Support for Inferencers

## 1. High-Level Approach

### Problem Statement

Five inferencers currently accept **text-only** prompts and cannot natively pass images or files as multimodal content to their backends:

| Inferencer | Backend | Current Input |
|---|---|---|
| `ClaudeCodeCliInferencer` | Claude Code CLI subprocess | Plain text string → positional arg or stdin |
| `ClaudeCodeSdkInferencer` | `claude_agent_sdk.ClaudeSDKClient.query()` | Plain text string |
| `CodexCliInferencer` | `codex exec` CLI subprocess | Plain text string → positional arg or stdin |
| `CodexSdkInferencer` | `openai_codex.AsyncCodex` thread turn | Plain text string |
| `AgClaudeApiInferencer` | AI Gateway → Anthropic Messages API | String or message dicts (text blocks only) |
| `AgOpenAIApiInferencer` | AI Gateway → OpenAI Chat Completions | String or message dicts (text content only) |
| `AgGeminiApiInferencer` | AI Gateway → Gemini (OpenAI-compat) | String or message dicts (text content only) |

### Architecture Decision: Shared Multimodal Content Model

Rather than ad-hoc image/file handling in each inferencer, we introduce a **shared content model** in the common inferencer layer that all five inferencer families can consume. This follows the existing pattern where `InferencerBase` defines common interfaces (`inference_input`, `_render_prompt`) and leaves use to each backend.

#### Key Design Principles

1. **Shared content model, backend-specific serialization** — A common `MultimodalContent` data class carries images/files through the framework pipeline. Each inferencer converts it to its backend's native format only at the serialization boundary.

2. **Backward-compatible** — All existing callers that pass `str` or `dict` prompts continue to work unchanged. Multimodal content is opt-in.

3. **Leverage existing `set_messages()` for AG inferencers** — The three AG inferencers already have `set_messages(messages: list)` which accepts raw API message dicts. The multimodal content model builds these message dicts, so the AG inferencers can adopt multimodal with minimal changes.

4. **CLI inferencers use native flags** — Claude Code CLI supports `--input-format stream-json` for structured input; Codex CLI supports `--image` / `-i` flags. We use these native mechanisms rather than workarounds.

5. **No over-engineering** — We add a content model and per-backend serializers, not a full multimodal pipeline abstraction. The scope is: accept images/files → serialize for backend → pass through.

---

## 2. Key Implementation Steps

### Phase 1: Shared Multimodal Content Model

**Goal:** Define the shared data classes that represent images and files in a backend-agnostic way.

**File to create:** `src/agent_foundation/common/inferencers/multimodal_content.py`

This module provides:

```
@attrs
class ImageContent:
    source: str                    # File path, URL, or "inline"
    data: Optional[bytes]          # Raw bytes (for inline/base64)
    media_type: Optional[str]      # e.g. "image/png", "image/jpeg"
    detail: Optional[str]          # OpenAI vision detail level: "auto", "low", "high"

@attrs
class FileContent:
    path: str                      # File path on disk
    content: Optional[str]         # Pre-read text content (if path not accessible by backend)
    media_type: Optional[str]      # MIME type, e.g. "text/plain", "application/pdf"

@attrs
class MultimodalContent:
    text: str                      # The text prompt (required)
    images: List[ImageContent]     # Zero or more images
    files: List[FileContent]       # Zero or more file attachments
```

**Utility functions in the same module:**

- `encode_image_base64(image: ImageContent) -> Tuple[str, str]` — Returns `(base64_data, media_type)`. Reads from `image.source` path if `image.data` is None; auto-detects `media_type` from file extension if not provided.

- `to_anthropic_content_blocks(mc: MultimodalContent) -> List[dict]` — Converts to Anthropic Messages API content block format:
  ```python
  [
      {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
      {"type": "text", "text": "..."},
  ]
  ```

- `to_openai_content_parts(mc: MultimodalContent) -> List[dict]` — Converts to OpenAI Chat Completions content parts format:
  ```python
  [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "auto"}},
      {"type": "text", "text": "..."},
  ]
  ```

- `is_multimodal(inference_input: Any) -> bool` — Type check helper.

**Acceptance Criteria:**
- Unit tests for `encode_image_base64` with PNG, JPEG, and inline bytes.
- Unit tests for `to_anthropic_content_blocks` and `to_openai_content_parts` round-trip.
- Existing inferencers continue to work with `str` inputs (no regressions).

---

### Phase 2: AG API Inferencers — Image & File Support

**Goal:** Enable the three AG API inferencers to accept `MultimodalContent` alongside existing string inputs.

These inferencers share the same pattern: they call `_get_messages(prompt_or_messages)` in their respective backend modules. The key integration points are:

#### 2A. AG Claude API Inferencer

**Files to modify:**
- `src/agent_foundation/apis/ag/ai_gateway_claude_llm.py` — `_get_messages()` function
- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_claude_api_inferencer.py`

**Changes to `_get_messages()`** (lines 155–207 of `ai_gateway_claude_llm.py`):
- Add a branch: if `prompt_or_messages` is `MultimodalContent`, call `to_anthropic_content_blocks()` to build content blocks, then wrap in a user message dict.
- The existing `_build_request_payload()` already forwards `messages` (list of dicts with `content` as list of blocks) directly to the API — no further changes needed there.

**Changes to `AgClaudeApiInferencer`:**
- In `_infer()`, `_ainfer()`, `_ainfer_streaming()`: detect `MultimodalContent` input and convert before passing to backend. Currently, `_infer` accepts `inference_input: str` — widen to `Any` and handle `MultimodalContent` → message conversion early.
- Alternative (cleaner): Override `_render_prompt()` to detect `MultimodalContent` and produce the message list, then use the existing `set_messages()` flow. This keeps the backend functions unchanged.

**Recommended approach:** Override `_render_prompt()` in `AgClaudeApiInferencer` to handle `MultimodalContent`:
```python
def _render_prompt(self, inference_input, extra_feed=None):
    if isinstance(inference_input, MultimodalContent):
        content_blocks = to_anthropic_content_blocks(inference_input)
        self.set_messages([{"role": "user", "content": content_blocks}])
        return inference_input.text  # return text for logging/caching
    return super()._render_prompt(inference_input, extra_feed)
```

#### 2B. AG OpenAI API Inferencer

**Files to modify:**
- `src/agent_foundation/apis/ag/ai_gateway_openai_llm.py` — `_get_messages()` function
- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_openai_api_inferencer.py`

**Changes to `_get_messages()`** (lines 105–140 of `ai_gateway_openai_llm.py`):
- Add branch for `MultimodalContent` → use `to_openai_content_parts()` to build content parts, wrap in user message.

**Changes to `AgOpenAIApiInferencer`:**
- Same pattern as Claude: override `_render_prompt()` to detect `MultimodalContent`, call `to_openai_content_parts()`, use `set_messages()`.

#### 2C. AG Gemini API Inferencer

**Files to modify:**
- `src/agent_foundation/apis/ag/ai_gateway_gemini_llm.py` — `_get_messages()` function
- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_gemini_api_inferencer.py`

**Changes:** Mirror the OpenAI pattern (Gemini uses OpenAI-compatible chat completions on the AG). Same `to_openai_content_parts()` conversion.

**Acceptance Criteria:**
- Each AG inferencer can accept `MultimodalContent(text="Describe this image", images=[ImageContent(source="/path/to/img.png")])` and produce the correct API payload.
- Existing `str` and `dict` inputs work unchanged.
- Unit test: mock the gateway HTTP call, verify the request payload contains the image block.

---

### Phase 3: Claude Code Inferencers — Image & File Support

**Goal:** Enable `ClaudeCodeCliInferencer` and `ClaudeCodeSdkInferencer` to pass images/files to the Claude Code agent.

#### 3A. ClaudeCodeCliInferencer

**File to modify:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_cli_inferencer.py`

**Mechanism:** The Claude Code CLI supports `--input-format stream-json` for structured input. When images need to be passed, the inferencer should:

1. Add an `images` attribute (list of image paths or `ImageContent` objects).
2. When `MultimodalContent` is detected as input (or `images` is non-empty):
   - Switch to `--input-format stream-json --output-format stream-json` mode.
   - Send the prompt as a stream-json message with image content blocks via stdin.
   - The stream-json input format expects JSON messages on stdin, each on its own line.

**Alternative (simpler) approach:** Since Claude Code agents have local file access (`has_local_access=True`), the simplest path is to:
- Embed image file paths in the prompt text so Claude Code reads them via its Read tool.
- For non-local images (inline bytes), write them to a temp file in the workspace, then reference the path.

However, this is less elegant than native multimodal input. The **recommended approach** is the stream-json mechanism for proper multimodal support when the CLI supports it, with the file-path-in-prompt fallback for reliability.

**Changes to `construct_command()`:**
- When `MultimodalContent` input is detected, add `--input-format stream-json` flag.
- Adjust stdin writing in `_ainfer_streaming()` to send structured JSON with image blocks.

**Changes to `_ainfer_streaming()`:**
- Detect `MultimodalContent` input.
- Build stream-json message payload with text and image content blocks.
- Write JSON message to stdin instead of plain text.

**New attribute:**
```python
images: Optional[List[str]] = attrib(default=None)  # Convenience: image file paths
```

#### 3B. ClaudeCodeSdkInferencer

**File to modify:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_sdk_inferencer.py`

**Mechanism:** The Claude Agent SDK's `client.query()` method. We need to check if the SDK accepts structured content (content blocks with images) in `query()`. Based on research, the SDK may accept:
- A plain string (current usage).
- A list of content blocks (similar to the Anthropic Messages API).

**Changes to `_ainfer_streaming()`:**
- Detect `MultimodalContent` input.
- If images are present, build content blocks and pass to `client.query()` as a structured content list (if supported by the SDK).
- Fallback: write images to workspace temp files and reference paths in the prompt.

**New attribute:**
```python
images: Optional[List[str]] = attrib(default=None)  # Convenience: image file paths
```

**Note:** The SDK's `query()` method signature needs runtime verification. If it only accepts strings, the file-path fallback is the only viable option without SDK changes.

---

### Phase 4: Codex Inferencers — Image & File Support

**Goal:** Enable `CodexCliInferencer` and `CodexSdkInferencer` to pass images to the Codex agent.

#### 4A. CodexCliInferencer

**File to modify:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_cli_inferencer.py`

**Mechanism:** The Codex CLI natively supports the `--image` / `-i` flag for attaching images. This is the cleanest integration path.

**Changes to `construct_command()`:**
- Accept `images` parameter (list of file paths).
- When `MultimodalContent` input is detected, extract image paths and add `--image <path>` flags to the command.
- For inline image data (bytes), write to temp file first.

**Changes to class:**
```python
images: Optional[List[str]] = attrib(default=None)  # Image file paths for --image flag
```

**Changes to `construct_command()`** (around line 166):
- Add image flag construction after prompt placement:
  ```python
  for img_path in (images or []):
      parts.append(f'--image "{self._escape_for_shell(img_path)}"')
  ```

**Note:** The `--image` flag must come before the prompt positional argument. Adjust ordering in `construct_command()` accordingly.

#### 4B. CodexSdkInferencer

**File to modify:** `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_sdk_inferencer.py`

**Mechanism:** The `openai_codex` SDK's `thread.turn()` may accept structured content or image attachments. Research indicates the SDK supports images via the thread API.

**Changes to `_ainfer_streaming()`:**
- Detect `MultimodalContent` input.
- If the SDK supports image attachments in `thread.turn()`, pass them natively.
- Fallback: write images to workspace temp files and reference paths.

---

### Phase 5: Input Preprocessing Integration

**Goal:** Wire `MultimodalContent` through the inferencer pipeline so it survives template rendering and preprocessing.

**Files to modify:**
- `src/agent_foundation/common/inferencers/inferencer_base.py` — `_infer_single()` / `_ainfer_single()` methods
- `src/agent_foundation/common/inferencers/templated_inferencer_base.py` — `_render_prompt()` method

**Key concern:** The inferencer pipeline (in `_infer_single` / `_ainfer_single`) does:
1. `input_preprocessor(inference_input)` — may stringify
2. `_render_prompt(inference_input)` — Jinja2 rendering expects string

We must ensure `MultimodalContent` survives this pipeline. Two approaches:

**Approach A (Recommended):** Override `_render_prompt()` in multimodal-capable inferencers (as shown in Phase 2). When the input is `MultimodalContent`, render only the `.text` field through Jinja2 but preserve the full object. This is localized and low-risk.

**Approach B:** Modify `_render_prompt()` in `TemplatedInferencerBase` to detect `MultimodalContent` and render `.text` while preserving images/files. This is more general but higher blast radius.

**Recommended:** Approach A — each multimodal-capable inferencer handles its own `_render_prompt()` override. This follows the existing pattern where `AgClaudeApiInferencer` already uses `set_messages()` for structured input.

**Acceptance Criteria:**
- `MultimodalContent` passes through `_render_prompt()` with `.text` rendered via Jinja2 and images preserved.
- Template variables are applied to the text portion.

---

## 3. Risk Register

| ID | Risk | Severity | Failure Mode | Mitigation |
|---|---|---|---|---|
| R1 | Claude Code CLI `--input-format stream-json` message format is undocumented | 🟡 Med | Image blocks silently ignored or cause errors | Test empirically; fall back to file-path-in-prompt strategy for Claude Code. The `stream-json` format is documented as a GitHub issue (#24594). |
| R2 | Claude Agent SDK `query()` may not accept structured content blocks | 🟡 Med | SDK rejects non-string input | Runtime test at init time; fall back to writing images to temp files + path reference in prompt. Claude Code has `has_local_access=True` so it can read local files. |
| R3 | Codex SDK `thread.turn()` may not support image attachments | 🟡 Med | SDK rejects image input | Fall back to writing images to temp files. Codex has `has_local_access=True`. |
| R4 | Large images (>5MB) cause timeouts or memory issues | 🟢 Low | Slow requests, OOM | Add validation in `encode_image_base64()`: warn if >5MB, hard-reject if >20MB. Compress or resize before encoding. |
| R5 | AG Gateway rejects multimodal payloads | 🟡 Med | Gateway returns 4xx | The AG routes to Bedrock (Claude) and OpenAI endpoints which both support multimodal. Test with a real request. AG's `ClaudeRequest` model must support image content blocks — verify with the AI Gateway SDK's `bedrock.chat.ClaudeRequest` schema. |
| R6 | Template rendering breaks with `MultimodalContent` | 🟢 Low | Jinja2 errors on non-string input | The recommended approach (override `_render_prompt` per inferencer) isolates the `.text` field for rendering. Existing tests cover string inputs. |
| R7 | Backward compatibility — existing callers break | 🔴 High | Regression for all current users | All changes are additive: `str` inputs take existing code paths. `MultimodalContent` is a new opt-in type. Type checking (`isinstance`) gates new behavior. |
| R8 | File content not accessible by agentic backends | 🟢 Low | Agent can't read referenced files | For agentic inferencers with `has_local_access=True`, files are accessible if in the workspace. For remote APIs, inline the content via base64. |

---

## 4. Files to Create / Modify

### New Files

| File | Purpose |
|---|---|
| `src/agent_foundation/common/inferencers/multimodal_content.py` | `MultimodalContent`, `ImageContent`, `FileContent` data classes + serialization utilities (`encode_image_base64`, `to_anthropic_content_blocks`, `to_openai_content_parts`) |
| `tests/unit/common/inferencers/test_multimodal_content.py` | Unit tests for the multimodal content model and serializers |
| `tests/unit/common/inferencers/api_inferencers/ag/test_ag_multimodal.py` | Integration tests for AG inferencers with multimodal input (mocked HTTP) |

### Modified Files

| File | Changes |
|---|---|
| **AG Claude API** | |
| `src/agent_foundation/apis/ag/ai_gateway_claude_llm.py` | Add `MultimodalContent` branch to `_get_messages()` (lines ~155–207) |
| `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_claude_api_inferencer.py` | Override `_render_prompt()` or widen `_infer`/`_ainfer`/`_ainfer_streaming` input type to handle `MultimodalContent` → `set_messages()` |
| **AG OpenAI API** | |
| `src/agent_foundation/apis/ag/ai_gateway_openai_llm.py` | Add `MultimodalContent` branch to `_get_messages()` (lines ~105–140) |
| `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_openai_api_inferencer.py` | Override `_render_prompt()` to handle `MultimodalContent` → `set_messages()` |
| **AG Gemini API** | |
| `src/agent_foundation/apis/ag/ai_gateway_gemini_llm.py` | Add `MultimodalContent` branch to `_get_messages()` |
| `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_gemini_api_inferencer.py` | Override `_render_prompt()` to handle `MultimodalContent` → `set_messages()` |
| **Claude Code CLI** | |
| `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_cli_inferencer.py` | Add `images: Optional[List[str]]` attribute; modify `construct_command()` to handle `MultimodalContent`; modify `_ainfer_streaming()` stdin writing for stream-json image input |
| **Claude Code SDK** | |
| `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_sdk_inferencer.py` | Add `images: Optional[List[str]]` attribute; modify `_ainfer_streaming()` to pass structured content to `client.query()` or fall back to file-path strategy |
| **Codex CLI** | |
| `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_cli_inferencer.py` | Add `images: Optional[List[str]]` attribute; modify `construct_command()` to add `--image` flags |
| **Codex SDK** | |
| `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_sdk_inferencer.py` | Add `images: Optional[List[str]]` attribute; modify `_ainfer_streaming()` to pass images via SDK if supported |
| **Common** | |
| `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/common.py` | No changes needed (model tag resolution is unrelated) |

---

## 5. Validation & Testing Strategy

### Unit Tests

1. **`test_multimodal_content.py`**
   - `ImageContent` from file path: reads file, auto-detects MIME type.
   - `ImageContent` from inline bytes: preserves data and media_type.
   - `encode_image_base64`: correct base64 encoding for PNG, JPEG, GIF, WebP.
   - `to_anthropic_content_blocks`: correct Anthropic content block format with images + text.
   - `to_openai_content_parts`: correct OpenAI content parts format with image_url + text.
   - `MultimodalContent` with no images/files: produces text-only blocks.
   - `is_multimodal()` type check.

2. **`test_ag_multimodal.py`** (per AG inferencer)
   - Mock the HTTP transport (httpx or AI Gateway SDK).
   - Verify request payload contains correct image content blocks.
   - Verify backward compatibility: `str` input produces text-only payload.

3. **Existing test suites** — Run all existing inferencer tests to verify no regressions.

### Integration Tests

4. **AG Claude API + real image:** Send a small test image to AG Claude endpoint, verify non-error response with image description. (Requires AG access; can be run in CI with proper auth.)

5. **Claude Code CLI + image:** Run `ClaudeCodeCliInferencer` with a test image in the workspace. Verify the agent can "see" and describe the image.

6. **Codex CLI + image:** Run `CodexCliInferencer` with `--image` flag pointing to a test image. Verify the agent describes the image.

### Manual Validation

7. **End-to-end smoke test:** Use each inferencer to describe a sample PNG image. Verify the response is contextually correct (not generic text).

### Test Data

- Create `tests/fixtures/test_image.png` — a simple 100x100 colored square (generated programmatically, no external dependency).
- Create `tests/fixtures/test_image.jpg` — same image in JPEG format.

---

## 6. Detailed Design Notes

### 6.1 Multimodal Content Model Design

The `MultimodalContent` class is designed as a **value object** (attrs `@attrs(slots=True)`):

```python
@attrs(slots=True)
class ImageContent:
    source: str = attrib()                              # File path, URL, or "inline"
    data: Optional[bytes] = attrib(default=None)        # Raw bytes for inline
    media_type: Optional[str] = attrib(default=None)    # MIME type
    detail: Optional[str] = attrib(default=None)        # OpenAI detail level

@attrs(slots=True)
class FileContent:
    path: str = attrib()
    content: Optional[str] = attrib(default=None)
    media_type: Optional[str] = attrib(default=None)

@attrs(slots=True)
class MultimodalContent:
    text: str = attrib()
    images: List[ImageContent] = attrib(factory=list)
    files: List[FileContent] = attrib(factory=list)
```

**Why attrs over dataclass?** Consistency with the rest of the codebase — every inferencer and data model uses `@attrs`.

### 6.2 Anthropic Messages API Image Format

```json
{
  "role": "user",
  "content": [
    {
      "type": "image",
      "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": "<base64-encoded-data>"
      }
    },
    {
      "type": "text",
      "text": "Describe this image."
    }
  ]
}
```

### 6.3 OpenAI Chat Completions Image Format

```json
{
  "role": "user",
  "content": [
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,<base64-encoded-data>",
        "detail": "auto"
      }
    },
    {
      "type": "text",
      "text": "Describe this image."
    }
  ]
}
```

### 6.4 Claude Code CLI Stream-JSON Image Format

Based on the Anthropic Messages API format (Claude Code uses the same content block schema):

```json
{"type": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}, {"type": "text", "text": "Describe this image."}]}
```

Sent as a single line to stdin with `--input-format stream-json`.

### 6.5 Codex CLI Image Flag

```bash
codex exec --json -s workspace-write --image /path/to/image.png "Describe this image"
```

Multiple images: repeat the flag:
```bash
codex exec --json --image img1.png --image img2.png "Compare these images"
```

### 6.6 File Content Strategy

For **text files** passed as `FileContent`:
- **AG API inferencers:** Inline the file content as a text block in the message. No special encoding needed.
- **Agentic inferencers (Claude Code, Codex):** These have local file access. If the file is in the workspace, reference the path in the prompt. If not, copy to a temp location in the workspace.

For **binary/PDF files** passed as `FileContent`:
- **AG Claude API:** Anthropic's API supports PDF via `document` content blocks (base64-encoded). Add `to_anthropic_document_block()` to the serialization utilities.
- **AG OpenAI API:** OpenAI doesn't natively support PDF in chat completions. Fall back to text extraction or skip with a warning.
- **Agentic inferencers:** Reference the file path; the agent can use its tools to read it.

### 6.7 Convenience API

For callers who don't want to construct `MultimodalContent` objects, each inferencer can accept image paths directly:

```python
# Via attribute (applied to all calls)
inferencer = AgClaudeApiInferencer(images=["/path/to/img.png"])
result = inferencer("Describe this image")

# Via kwargs (per-call)
result = inferencer("Describe this image", images=["/path/to/img.png"])

# Via MultimodalContent (full control)
mc = MultimodalContent(
    text="Describe this image",
    images=[ImageContent(source="/path/to/img.png")],
)
result = inferencer(mc)
```

The attribute-based and kwargs-based conveniences are syntactic sugar that internally construct `MultimodalContent`.

---

## 7. Implementation Order

```
Phase 1: MultimodalContent model (0 dependencies)
    ↓
Phase 2: AG API inferencers (depends on Phase 1)
    ├─ 2A: AG Claude
    ├─ 2B: AG OpenAI     (parallel with 2A)
    └─ 2C: AG Gemini     (parallel with 2A)
    ↓
Phase 3: Claude Code inferencers (depends on Phase 1)
    ├─ 3A: CLI inferencer
    └─ 3B: SDK inferencer (parallel with 3A)
    ↓
Phase 4: Codex inferencers (depends on Phase 1)
    ├─ 4A: CLI inferencer
    └─ 4B: SDK inferencer (parallel with 4A)
    ↓
Phase 5: Pipeline integration (depends on Phases 2-4)
```

Phases 2, 3, and 4 are independent and can proceed in parallel after Phase 1 is complete.

---

## 8. Self-Validation Checklist

- [x] All 7 inferencers addressed (2 Claude Code + 2 Codex + 3 AG API)
- [x] Shared content model avoids duplication
- [x] Backward compatibility preserved (string inputs unchanged)
- [x] Each backend uses its native mechanism (CLI flags, SDK methods, API formats)
- [x] Risk register covers SDK compatibility unknowns
- [x] File paths verified against actual codebase structure
- [x] Testing strategy covers unit, integration, and manual validation
- [x] No over-engineering: content model + per-backend serializers, nothing more
- [x] Leverages existing patterns (`set_messages()`, `has_local_access`, attrs conventions)
