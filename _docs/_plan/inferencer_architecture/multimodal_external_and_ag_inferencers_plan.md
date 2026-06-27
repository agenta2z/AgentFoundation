# Plan: Multimodal Images and Files for Claude Code, Codex, and AI Gateway Inferencers

Date: 2026-06-26

Chosen artifact path: `_docs/_plan/inferencer_architecture/multimodal_external_and_ag_inferencers_plan.md`.

The original artifact output path was empty, so this plan is stored with the existing inferencer architecture plans.

## Scope

Implement first-class image and file input support for these five target inferencer surfaces:

- `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_cli_inferencer.py`
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_sdk_inferencer.py`
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_cli_inferencer.py`
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_sdk_inferencer.py`
- The three AI Gateway API inferencer wrappers under `src/agent_foundation/common/inferencers/api_inferencers/ag/`:
  - `ag_claude_api_inferencer.py`
  - `ag_openai_api_inferencer.py`
  - `ag_gemini_api_inferencer.py`

This also requires updates to the AI Gateway request builders in `src/agent_foundation/apis/ag/`, because the wrappers delegate payload construction there.

Non-goals for the first implementation:

- Do not redesign conversational inbox/server schemas, which are currently text-oriented.
- Do not repurpose `AgentAttachment`; it is a prompt-rendering abstraction, not a MIME/path/media abstraction.
- Do not promise "any binary file works everywhere." Provider and gateway support differs by media type and route.
- Do not inline arbitrary large binaries in prompts or logs.

## Evidence-Based Findings

### Current Target Inferencers Are Text-First

The Claude Code and Codex inferencers accept `Any` in some places, but operationally they extract text prompts:

- `ClaudeCodeCliInferencer` is defined at `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_cli_inferencer.py:34`.
  - `construct_command()` starts at line 233.
  - Streaming starts at `_ainfer_streaming()` line 450.
  - Sync `_infer()` starts at line 612.
  - It has `large_input_mode: LargeInputMode = STDIN` at line 98, so text prompt delivery already avoids command-line argument limits.
- `ClaudeCodeSdkInferencer` is defined at `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_sdk_inferencer.py:33`.
  - Streaming starts at line 265.
  - It calls `client.query(prompt)` at line 303.
  - Sync `_infer()` starts at line 363.
- `CodexCliInferencer` is defined at `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_cli_inferencer.py:53`.
  - `construct_command()` starts at line 166.
  - Streaming starts at line 345.
  - It uses `large_input_mode == STDIN` at line 359.
  - Sync `_infer()` starts at line 480.
- `CodexSdkInferencer` is defined at `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_sdk_inferencer.py:71`.
  - Streaming starts at line 254.
  - It calls `thread.turn(prompt)` at line 276.
  - Sync `_infer()` starts at line 332.

There is no target-local `attachments`, `images`, `files`, `content parts`, or equivalent first-class media model.

### AI Gateway Wrappers Delegate All Payload Semantics

The three AG wrappers primarily configure defaults and call helper modules:

- `AgClaudeApiInferencer`
  - class at `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_claude_api_inferencer.py:111`
  - `_messages_override` at line 154
  - `set_messages()` at line 168
  - `_infer()` at line 191, `_ainfer()` at line 204, `_ainfer_streaming()` at line 221
- `AgOpenAIApiInferencer`
  - class at `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_openai_api_inferencer.py:69`
  - `_messages_override` at line 108
  - `set_messages()` at line 122
  - `_infer()` at line 145, `_ainfer()` at line 157, `_ainfer_streaming()` at line 173
- `AgGeminiApiInferencer`
  - class at `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_gemini_api_inferencer.py:61`
  - `_messages_override` at line 95
  - `set_messages()` at line 108
  - `_infer()` at line 125, `_ainfer()` at line 136, `_ainfer_streaming()` at line 151

Therefore, wrapper changes alone are insufficient. Multimodal support must be added where messages and payloads are constructed:

- Claude AI Gateway backend:
  - `_get_messages()` at `src/agent_foundation/apis/ag/ai_gateway_claude_llm.py:155`
  - `_build_request_payload()` at line 472
  - `generate_text()` at line 510
  - `generate_text_streaming()` at line 802
  - `generate_text_async()` at line 972
- OpenAI AI Gateway backend:
  - `_get_messages()` at `src/agent_foundation/apis/ag/ai_gateway_openai_llm.py:105`
  - `_build_request_payload()` at line 162
  - `generate_text()` at line 390
  - `generate_text_streaming()` at line 496
  - `generate_text_async()` at line 620
- Gemini AI Gateway backend:
  - `_get_messages()` at `src/agent_foundation/apis/ag/ai_gateway_gemini_llm.py:93`
  - `_build_request_payload()` at line 140
  - `generate_text()` at line 360
  - `generate_text_streaming()` at line 445
  - `generate_text_async()` at line 538

Important nuance: the AG modules already accept raw dict/list messages. Claude's `_get_messages()` converts string input to Anthropic-style text content blocks and preserves dict message input. OpenAI and Gemini normalize to OpenAI-style chat message dicts and also preserve dict/list messages. The existing `set_messages()` escape hatch can already pass structured provider-shaped content in some cases, but it provides no normalization, MIME handling, file reading, size checks, redaction, or parity across sync/async/streaming.

### Existing "File Path as Prompt" Is Not Attachment Support

In the AG backends, a string that happens to be a local file path is read as UTF-8 prompt text:

- Claude uses `path.exists()` and reads the file at `ai_gateway_claude_llm.py:160-163`.
- OpenAI uses `path.isfile()` and reads the file at `ai_gateway_openai_llm.py:117-123`.
- Gemini uses `path.isfile()` and reads the file at `ai_gateway_gemini_llm.py:97-103`.

That behavior is useful but semantically different from attaching an image, PDF, or other file. The implementation should preserve this legacy behavior for text-only callers while adding explicit attachment input.

### Shared Base Classes Offer Useful Hooks but Also a Cache Risk

Relevant shared infrastructure:

- `InferencerBase` is defined at `src/agent_foundation/common/inferencers/inferencer_base.py:69`.
- `has_local_access` is declared at line 161.
- `additional_allowed_paths` is declared at line 186.
- `target_path` is declared at line 250.
- `effective_cwd` starts at line 377 and documents priority `target_path > workspace.root > os.getcwd()`.
- `StreamingInferencerBase` is defined at `src/agent_foundation/common/inferencers/streaming_inferencer_base.py:119`.
- `_extract_prompt()` at `streaming_inferencer_base.py:564` handles only strings and dicts with a `prompt` key.
- `LargeInputMode` is defined at `src/agent_foundation/common/inferencers/terminal_inferencers/terminal_session_inferencer_base.py:32`.

The local external inferencers already set `has_local_access=True`:

- Claude Code CLI: `claude_code_cli_inferencer.py:90`
- Claude Code SDK: `claude_code_sdk_inferencer.py:151`
- Codex CLI: `codex_cli_inferencer.py:68`
- Codex SDK: `codex_sdk_inferencer.py:83`

This supports a clean path-reference strategy for local agentic tools, but the streaming cache key currently depends on prompt extraction only. If two requests share the same prompt but attach different files, caching can return an incorrect stream unless the attachment fingerprint is included in the cache identity or caching is disabled for multimodal requests.

### No Existing Inferencer-Level Multimodal Model

No shared `Attachment`, `ImageAttachment`, `FileAttachment`, or `ContentPart` model exists in `src/agent_foundation/common/inferencers`.

The existing attachment type is `AgentAttachment` at `src/agent_foundation/agents/agent_attachment.py:33`. It has `id`, `description`, `content`, and `formatter`, and renders XML-ish prompt text through `.text` or `.full_text`. It has no MIME type, source path, URL, byte payload, base64 handling, media kind, or provider serialization semantics. It should not be reused as the new media model, though an adapter from `AgentAttachment` to text content can preserve compatibility later.

Adjacent systems are also text-oriented:

- Conversational `UserMessage.content` is string-based in `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/inbox.py:21`.
- Server `ChatMessage` is defined in `src/agent_foundation/server/schema.py:65`, with text content assumptions.

These are outside the requested first implementation, but they matter for future rollout.

### Tests and Dependency Clues

Existing test coverage is uneven:

- Claude Code has a dedicated test folder: `test/agent_foundation/common/inferencers/external/claude_code/`.
- Codex has no parallel dedicated test folder under `test/agent_foundation/common/inferencers/external/codex/`.
- AG gateway tests currently live under `test/agent_foundation/apis/ag/`.
- Existing large input tests are text-delivery tests:
  - `test/agent_foundation/common/inferencers/test_large_input_mode.py`
  - `test/agent_foundation/common/inferencers/test_large_arg_offload.py`

Dependency/config findings:

- The repo appears Buck-driven for this source tree. `src/BUCK` defines the `agent_foundation` Python library and includes `attrs`, `httpx`, `pydantic`, `requests`, `pyyaml`, and `claude-agent-sdk`.
- `src/BUCK` does not list an obvious image/MIME dependency such as Pillow, `python-magic`, or `filetype`.
- `src/BUCK` also does not list `openai-codex`, even though `CodexSdkInferencer` soft-imports Codex SDK types. Per the request's dependency note, this is not proof that the dependency is unavailable; it means implementation should verify runtime and build packaging before relying on SDK-native media APIs.

### Provider Documentation Findings Checked on 2026-06-26

Provider docs are volatile, so implementation should re-check them before coding. The current plan is based on these checked sources:

- OpenAI Images and Vision guide: `https://developers.openai.com/api/docs/guides/images-vision`
  - Chat Completions image inputs use message content arrays with `{"type": "text"}` and `{"type": "image_url", "image_url": {"url": ...}}`.
  - The URL can be a `data:image/...;base64,...` data URL.
- OpenAI File Inputs guide: `https://developers.openai.com/api/docs/guides/file-inputs`
  - Chat Completions can use `{"type": "file", "file": {"filename": ..., "file_data": ...}}` for supported file inputs.
  - Responses API has `input_file`, but the current AG OpenAI code builds Chat Completions-style payloads, so do not accidentally introduce a Responses API shape without a larger API migration.
- Anthropic Claude Vision guide: `https://platform.claude.com/docs/en/build-with-claude/vision`
  - Image blocks use `{"type": "image", "source": {"type": "base64", "media_type": ..., "data": ...}}`.
  - Size limits vary by route; direct API documentation includes a 10 MB image limit, while Bedrock/Google Cloud paths are more constrained.
- Anthropic Claude PDF support: `https://platform.claude.com/docs/en/build-with-claude/pdf-support`
  - PDF/document blocks use `{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": ...}}` where supported.
- Claude Code CLI reference: `https://code.claude.com/docs/en/cli-reference`
  - `claude -p` print mode supports stdin prompt text and JSON/stream-json output.
  - No native image/file flag was found in the checked CLI reference.
- Claude Agent SDK overview: `https://code.claude.com/docs/en/agent-sdk/overview`
  - SDK examples use `query(prompt=...)`.
  - Agents can use tools such as Read, Write, Edit, Bash, Glob, and Grep, which supports a staged-file/path-reference strategy.
- Codex CLI reference: `https://developers.openai.com/codex/cli/reference`
  - `codex exec` accepts a prompt string or `-` for stdin.
  - The reference also documents image attachment support on interactive/resume-style commands, including `--image, -i path[,path...]` in the checked page. The installed CLI must be probed before assuming exact flag support for fresh `codex exec`.
- Codex SDK docs: `https://developers.openai.com/codex/sdk`
  - The checked docs did not establish a direct typed image/file payload API matching the local `thread.turn(prompt)` call.
- Gemini OpenAI compatibility docs: `https://ai.google.dev/gemini-api/docs/openai`
  - Gemini can be accessed through OpenAI-compatible chat APIs by changing base URL.
  - The checked page confirms OpenAI-compatible chat and streaming patterns; gateway acceptance of image/file blocks still needs implementation-time tests.

## High-Level Approach

### 1. Add a Provider-Neutral Multimodal Input Contract

Introduce a small shared model under `src/agent_foundation/common/inferencers/` for normalized multimodal input. Keep it independent of any one provider.

Recommended model concepts:

- `MultimodalInput`
  - `prompt: str`
  - optional `messages: list[Message]`
  - `attachments: list[Attachment]`
- `Attachment`
  - `kind`: `image`, `file`, or `text`
  - `source`: local path, URL, data URI, bytes, or already-base64 payload
  - `mime_type`
  - `filename`
  - optional `description`
  - computed metadata: byte size, SHA-256 fingerprint, extension, safe display name
- `ContentPart`
  - `TextPart`
  - `ImagePart`
  - `FilePart`

The public caller-facing forms should be ergonomic and backward compatible:

- Existing `infer("prompt")` remains unchanged.
- Existing `infer({"prompt": "..."})` remains unchanged.
- New explicit attachment forms are supported:
  - `infer("describe this", attachments=[...])`
  - `infer({"prompt": "describe this", "attachments": [...]})`
  - `infer(MultimodalInput(...))`
- Existing raw provider-shaped message lists/dicts remain accepted, especially for `set_messages()`.

Use the Python standard library for the first implementation:

- `mimetypes` for MIME guessing
- `base64` for provider payload encoding
- `hashlib` for fingerprints
- `pathlib` for path normalization

Do not add Pillow or file-type detection dependencies unless a later requirement needs image resizing or magic-byte validation. The repository already lacks those packages in `src/BUCK`, and the initial problem does not require transformation/compression.

### 2. Keep Normalization Separate From Provider Serialization

Do not duplicate media parsing in each inferencer. The proper layering is:

1. Normalize user input into a provider-neutral `MultimodalInput`.
2. Validate size, source, MIME type, and allowed path rules once.
3. Serialize into provider-specific payloads:
   - Anthropic/Claude content blocks
   - OpenAI Chat Completions content arrays
   - Gemini OpenAI-compatible content arrays
   - Local agent path-reference manifests for Claude Code/Codex CLI/SDK
4. Keep raw provider-shaped messages as an escape hatch.

This keeps the surface extensible without coupling local CLI tools to cloud API payload formats.

### 3. Treat "Files" by Capability, Not by Marketing Label

File support must be precise:

- Images:
  - API inferencers should use provider-native multimodal blocks where the route supports them.
  - Local Claude Code/Codex inferencers should use native CLI image flags only when verified; otherwise stage/reference the image file and instruct the local agent to inspect it if its tools support that workflow.
- PDFs:
  - Claude API can use document blocks where route/mode supports them.
  - OpenAI Chat Completions can use file blocks where supported by the gateway.
  - Gemini/OpenAI-compatible handling must be gated by gateway tests.
- Text files:
  - API inferencers may inline text content up to configured limits.
  - Local agentic inferencers should prefer path references to avoid bloating prompts.
- Other binary files:
  - Local agentic inferencers can pass path references if the agent has filesystem access.
  - API inferencers should fail clearly unless that provider/route explicitly supports the MIME type.

This avoids false success where a binary is technically embedded but the model or gateway cannot use it.

### 4. Local Agentic Tools Should Prefer Staged Path References

Claude Code and Codex are local agentic inferencers with `has_local_access=True` and `effective_cwd`. For these tools, the clean default is:

- If an attached local file is already under `effective_cwd`, reference it by a stable relative path.
- If it is outside `effective_cwd` but allowed and readable, copy it into an attachment staging directory under `effective_cwd`, for example `.agent_foundation/attachments/<sha256>/<filename>`.
- Add a compact attachment manifest to the prompt:
  - filename
  - relative staged path
  - MIME type
  - size
  - optional description
- Do not inline base64 into the prompt.
- Do not silently follow symlinks outside allowed roots.

Native attachment flags can be layered on top only when verified:

- Codex CLI may support `--image` for some command forms. Probe the installed CLI and add command-construction tests before using it.
- Claude Code CLI docs checked for this plan did not show a native image/file flag. Do not invent one.
- Claude Code SDK and Codex SDK should keep the path-reference strategy unless the installed SDK exposes and tests a typed media input API.

### 5. API Inferencers Should Use Provider-Native Payloads With Gateway Gating

The three AG inferencers should produce provider-native structured messages:

- Claude:
  - Text: `{"type": "text", "text": "..."}`
  - Image: Anthropic image block with base64 `source`
  - PDF/document: Anthropic document block where the selected gateway mode supports it
- OpenAI:
  - Text: `{"type": "text", "text": "..."}`
  - Image: `{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}`
  - Supported files: Chat Completions `{"type": "file", "file": {"filename": ..., "file_data": ...}}`
- Gemini:
  - Start with OpenAI-compatible content arrays because the local AG Gemini module builds OpenAI-style chat payloads.
  - Gate image/file support behind tests against the AI Gateway route; if the gateway rejects these blocks, fail clearly or feature-gate the path rather than silently flattening binaries to text.

### 6. Preserve Backward Compatibility

No-attachment text requests must produce the same command and payload shapes as before, except for harmless internal refactoring.

Backwards compatibility requirements:

- `infer("hello")` unchanged.
- `infer({"prompt": "hello"})` unchanged.
- Existing raw dict/list message callers unchanged.
- `set_messages()` unchanged as a raw override.
- Existing file-path-as-prompt behavior in AG backends remains for text files, unless an explicit `attachments=` argument is used.
- Existing streaming response parsing and terminal response metadata remain unchanged.

### 7. Redaction and Cache Safety Are First-Class Requirements

Attachment support creates new failure modes. The implementation must:

- Never log raw base64 or raw bytes.
- Include only sanitized metadata in debug logs.
- Include attachment fingerprints in cache identity where streaming caches apply.
- Reject or cap oversized inputs before provider calls.
- Keep generated staging files under a deterministic ignored runtime location.
- Avoid writing attachment bytes into checkpoint/history artifacts unless explicitly intended.

## Key Execution and Implementation Steps

### Phase 0: Confirm Runtime Capabilities and Lock the Contract

Tasks:

- Re-check provider documentation immediately before implementation.
- Inspect installed CLI/SDK help locally:
  - `claude --help`
  - `claude -p --help`
  - `codex exec --help`
  - `codex exec resume --help`
- Inspect installed Python SDK capabilities without assuming docs:
  - `claude_agent_sdk` typed input support beyond `query(prompt=...)`
  - `openai-codex` availability and whether `thread.turn()` accepts typed content
- Record supported media matrix:
  - provider
  - inferencer surface
  - image path
  - PDF/file path
  - arbitrary binary behavior
  - size limits
  - unsupported-mode behavior

Acceptance criteria:

- A small capability note exists, either in this plan as an update or a nearby implementation note.
- The implementation contract explicitly distinguishes `image`, `text file`, `PDF`, and `other binary`.
- No code path depends on undocumented CLI flags without a local help/probe test.
- The executor confirms whether `openai-codex` needs build/dependency metadata updates rather than dismissing it based only on absence from `src/BUCK`.

### Phase 1: Add Shared Multimodal Input and Validation Utilities

Create `src/agent_foundation/common/inferencers/multimodal_input.py`.

Responsibilities:

- Define the neutral data structures:
  - attachment kind
  - source type
  - content part/message shapes if needed
  - normalized request object
- Parse supported input forms:
  - string prompt
  - `{"prompt": ...}`
  - `{"prompt": ..., "attachments": ...}`
  - raw message list/dict where appropriate
  - explicit `attachments=` keyword argument
- Resolve local paths against:
  - explicit absolute paths
  - `effective_cwd`
  - optional allowed roots
- Infer MIME type using `mimetypes`, with caller override.
- Compute safe metadata:
  - filename
  - size
  - SHA-256
  - display-safe redacted representation
- Convert local files to:
  - bytes/base64 for API payloads
  - staged relative path references for local agents
- Enforce provider-agnostic validation:
  - path exists
  - is a regular file
  - readable
  - size under configured global cap
  - explicit unsupported type error for binary API files when no provider supports them

Avoid:

- Global behavior changes in `StreamingInferencerBase._extract_prompt()` unless needed for a clean cache hook.
- Reading huge files just to compute logs.
- Storing raw bytes in long-lived attrs by default.

Acceptance criteria:

- Unit tests cover string, dict, structured object, and keyword attachment input.
- Unit tests cover non-existent path, directory path, unreadable path where feasible, missing MIME, explicit MIME override, URL/data URI forms, and oversized file rejection.
- Sanitized `repr`/debug rendering never includes base64 or raw bytes.
- Text-only inputs produce the same normalized prompt as existing behavior.

### Phase 2: Add Provider Serializers

Create `src/agent_foundation/apis/ag/multimodal.py` or a similarly scoped helper module for AG provider payload conversion.

Responsibilities:

- Convert `MultimodalInput` to Anthropic message content blocks.
- Convert `MultimodalInput` to OpenAI Chat Completions messages.
- Convert `MultimodalInput` to Gemini/OpenAI-compatible messages.
- Render a local-agent attachment manifest for Claude Code/Codex.
- Expose route-aware capability checks:
  - direct Claude versus Bedrock/proximity modes
  - OpenAI file block availability through AI Gateway
  - Gemini gateway compatibility

Provider-specific details:

- Claude serializer:
  - Preserve existing Anthropic text block shape for prompt text.
  - Add image base64 blocks for supported image MIME types.
  - Add document blocks for PDFs only when the selected mode supports them.
  - Use conservative size caps for gateway modes with stricter limits.
- OpenAI serializer:
  - Use `image_url` content parts with data URLs for images.
  - Use Chat Completions `file` blocks for supported files.
  - Do not mix Responses API `input_file` structures into current chat payloads.
- Gemini serializer:
  - Use OpenAI-compatible chat content parts initially.
  - Keep a capability gate so failing gateway support is caught as an explicit unsupported multimodal mode, not a malformed request downstream.
- Local manifest renderer:
  - Produce concise, deterministic text that lists staged/relative file paths and metadata.
  - Avoid XML or provider-specific markup unless an existing local prompt convention requires it.

Acceptance criteria:

- Golden payload tests prove exact output shapes for:
  - text only
  - prompt plus one PNG
  - prompt plus multiple images
  - prompt plus PDF
  - prompt plus text file
  - unsupported binary file
- Tests assert base64 appears only inside intended API payload fields, never inside sanitized logs/manifests.
- Raw provider-shaped messages pass through unchanged.

### Phase 3: Wire AI Gateway Backend Modules

Modify:

- `src/agent_foundation/apis/ag/ai_gateway_claude_llm.py`
- `src/agent_foundation/apis/ag/ai_gateway_openai_llm.py`
- `src/agent_foundation/apis/ag/ai_gateway_gemini_llm.py`

Implementation outline:

- Extend public helper signatures to accept `attachments: Optional[Sequence[...]] = None`:
  - `_get_messages(...)`
  - `generate_text(...)`
  - `generate_text_streaming(...)`
  - `generate_text_async(...)`
- Allow `prompt_or_messages` to be a `MultimodalInput` without breaking current accepted types.
- Keep raw `dict` and `list[dict]` handling as pass-through.
- Route all explicit attachments through the shared normalizer and provider serializer.
- Add capability checks before `_build_request_payload()`, so unsupported media fails with a clear local exception.
- Keep `_build_request_payload()` mostly focused on request-level parameters; avoid burying media validation there.

Acceptance criteria:

- Existing AG text tests pass unchanged.
- New tests prove sync, async, and streaming call the same message serializer.
- `set_messages()` still bypasses normalization exactly as before.
- String path prompt reading still works for legacy text prompt files.
- Explicit `attachments=[...]` treats the path as an attachment, not as a replacement prompt.
- Unsupported media errors identify provider, gateway mode, MIME type, and suggested supported alternatives.

### Phase 4: Wire AG Inferencer Wrappers

Modify:

- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_claude_api_inferencer.py`
- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_openai_api_inferencer.py`
- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_gemini_api_inferencer.py`

Implementation outline:

- Accept `attachments` in `_infer()`, `_ainfer()`, and `_ainfer_streaming()` via `**_inference_args` / `**kwargs`.
- Pass attachments through to the AG backend module.
- If `inference_input` is already a `MultimodalInput`, avoid double-normalization.
- Preserve `set_messages()` raw override semantics:
  - If `_messages_override` is present, use it and ignore explicit attachments only if documented.
  - Prefer raising a clear error if both `_messages_override` and `attachments` are provided, because merging raw provider messages with neutral attachments can be ambiguous.
- Ensure streaming resets `_messages_override` as it does today.

Acceptance criteria:

- Wrapper tests monkeypatch AG backend functions and assert `attachments` reaches them unchanged.
- Text-only wrapper calls are byte-for-byte or structurally equivalent to current behavior.
- Ambiguous `set_messages()` plus `attachments` behavior is tested and documented.

### Phase 5: Wire Claude Code CLI and SDK Inferencers

Modify:

- `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_cli_inferencer.py`
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_sdk_inferencer.py`
- Possibly `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/common.py` for shared prompt/manifest helpers.

Implementation outline:

- Accept explicit `attachments` and `MultimodalInput`.
- Normalize using `effective_cwd` and allowed path policy.
- Stage files outside `effective_cwd` into `.agent_foundation/attachments/<sha256>/`.
- Build a manifest and append/prepend it to the prompt before passing to:
  - CLI stdin prompt flow
  - SDK `client.query(prompt)`
- Keep existing session, streaming, JSON parsing, and usage extraction unchanged.
- Do not add nonexistent Claude Code CLI media flags.
- If the installed Claude Agent SDK exposes a native content-block/media API, add it behind a version/capability check and keep path-manifest fallback.

Acceptance criteria:

- Unit tests prove text-only command construction and SDK prompt calls remain unchanged.
- Attachment tests prove:
  - under-cwd files are referenced by relative path
  - outside-cwd allowed files are staged
  - outside-cwd disallowed files fail clearly
  - prompt sent to CLI/SDK contains manifest metadata but not base64
  - `large_input_mode=STDIN` still sends the final prompt over stdin
- Streaming tests still parse `stream-json` events as before.

### Phase 6: Wire Codex CLI and SDK Inferencers

Modify:

- `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_cli_inferencer.py`
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_sdk_inferencer.py`
- Optionally create `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/common.py` if helper duplication appears.

Implementation outline:

- Accept explicit `attachments` and `MultimodalInput`.
- Use the same staging and local manifest strategy as Claude Code for generic files.
- Probe and test Codex CLI image flag support before adding native `--image`:
  - If fresh `codex exec` supports `--image`, add image flags with properly quoted staged paths.
  - If only resume supports `--image`, use it only on compatible resume commands.
  - If installed CLI lacks support, fall back to the manifest strategy and mark native image flag support unavailable.
- Keep `codex exec --json` event parsing unchanged.
- For Codex SDK, keep `thread.turn(prompt)` manifest strategy unless typed media inputs are confirmed.
- Verify the existing resume command behavior while touching command construction. There is already risk around resume commands and flags that are valid only for fresh exec; do not make that worse when adding image flags.

Acceptance criteria:

- New Codex test folder exists under `test/agent_foundation/common/inferencers/external/codex/`.
- Text-only command construction remains unchanged.
- Image attachments produce native CLI flags only when the test fixture declares support.
- Generic file attachments produce staged manifest references.
- SDK tests assert `thread.turn()` receives manifest-enriched text only when attachments are present.
- Streaming JSON parsing still handles `thread.started`, `item.completed`, and `turn.completed` events.

### Phase 7: Cache, Logging, and Artifact Hygiene

Modify target base classes only if a narrow hook is cleaner than per-inferencer overrides.

Likely options:

- Add a target-local cache key override in the four external streaming inferencers when attachments are present.
- Or add a small base hook such as `get_inference_cache_key(inference_input, **kwargs)` that defaults to current prompt extraction.

Requirements:

- Attachment fingerprint must affect streaming cache identity.
- Raw bytes/base64 must not enter cache keys.
- Debug logs should include only sanitized metadata:
  - filename
  - MIME type
  - byte size
  - SHA-256 prefix
  - staged relative path for local agents
- Staging directory should be deterministic, easy to clean, and excluded from source control if needed.

Acceptance criteria:

- Same prompt plus different attachment fingerprints does not reuse cached stream output.
- Same prompt plus same attachment fingerprint may reuse cache if existing caching semantics allow it.
- Logs and checkpoints contain no raw base64 payloads.

### Phase 8: Documentation and Examples

Modify or add concise docs:

- `src/agent_foundation/apis/ag/README.md`
- `test/agent_foundation/common/inferencers/external/claude_code/README.md`
- New `test/agent_foundation/common/inferencers/external/codex/README.md` if useful
- Optional docs under `docs/` only if this repo has a user-facing inferencer guide

Document:

- Supported attachment forms.
- Provider support matrix.
- Size limits and unsupported behavior.
- Local agent staging behavior.
- Security/logging guarantees.
- Examples for:
  - one image
  - multiple images
  - text file
  - PDF where supported
  - raw provider message override

Acceptance criteria:

- An implementer or caller can use the feature without reading source code.
- Docs state route/mode limitations honestly.
- Docs do not claim unsupported generic binary upload.

## Risk Register

| Severity | Risk | Specific failure mode | Mitigation |
|---|---|---|---|
| 🔴 High | Gateway route incompatibility | AI Gateway direct, Bedrock, proximity, or Gemini OpenAI-compatible routes reject content blocks that provider docs support elsewhere. | Add route-aware capability gates and tests. Fail locally with clear unsupported-mode errors before sending malformed requests. |
| 🔴 High | Misleading "file support" | Arbitrary binaries are accepted but the model cannot inspect them, producing false confidence. | Define file support by MIME and provider route. Support text/images/PDFs first. Reject unknown binaries for API inferencers unless explicitly supported. |
| 🔴 High | Base64 leakage | Raw image/PDF bytes appear in logs, cache keys, checkpoints, test snapshots, or exception messages. | Centralize sanitized representation. Unit test redaction. Never use raw payload in repr/cache/log messages. |
| 🔴 High | Path safety and symlink escapes | Local attachment staging reads files outside allowed roots or follows symlinks unexpectedly. | Resolve real paths, validate regular files, require allowed roots, copy into controlled staging directory, and test symlink escape cases. |
| 🟡 Med | Streaming cache collisions | Same prompt with different attachments reuses old cached output. | Include attachment fingerprint metadata in streaming cache key or disable caching for multimodal requests until cache hooks are updated. |
| 🟡 Med | CLI flag drift | Codex or Claude CLI versions differ from docs; `--image` support is absent or only valid on some subcommands. | Probe installed CLI help and test command construction by capability. Default to path manifest. |
| 🟡 Med | SDK capability uncertainty | `claude-agent-sdk` or `openai-codex` may support native media differently than docs or local imports suggest. | Inspect installed SDK types during implementation. Keep manifest fallback. Update `src/BUCK` or packaging only after dependency path is verified. |
| 🟡 Med | Raw message override regression | Existing callers using `set_messages()` lose structured message pass-through. | Keep override behavior unchanged. Add regression tests with structured content arrays. |
| 🟡 Med | Token and payload bloat | Text files or PDFs are inlined too aggressively and exceed token/gateway limits. | Use size caps, prefer local path references for agentic tools, and expose explicit unsupported/too-large errors. |
| 🟡 Med | Sync/async/streaming drift | One mode supports attachments but another silently ignores them. | Route all modes through shared normalization/serialization and add tests for all three per AG provider. |
| 🟡 Med | Existing path-as-prompt ambiguity | A caller passes a file path string expecting legacy text-read behavior, but it becomes an attachment. | Preserve legacy behavior for string input. Only explicit `attachments=` or `MultimodalInput.attachments` triggers attachment semantics. |
| 🟢 Low | MIME guessing inaccuracies | `mimetypes` guesses `application/octet-stream` for some valid files. | Allow explicit `mime_type` override and clear error messages. Avoid new binary dependencies unless needed. |
| 🟢 Low | Staging directory clutter | `.agent_foundation/attachments` grows over time. | Use content-addressed staging, document cleanup, and optionally add TTL cleanup later. |
| 🟢 Low | Scope pressure into conversational/server APIs | Text-only schema boundaries tempt broad refactors. | Keep first implementation limited to requested inferencers. Document future expansion separately. |

## Files to Create or Modify

### New Files

- `src/agent_foundation/common/inferencers/multimodal_input.py`
  - Neutral attachment/input/content models.
  - Normalization, validation, MIME inference, fingerprinting, redacted display.
  - Local staging helper or shared staging primitives.
- `src/agent_foundation/apis/ag/multimodal.py`
  - AG provider serializers and route-aware support helpers.
  - Anthropic, OpenAI, and Gemini message conversion.
- `test/agent_foundation/common/inferencers/test_multimodal_input.py`
  - Neutral model and validation tests.
- `test/agent_foundation/apis/ag/test_multimodal_payloads.py`
  - Provider serializer golden tests and AG backend integration tests.
- `test/agent_foundation/common/inferencers/external/codex/test_codex_multimodal_inputs.py`
  - New Codex CLI/SDK multimodal tests.
- `test/agent_foundation/common/inferencers/external/claude_code/test_claude_code_multimodal_inputs.py`
  - Claude Code CLI/SDK multimodal tests.

### Existing Files to Modify

- `src/agent_foundation/apis/ag/ai_gateway_claude_llm.py`
  - Accept explicit attachments.
  - Use Anthropic serializer.
  - Preserve raw message pass-through and legacy path-as-text prompt behavior.
- `src/agent_foundation/apis/ag/ai_gateway_openai_llm.py`
  - Accept explicit attachments.
  - Use OpenAI Chat Completions serializer.
  - Preserve existing message normalization.
- `src/agent_foundation/apis/ag/ai_gateway_gemini_llm.py`
  - Accept explicit attachments.
  - Use Gemini/OpenAI-compatible serializer with capability gate.
- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_claude_api_inferencer.py`
  - Pass attachments from inferencer methods to backend.
- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_openai_api_inferencer.py`
  - Pass attachments from inferencer methods to backend.
- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_gemini_api_inferencer.py`
  - Pass attachments from inferencer methods to backend.
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_cli_inferencer.py`
  - Normalize explicit attachments.
  - Stage/reference files.
  - Add manifest to prompt while preserving CLI command and streaming parsing.
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_sdk_inferencer.py`
  - Normalize explicit attachments.
  - Use manifest-enhanced prompt or native SDK media if verified.
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/common.py`
  - Optional home for shared Claude Code manifest/staging helpers.
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_cli_inferencer.py`
  - Normalize explicit attachments.
  - Use manifest strategy and capability-gated image flags.
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_sdk_inferencer.py`
  - Normalize explicit attachments.
  - Use manifest-enhanced prompt or native SDK media if verified.
- `src/agent_foundation/common/inferencers/streaming_inferencer_base.py`
  - Optional narrow cache-key hook only if target-local overrides are insufficient.
- `src/BUCK`
  - Add only necessary dependency metadata.
  - Prefer no new third-party deps for initial implementation.
  - Verify `openai-codex` packaging before adding or relying on SDK-native media.
- `src/agent_foundation/apis/ag/README.md`
  - Document AG multimodal usage and limits.

### Files Expected Not to Change in First Pass

- `src/agent_foundation/agents/agent_attachment.py`
  - Do not repurpose. Optional future adapter only.
- `src/agent_foundation/common/inferencers/agentic_inferencers/conversational/inbox.py`
  - Out of scope for this request.
- `src/agent_foundation/server/schema.py`
  - Out of scope for this request.
- `src/agent_foundation/common/configs/registered_targets.py`
  - Existing aliases already register the target classes; no alias change is expected.

## Validation and Testing Strategy

### Unit Tests

Add focused unit coverage for the new neutral model:

- Normalize string prompt.
- Normalize dict prompt.
- Normalize `MultimodalInput`.
- Normalize explicit `attachments=` keyword input.
- Resolve relative and absolute paths.
- Reject missing files, directories, unsupported sources, and oversized files.
- Infer MIME types for `.png`, `.jpg`, `.pdf`, `.txt`, and unknown extensions.
- Respect explicit MIME overrides.
- Compute deterministic fingerprints.
- Verify redacted display excludes bytes and base64.

Add provider serializer golden tests:

- Claude text-only output remains existing Anthropic text block shape.
- Claude image output uses image source blocks.
- Claude PDF output uses document source blocks only when supported.
- OpenAI image output uses `image_url` data URLs.
- OpenAI supported file output uses Chat Completions file blocks, not Responses API blocks.
- Gemini output uses OpenAI-compatible chat content arrays only behind capability support.
- Unsupported binary raises clear local exceptions.

### AG Backend and Wrapper Tests

For each of Claude, OpenAI, and Gemini:

- Existing text-only tests still pass.
- `generate_text()`, `generate_text_async()`, and `generate_text_streaming()` all serialize attachments through the same helper.
- Wrapper `_infer()`, `_ainfer()`, and `_ainfer_streaming()` pass attachments through.
- `set_messages()` still passes raw messages unchanged.
- `set_messages()` plus explicit attachments has documented and tested behavior.
- Legacy path-as-prompt text reading remains unchanged.

Recommended commands:

- `python -m pytest test/agent_foundation/apis/ag/test_gateway_mode.py`
- `python -m pytest test/agent_foundation/apis/ag/test_multimodal_payloads.py`
- `python -m pytest test/agent_foundation/common/inferencers/test_multimodal_input.py`

### Claude Code and Codex Tests

Claude Code:

- Extend existing folder `test/agent_foundation/common/inferencers/external/claude_code/`.
- Test prompt manifest rendering.
- Test staging behavior under a temp `target_path`.
- Test no base64 in prompt for local agentic inferencers.
- Test CLI `large_input_mode=STDIN` still receives the final prompt.
- Test SDK `client.query()` receives unchanged text for no-attachment calls and manifest text for attachment calls.

Codex:

- Create `test/agent_foundation/common/inferencers/external/codex/`.
- Test command construction with no attachments.
- Test image attachment behavior with capability fixture:
  - native flag enabled
  - native flag disabled
- Test generic file manifest behavior.
- Test SDK `thread.turn()` prompt behavior.
- Test resume command flag placement if native image flags are added.

Recommended commands:

- `python -m pytest test/agent_foundation/common/inferencers/external/claude_code`
- `python -m pytest test/agent_foundation/common/inferencers/external/codex`
- `python -m pytest test/agent_foundation/common/inferencers/test_large_input_mode.py`
- `python -m pytest test/agent_foundation/common/inferencers/test_large_arg_offload.py`

### Security and Regression Tests

Add tests for:

- Symlink escaping out of allowed roots.
- Staging an outside-cwd but allowed file.
- Rejecting outside-cwd disallowed files.
- Reusing staged path for identical file fingerprint.
- Cache key includes attachment fingerprint.
- Logs/checkpoints do not include raw base64.

Run relevant existing inferencer tests after implementation:

- `python -m pytest test/agent_foundation/common/inferencers`
- `python -m pytest test/agent_foundation/apis/ag`

If the full suite is too slow, run targeted tests first, then a broader inferencer/API subset before merge.

### Optional Live Smoke Tests

Live tests should be opt-in and environment-gated, not part of the default unit suite.

Suggested live smoke cases:

- AG Claude direct mode with a tiny PNG.
- AG Claude supported mode with a tiny PDF if route supports documents.
- AG OpenAI with a tiny PNG.
- AG Gemini with a tiny PNG only after gateway compatibility is confirmed.
- Claude Code CLI with a staged text file and a small image path.
- Codex CLI with a staged text file and, if supported, native `--image`.

For reproducibility, live smoke tests should write structured results to an artifact file such as `_runtime/multimodal_smoke/results.json` rather than relying only on stdout.

## Implementation Acceptance Criteria

The feature is complete when:

- All five requested inferencer surfaces accept explicit attachments.
- The three AG backends serialize supported images/files into provider-appropriate structured payloads.
- Claude Code and Codex local inferencers can use images/files through safe path references, with Codex native image flags used only when verified.
- Text-only calls remain backward compatible.
- Existing raw message overrides remain backward compatible.
- Unsupported files fail clearly and early.
- No raw base64/bytes appear in logs, cache keys, prompts for local agentic tools, or checkpoints.
- Different attachment content changes streaming cache identity.
- New unit tests and targeted regression tests pass.
- Documentation states supported media types, limits, and route caveats.

## Self-Validation

- Required section present: High-Level Approach.
- Required section present: Key Execution/Implementation Steps with dependency-ordered phases and acceptance criteria.
- Required section present: Risk Register with severity ratings and mitigations.
- Required section present: Files to Create/Modify.
- Required section present: Validation/Testing Strategy.
- Referenced target paths exist in the repository.
- The plan covers both Claude Code inferencers, both Codex inferencers, and all three AG API inferencers.
- The plan accounts for backend AI Gateway modules under `src/agent_foundation/apis/ag/`, where payload construction actually occurs.
- The plan avoids implementing the full solution and stays at roadmap level.
- The plan distinguishes image support, text file support, PDF/document support, and unsupported arbitrary binary files.
- The plan follows the dependency note: absence from `src/BUCK` is treated as an investigation item, not proof of unavailability.
