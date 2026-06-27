# Multimodal Attachment Support for Claude Code, Codex, and AI Gateway Inferencers

## Summary

Add first-class image/file attachment support across:

- Claude Code external inferencers:
  - `agentic_inferencers/external/claude_code`
- Codex external inferencers:
  - `agentic_inferencers/external/codex`
- AI Gateway API inferencers:
  - `api_inferencers/ag`
  - the underlying `apis/ag` OpenAI, Claude, and Gemini clients

The current code is effectively prompt-first: string input works, `{"prompt": ...}` works, but non-prompt fields are either dropped or only pass through implicitly in narrow API cases. The proper fix is a shared, typed attachment input contract plus provider-specific conversion, not ad-hoc prompt string concatenation.

This plan was created after a task-tool plan-only smoke test. The task run allocated:

`/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_runtime/tasks/attachment_support_plan_test/attachment_support_plan_test_20260626_163754_266eed24`

That smoke test did not produce a usable plan: `children/propose/outputs/output.md` was zero bytes, and the RovoDev breakdown process was killed after its output file stayed empty for 120 seconds. The plan below is therefore based on direct source inspection and independent explorer audits, with the failed task run recorded as a task-tool/runtime issue rather than treated as valid planning output.

## Current Findings

- `StreamingInferencerBase._extract_prompt()` in `common/inferencers/streaming_inferencer_base.py` returns only `dict["prompt"]` for dict inputs. This is the main shared point where `attachments`, `images`, or `files` would currently be erased before reaching streaming subclasses.
- `TemplatedInferencerBase._render_prompt()` renders to a single prompt value. Any attachment-aware input must preserve attachments across template rendering instead of replacing the whole input with a plain string.
- Claude Code CLI/SDK inferencers currently operate on a prompt string. The CLI has local filesystem access and supports `--add-dir`; installed CLI help does not show a local `--image` flag. It does show `--file file_id:relative_path`, which is a remote file-resource startup mechanism, not a general local attachment API.
- Codex CLI has native image support via `codex exec --image <FILE>...`; Codex SDK currently calls `thread.turn(prompt)` and needs a separate SDK capability check before promising native multimodal support.
- AI Gateway OpenAI and Gemini clients normalize strings to OpenAI-style chat messages. Claude normalizes strings to Anthropic text blocks. All three accept raw dict/list messages in some paths, but there is no canonical attachment model, MIME validation, base64 conversion, or consistent streaming preservation.
- API streaming wrappers currently use `set_messages()` as a one-shot structured-message side channel; otherwise the streaming base reduces input to a string prompt.

## Target Input Contract

Introduce a backward-compatible attachment contract in a shared module, for example:

`agent_foundation/common/inferencers/attachment_input.py`

Define:

- `AttachmentSpec`
  - `kind`: `"image" | "file" | "text"`
  - `source`: `"path" | "url" | "bytes" | "file_id"`
  - `path`: optional local path
  - `url`: optional remote URL
  - `data`: optional bytes/base64 payload
  - `mime_type`: optional explicit MIME type
  - `filename`: optional display name
  - `description`: optional user-facing hint
  - `metadata`: optional dict
- `InferenceInputEnvelope`
  - `prompt`: string
  - `attachments`: list of `AttachmentSpec`
  - `messages`: optional provider-neutral structured messages

Supported public input shapes:

- existing string input: unchanged
- existing `{"prompt": "..."}`: unchanged
- new `{"prompt": "...", "attachments": [...]}`
- new `InferenceInputEnvelope`
- provider-native `messages` for API inferencers, preserved when already supplied

Validation rules:

- local paths must exist before transport conversion
- local paths must remain inside allowed workspace/additional directories unless an existing inferencer already has broader local access
- infer MIME type from extension when omitted, but allow explicit override
- distinguish image attachments from generic files; do not silently send an unsupported generic file as an image
- impose configurable size limits before base64 encoding
- preserve string-input byte-for-byte behavior for existing callers

## Implementation Plan

### 1. Shared normalization and preservation

Modify the shared inferencer layer so attachment-bearing input survives all common paths.

- Add `normalize_inference_input(value) -> InferenceInputEnvelope`.
- Keep `_extract_prompt()` as a compatibility helper, but internally use the normalized envelope.
- In `StreamingInferencerBase._ainfer_streaming_pipeline()`:
  - normalize once
  - cache by rendered prompt plus attachment metadata digest
  - pass `attachments` and/or the full envelope to `_ainfer_streaming(prompt, **kwargs)`
  - do not require all subclasses to implement attachment handling immediately; unsupported subclasses should either ignore empty attachments or raise a clear unsupported error for non-empty attachments.
- In `TemplatedInferencerBase._render_prompt()`:
  - if the input is an envelope or `{"prompt": ..., "attachments": ...}`, render only the prompt text and return the same envelope shape with attachments preserved.
  - if the input is a plain string, preserve current behavior.
- Update docs/comments so future subclasses do not call `str(input)` or `dict["prompt"]` directly when they need structured input.

Acceptance criteria:

- string input, prompt-only dict input, and rendered templated input produce the same prompt text as before.
- `{"prompt": "x", "attachments": [...]}` reaches a test subclass with attachments intact in sync and async streaming paths.

### 2. AI Gateway provider conversion

Implement real provider-specific attachment conversion in `apis/ag` first, then make the three wrapper inferencers call it.

- OpenAI/Gemini conversion:
  - text-only remains `{"role": "user", "content": "..."}` where currently expected.
  - image attachments convert to OpenAI-compatible content parts, e.g. text part plus image URL/data URL part, subject to the gateway schema actually accepted by the typed request model.
  - generic file attachments should initially be converted only when the gateway has an explicit accepted schema; otherwise fail clearly with "generic file attachments are not supported by this provider path yet" and include the filename/path.
- Claude conversion:
  - text remains Anthropic `{"type": "text", "text": ...}` blocks.
  - image attachments convert to Anthropic image blocks using media type plus base64 source where supported.
  - generic file attachments require explicit gateway support; otherwise fail clearly.
- Streaming:
  - remove reliance on `set_messages()` for new attachment input by allowing structured envelope input to pass through normal streaming calls.
  - keep `set_messages()` backward-compatible for existing callers.
- Direct/raw JSON mode:
  - do not over-normalize already provider-native messages; validate lightly and pass through.
  - document that explicit provider-native messages bypass the provider-neutral attachment conversion.

Acceptance criteria:

- `AgOpenAIApiInferencer`, `AgClaudeApiInferencer`, and `AgGeminiApiInferencer` can receive prompt plus image attachment through sync and async paths.
- Streaming paths preserve attachments without requiring callers to pre-call `set_messages()`.
- Unsupported file/provider combinations raise explicit errors before making network calls.

### 3. Codex external inferencers

Support Codex CLI images natively and support files through the safest transport available.

- `CodexCliInferencer`
  - parse attachments from normalized input.
  - for image path attachments, add `--image <path>` once per image.
  - preserve stdin prompt mode for large prompt text.
  - for generic local files, stage as accessible workspace files and inject a short generated attachment manifest into the prompt unless Codex CLI gains a native file flag.
  - ensure `--add-dir` includes required attachment directories only when needed and only within allowed paths.
- `CodexSdkInferencer`
  - inspect the installed SDK API before implementation.
  - if SDK supports multimodal turn input, map image attachments to the native message/part shape.
  - if not, use the same workspace-staged file-reference fallback as the CLI, with an explicit warning/log entry that this is path-reference support rather than model-native upload.

Acceptance criteria:

- `codex exec --image` command construction is covered by unit tests.
- string prompt command construction is unchanged.
- generic files are visible to the agent through workspace paths and described in the prompt manifest.
- unsupported byte/url/file cases fail before subprocess launch unless conversion is implemented.

### 4. Claude Code external inferencers

Treat Claude Code as local-agent attachment support unless verified native transport support exists.

- `ClaudeCodeCliInferencer`
  - parse attachments from normalized input.
  - for local files/images, ensure the containing directory is readable through `--add-dir` or existing target/workspace access.
  - inject a compact attachment manifest into the prompt with paths, filenames, MIME types, and descriptions.
  - do not misuse `--file file_id:relative_path` for local files; only use it if the attachment source is explicitly `file_id`.
- `ClaudeCodeSdkInferencer`
  - inspect SDK capabilities before deciding whether to send native multimodal content.
  - if only string query is available, use the same manifest-plus-local-path approach as CLI.
  - preserve existing session/resume behavior and partial-message streaming.

Acceptance criteria:

- both Claude Code inferencers accept prompt plus local image/file paths without dropping them.
- generated prompt manifest is deterministic and does not duplicate absolute paths unnecessarily.
- `--add-dir` behavior is tested and does not broaden access beyond attachment directories already under the allowed workspace roots.
- `file_id` attachments are either implemented with Claude's `--file` semantics or rejected with a precise message.

### 5. Attachment staging and manifest helper

Create one shared helper used by CLI-style agentic inferencers:

`agent_foundation/common/inferencers/attachment_staging.py`

Responsibilities:

- validate and canonicalize attachment specs
- compute MIME type and safe display name
- optionally copy or symlink attachments into a run-local `attachments/` folder when needed
- generate a deterministic markdown manifest for prompt injection
- return extra readable directories for CLI command construction
- produce stable metadata for cache keys and run-state persistence

Use this helper for Claude Code and Codex local-agent paths. Do not duplicate path validation and prompt-manifest formatting in each inferencer.

Acceptance criteria:

- one test fixture validates the helper with image path, text file path, missing path, unsupported URL, and explicit MIME override.
- CLI inferencer tests assert they call the helper rather than each having divergent formatting logic.

## Files to Modify or Create

Create:

- `src/agent_foundation/common/inferencers/attachment_input.py`
- `src/agent_foundation/common/inferencers/attachment_staging.py`
- focused tests under `test/agent_foundation/common/inferencers/` for shared normalization/staging

Modify:

- `src/agent_foundation/common/inferencers/streaming_inferencer_base.py`
- `src/agent_foundation/common/inferencers/templated_inferencer_base.py`
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_cli_inferencer.py`
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/claude_code_sdk_inferencer.py`
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_cli_inferencer.py`
- `src/agent_foundation/common/inferencers/agentic_inferencers/external/codex/codex_sdk_inferencer.py`
- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_openai_api_inferencer.py`
- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_claude_api_inferencer.py`
- `src/agent_foundation/common/inferencers/api_inferencers/ag/ag_gemini_api_inferencer.py`
- `src/agent_foundation/apis/ag/ai_gateway_openai_llm.py`
- `src/agent_foundation/apis/ag/ai_gateway_claude_llm.py`
- `src/agent_foundation/apis/ag/ai_gateway_gemini_llm.py`
- API gateway README/docs for the new input shapes and provider limitations

## Testing Strategy

Unit tests:

- normalize string input, prompt-only dict, envelope object, and prompt-plus-attachments dict
- template rendering preserves attachments while changing only the prompt
- streaming base passes attachments to a fake subclass
- attachment staging validates path existence, MIME inference, size limits, stable manifest output, and allowed-directory behavior
- provider conversion tests for OpenAI/Gemini content parts and Claude content blocks
- unsupported provider/file combinations fail before network/subprocess calls
- Codex CLI command includes `--image` for image path attachments
- Claude Code CLI command includes only necessary `--add-dir` entries and prompt manifest text

Integration tests with mocked transports:

- `AgOpenAIApiInferencer`, `AgClaudeApiInferencer`, and `AgGeminiApiInferencer` sync/async/streaming calls receive structured messages with attachments preserved.
- `CodexCliInferencer` subprocess command construction is correct without launching the real CLI.
- `ClaudeCodeCliInferencer` subprocess command construction is correct without launching the real CLI.
- SDK inferencer tests should mock SDK clients and assert exact turn/query payloads or manifest fallback behavior.

Manual smoke tests:

- one image-only prompt through Codex CLI path
- one local image plus local text file through Claude Code CLI path
- one image prompt through each AI Gateway provider path, using a small checked-in fixture image
- existing text-only plan/tool workflows still run unchanged

Task-tool regression:

- rerun the plan-only task command that produced the zero-byte artifact after the RovoDev empty-output issue is fixed or avoided.
- acceptance: the task run produces non-empty `children/propose/outputs/output.md` and returns successfully without needing manual interruption.

## Risks and Mitigations

- Risk: shared streaming changes break text-only inferencers.
  - Mitigation: preserve `_extract_prompt()` behavior for string/prompt-only dict input; add regression tests using fake streaming subclasses and representative existing inferencers.
- Risk: provider schemas differ more than expected.
  - Mitigation: keep provider conversion explicit and fail early for unsupported attachment/provider combinations; do not silently stringify binary or generic files.
- Risk: CLI local-agent attachment support is mistaken for true model-native upload.
  - Mitigation: distinguish native transport support from workspace-staged path-reference support in logs, docs, and tests.
- Risk: attachment paths broaden filesystem access.
  - Mitigation: validate allowed roots, add only minimal `--add-dir` entries, and reuse existing target/workspace access rules.
- Risk: cache/resume keys ignore attachments.
  - Mitigation: include attachment metadata digest in cache identity and persist manifest metadata in run artifacts.

## Success Criteria

- All five requested inferencer families accept a common prompt-plus-attachments input shape.
- Existing text-only callers remain backward-compatible.
- API inferencers send image attachments as provider-appropriate multimodal message parts where supported.
- Codex CLI uses native `--image` for image path attachments.
- Claude Code and non-native SDK/file paths use a deterministic, validated workspace-reference manifest rather than dropping attachments.
- Unsupported combinations fail with clear, preflight errors.
- The implementation includes enough tests that future inferencer additions can reuse the shared attachment normalization instead of inventing another ad-hoc path.
