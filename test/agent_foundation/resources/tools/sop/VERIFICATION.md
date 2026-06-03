# SOP CLI End-to-End Verification

## Last Verified

- **Date**: 2026-05-30
- **Workspace**: `role_creation__20260530_085118__2e0d4637`
- **Model**: sonnet
- **Mode**: yolo (auto-advance all confirmations)
- **Runtime**: ~2h 38m (01:51 → 04:29 PDT)

## Verification Checklist

### Request Passthrough
- [x] User query reaches LLM as first turn input (use `--request` flag, not positional after `nargs=*` flags)
- [x] `turn_001/user_input.txt` contains the actual user request
- **Known issue (fixed)**: Positional `request` arg placed after `--extra-tool-dirs` (nargs=*) was silently swallowed by argparse. Fixed by adding `--request` named flag. Test script updated to use `--request`.

### Phase Progression
- [x] Phase 0 — Role Specification (multiple_choice / clarification, yolo auto-advanced)
- [x] Phase 1 — Role Document Creation (`create_role` BTA, 8 workers, 8 outputs)
- [x] Phase 1b — Confirmation (yolo auto-approved)
- [x] Phase 2 — Role Setup (`role_setup` BTA, 8 workers, 67 outputs)
- [x] Phase 2b — Confirmation (yolo auto-approved)
- [x] SOP completion detected, process exited cleanly with "SOP completed successfully."

### Turn Logging
- [x] 5 turns logged with full artifacts per turn:
  - `user_input.txt` — captured via `on_new_turn` (fires before first iteration)
  - `rendered_prompt.txt` — captured via `on_prompt_rendered`
  - `response.md` — LLM raw response
  - `template_feed.json`, `template_config.json`, `template_source.txt`
  - `messages.json` — full conversation snapshot via `on_turn_complete` (fires on all exit paths)
- [x] `cache_folder` set before first LLM call (streaming files land in `turn_001/`)

### Deliverables Produced
- 11-section role responsibility document
- 17 skills (6 reused + 11 new SKILL.md)
- 30 tools (19 reused + 11 new tool.json)
- 13 knowledge blocks
- Role Setup Report with Day-1 readiness checklist

## How to Run

```bash
bash test/agent_foundation/resources/tools/sop/test_sop.sh role_creation "hire a machine learning engineer"
```

Python >= 3.11 required (StrEnum). The script auto-detects a suitable Python via `_find_python()`.
