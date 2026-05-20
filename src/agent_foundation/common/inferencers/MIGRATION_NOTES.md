# Migration Dependency Gaps

Summary of missing modules discovered during the inferencers migration from
`_dev/rankevolve/` to `AgentFoundation/`. These modules are referenced by
migrated code but do not exist in the destination project. They require
separate migration or stub creation before the affected imports will resolve.

**Last updated:** after agentic-foundation-migration spec completion.

---

## RESOLVED — Migrated Modules

The following modules were migrated and are now available in `agent_foundation/`.
All internal imports have been updated from `rankevolve.src.*` /
`agentic_foundation.*` to `agent_foundation.*`.

| Module | Status | Notes |
|--------|--------|-------|
| `agent_foundation.resources.tools` | ✅ Migrated | `models.py`, `registry.py`, `formatters/markdown.py`, 13 tool.json files, 2 jinja2 templates |
| `agent_foundation.ui.widget_protocol` | ✅ Migrated | `WidgetField`, `WidgetMessage`, `WidgetResponse` |
| `agent_foundation.ui.interactive_checkpoint` | ✅ Migrated | `run_checkpoint`, `checkpoint_plan_review` |
| `agent_foundation.server.workflow_context` | ✅ Migrated | `_WORKFLOW_DESC_PHASE_RE`, `WorkflowContext`, `WorkflowPhaseRecord` |
| `agent_foundation.common.response_parsers` | ✅ Migrated | `extract_delimited`, `delimiter_parser.py` |
| `agent_foundation.apis.plugboard` | ✅ Migrated | `generate_text`, `generate_text_async`, `generate_text_streaming` |

### Additional fixes applied

- **`apis/claude_llm.py`** — `ClaudeModels` enum synced with source: `CLAUDE_41OPUS` typo fixed to `CLAUDE_41_OPUS`, missing models added (`CLAUDE_45_HAIKU`, `CLAUDE_45_OPUS`, `CLAUDE_46_SONNET`), `CLAUDE_46_OPUS` value corrected.
- **`apis/metagen/metagen_llm.py`** — `CLAUDE_4_6_OPUS` model added, `CompletionMode` enum added, `generate_text_streaming()` added.
- **`ui/webui/backend/services/agent_service_bridge.py`** — stale `rankevolve` imports fixed (`interactive_base`, `queue_client_base`); remaining `rankevolve` imports marked TODO.
- **`ui/webui/backend/routes/workspace_routes.py`** — stale `rankevolve` import marked TODO.

---

## REMAINING GAPS (not yet migrated)

These modules are still referenced via `rankevolve.src.*` imports in the
destination codebase. They are marked with `# TODO` comments in the affected
files.

### `rankevolve.src.common.workspace.layout`

Workspace layout utilities. Not yet migrated to `agent_foundation`.

| File | Line(s) | Import | Context |
|------|---------|--------|---------|
| `ui/webui/backend/routes/workspace_routes.py` | 17 | `from rankevolve.src.common.workspace.layout import ...` | TODO — lazy/guarded |
| `ui/webui/backend/services/agent_service_bridge.py` | 71 | `from rankevolve.src.common.workspace.layout import ...` | TODO — lazy/guarded |

### `rankevolve.src.common.streaming.file_tailer`

File tailing utility for streaming logs. Not yet migrated.

| File | Line(s) | Import | Context |
|------|---------|--------|---------|
| `ui/webui/backend/services/agent_service_bridge.py` | 70, 110 | `from rankevolve.src.common.streaming.file_tailer import ...` | TODO — lazy/guarded |

### `rankevolve.src.server.llm.plugboard_client`

Plugboard client used by the plugboard LLM module. Not yet migrated.

| File | Line(s) | Import | Context |
|------|---------|--------|---------|
| `apis/plugboard/plugboard_llm.py` | ~110 | `from rankevolve.src.server.llm.plugboard_client import PlugboardClient` | Lazy import inside function — does not block module load |

### `rankevolve.src.resources.prompt_templates`

Prompt template loading used by workflow context. Not yet migrated.

| File | Line(s) | Import | Context |
|------|---------|--------|---------|
| `server/workflow_context.py` | (lazy) | `from rankevolve.src.resources.prompt_templates import ...` | Lazy import — does not block module load |

---

## Summary

| Remaining Gap | Severity | Affected Files |
|---------------|----------|----------------|
| `rankevolve.src.common.workspace.layout` | TODO (guarded) | `workspace_routes.py`, `agent_service_bridge.py` |
| `rankevolve.src.common.streaming.file_tailer` | TODO (guarded) | `agent_service_bridge.py` |
| `rankevolve.src.server.llm.plugboard_client` | LAZY | `plugboard_llm.py` |
| `rankevolve.src.resources.prompt_templates` | LAZY | `workflow_context.py` |

**Total: 4 remaining gaps, all guarded or lazy — no top-level blockers remain.**

All previously identified top-level import blockers (`resources.tools`,
`ui.widget_protocol`) have been resolved. The remaining gaps are deferred
imports that will only fail at runtime if the specific code paths are hit.
