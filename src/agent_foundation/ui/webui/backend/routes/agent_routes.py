# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
REST endpoints for real agent mode.

Provides help text and other non-streaming agent APIs.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

HELP_TEXT = """\
**Available Commands:**

| Command | Description |
|---------|-------------|
| `/task <request>` | Run dual-agent task (default: full workflow) |
| `/task-plan <request>` | Plan only (no implementation) |
| `/task-execute <request>` | Execute only (skip planning) |
| `/task-full <request>` | Full plan + implement workflow |
| `/task-confirm <request>` | Plan then confirm before implementing |
| `/model <name>` | Change the LLM model |
| `/target-path [path]` | Show or set the codebase root |
| `/clear` | Clear conversation history |
| `/help` | Show this help |
| `/kn <subcommand>` | Knowledge management |
| `/exit` | End session |

**Task Flags:**
`--analysis`, `--multi-iter`, `--resume <path>`, `--analysis-only <path>`, \
`--analysis-mode <last|cross-ref|all-rounds>`, `--base-inferencer <type>`, \
`--review-inferencer <type>`, `--no-planning`, `--no-implementation`, \
`--in-place`, `--replay-streaming`, `--model <name>`, `--claude-only`

**Knowledge Subcommands:**
`add`, `load`, `search`, `list`, `get`, `update`, `delete`, `restore`, \
`status`, `clear`, `history`, `rollback`, `export`, `import`, `spaces`
"""


@router.get("/help")
async def get_help() -> dict[str, str]:
    """Return formatted help text for all agent commands."""
    return {"content": HELP_TEXT}
