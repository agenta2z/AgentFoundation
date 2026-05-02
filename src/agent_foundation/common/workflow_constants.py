"""Framework-level workflow constants shared across agent_foundation and server.

Lives here (not in ``server/workflow_context.py``) to break the dependency
cycle: the conversational inferencer needs ``_WORKFLOW_DESC_PHASE_RE`` to
render SOP guidance, but ``agent_foundation`` cannot depend on ``server_lib``
(which already depends on ``agent_foundation``). Extracting this constant to
a small framework-side module lets both directions resolve cleanly.
"""

from __future__ import annotations

import re

_WORKFLOW_DESC_PHASE_RE: re.Pattern[str] = re.compile(
    r"\*\*\s*Phase\s+(\w+)\s*[—–\-]+\s*(.+?)\s*\*\*\s*:",
)
