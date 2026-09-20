# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Service logic exposing the reference script library manifest.

Ported from RankEvolve ``submission_templates_routes.py``. FastAPI stripped:
the former endpoint is a plain service function. The thin REST router lives in
OpenTeam and calls this function.

The setup wizard's "From Library" tab fetches this on open and lists the
templates so the user can pick a starting point. Selecting a template sets
``libraryTemplate`` on the POST to ``/api/hubs/<mid>/submission-setup``; the
agent server's ``setup_submission_script`` resolves it via
``submission_templates_loader`` and inlines the bundled scripts into the PTI
prompt.

Former endpoint → service function:
  GET .../submission-templates → list_submission_templates
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


async def list_submission_templates() -> dict[str, Any]:
    """Return the reference-script manifest as a flat list.

    Empty list if the bundled manifest is missing — never raises (the
    setup wizard still works; only the From-Library tab degrades).
    """
    # Lazy import — keeps the module importable when the
    # submission_templates resource isn't yet vendored.
    try:
        # ``submission_templates_loader`` is ported alongside this service
        # (mapped via the rankevolve.src.server.X ->
        # agent_foundation.experiment_hub.X convention, same as
        # submission_launcher). The bundled ``resources/submission_templates/``
        # manifest + reference scripts ship with the package.
        from agent_foundation.experiment_hub.submission_templates_loader import (
            list_library_templates,
        )
    except Exception as e:
        logger.warning("submission_templates_loader unavailable: %s", e)
        return {"templates": []}

    templates = await asyncio.to_thread(list_library_templates)
    return {"templates": templates}
