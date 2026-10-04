# pyre-strict

"""Common utilities for Codex inferencers."""

import logging
import shutil
from typing import Optional

_logger: logging.Logger = logging.getLogger(__name__)

# The Codex CLI looked up on ``PATH`` when no path is configured.
CODEX_BINARY = "codex"

# Model used in place of a tag Codex cannot serve (the classic Codex CLI
# inferencer's default ``model_name``).
CODEX_DEFAULT_MODEL = "gpt-5.5"

# Anthropic model families a heterogeneous config may cascade to every backend
# (``CodexCliInferencer._NON_CODEX_MODEL_PREFIXES``).
NON_CODEX_MODEL_PREFIXES: tuple[str, ...] = (
    "opus",
    "sonnet",
    "haiku",
    "claude",
    "fable",
)


def find_codex_binary(explicit: Optional[str] = None) -> Optional[str]:
    """The ``codex`` executable to run: ``explicit`` when set, else the one on
    ``PATH``; ``None`` when neither (callers then use the bare name)."""
    return explicit or shutil.which(CODEX_BINARY)


def resolve_model_tag(model_tag: str) -> str:
    """The Codex model for ``model_tag``, as the classic Codex CLI inferencer
    resolves it: a Claude-family tag (``opus[1m]``, ``claude-sonnet-4-6``)
    becomes ``CODEX_DEFAULT_MODEL``; any other tag is passed through."""
    if model_tag.lower().startswith(NON_CODEX_MODEL_PREFIXES):
        _logger.info(
            "Codex cannot use model %r; using %s", model_tag, CODEX_DEFAULT_MODEL
        )
        return CODEX_DEFAULT_MODEL
    return model_tag
