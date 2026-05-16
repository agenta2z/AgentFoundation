"""Path constants for inferencer resources."""

from pathlib import Path

# Canonical prompt_templates root under agent_foundation/resources/.
# All template spaces (plan/, implementation/, recovery/, etc.) live here.
DEFAULT_PROMPT_TEMPLATES_DIR = str(
    Path(__file__).resolve().parent.parent.parent.parent / "resources" / "prompt_templates"
)

# Backward-compat alias (used by prompt_templates/__init__.py).
DEFAULT_RECOVERY_DIR = DEFAULT_PROMPT_TEMPLATES_DIR
