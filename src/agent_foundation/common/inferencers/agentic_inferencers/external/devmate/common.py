# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""Common utilities for Devmate inferencers.

The Devmate server validates model names against a strict ``ModelName``
enum that uses dot-separated version numbers (e.g. ``claude-opus-4.6``),
while Claude Code CLI uses dash-separated versions (e.g. ``claude-opus-4-6``)
and the Anthropic API uses full date-qualified identifiers
(e.g. ``claude-opus-4-6-20260204`` from ``ClaudeModels``).

This module provides:
- ``resolve_model_tag``: Normalize any model tag string for Devmate.
- ``DevmateConfig``: Enum of known Devmate config files for inferencers.
"""

import logging
import os
import hashlib
import re
import shutil
from enum import Enum
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Devmate config enum
# ---------------------------------------------------------------------------

# Path (relative to fbsource root) for custom Devmate configs
# co-located with the inferencer code.
_DEVMATE_CONFIG_DIR: str = (
    "fbcode/agent_foundation/common/inferencers"
    "/agentic_inferencers/external/devmate/configs"
)


class DevmateConfig(str, Enum):
    """Known Devmate config files for AgentFoundation inferencers.

    Members are valid ``config_name`` / ``config_file_path`` values
    accepted by ``DevmateCliInferencer`` and ``DevmateSDKInferencer``.
    Extends ``str`` so it works wherever a plain string config name
    is expected (CLI args, SDK client, etc.).

    Built-in configs (resolved from ``tools/devmate/configs/``):
        FREEFORM:      Minimal prompt-only config. Uses server defaults
                       (``max_iterations=50``, default model). Variables
                       ``model_name`` / ``max_tokens`` are NOT declared
                       and will be silently ignored.
        FREEFORM_FAST: ``max_iterations=200``, limited tool set
                       (read/edit/search/delete/exit only — no
                       ``execute_command`` or ``write_to_file``).

    Custom AgentFoundation configs (co-located under ``configs/``):
        AGENT_FOUNDATION_AGENTIC: Extends ``freeform.md``. Declares
                            ``model_name``, ``max_iteration``, and
                            ``max_output_tokens`` as template variables
                            so they are properly substituted. Full
                            default tool set, ``max_iterations=200``,
                            ``max_time_mins=60``,
                            ``max_total_tokens=10_000_000``,
                            ``max_output_tokens=64_000``.
    """

    # Built-in configs (in tools/devmate/configs/)
    FREEFORM = "freeform"
    FREEFORM_FAST = "freeform_fast"

    # Custom AgentFoundation configs (co-located with inferencer code)
    AGENT_FOUNDATION_AGENTIC = f"{_DEVMATE_CONFIG_DIR}/freeform_agentic"


_logger = logging.getLogger(__name__)


def get_source_repo_root() -> str:
    """Auto-detect the fbsource root where this inferencer code lives.

    Walks up from this file to the fbsource root. This file is at:
        <fbsource>/fbcode/agent_foundation/common/
                   inferencers/agentic_inferencers/external/devmate/common.py
    So fbsource root is 8 parents up.
    """
    return str(Path(__file__).resolve().parents[8])


def sync_config_to_target(
    config_name: str,
    source_path: str,
    target_path: str,
) -> None:
    """Copy a custom config from source_path repo to target_path repo.

    When the Devmate agent operates on a different repo (target_path) than
    where the inferencer code lives (source_path), custom configs must be
    present in the target repo for the Devmate server to resolve them.

    Always copies the latest version from source to target to ensure
    consistency. Skips built-in configs (no path separator) that live in
    the shared ``tools/devmate/configs/`` directory.

    Note: This creates files in the target repo (e.g., fbs_cfr_dev) that
    appear as untracked in source control. This is expected — the Devmate
    server resolves configs relative to its operating repo, so the config
    must exist there. These copied files are safe to delete or ignore.

    Args:
        config_name: Config name/path (e.g. DevmateConfig.AGENT_FOUNDATION_AGENTIC).
        source_path: fbsource root where configs are authored.
        target_path: fbsource root where Devmate agent operates.
    """
    if os.sep not in config_name and "/" not in config_name:
        return

    src_resolved = os.path.realpath(source_path)
    tgt_resolved = os.path.realpath(target_path)
    if src_resolved == tgt_resolved:
        return

    for suffix in (".md", ".yaml", ""):
        src = Path(source_path) / (config_name + suffix)
        if src.exists():
            dst = Path(target_path) / (config_name + suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            _logger.info("Synced config %s -> %s", src.name, dst)
            return

    _logger.warning(
        "Config '%s' not found in source_path '%s'",
        config_name,
        source_path,
    )


def generate_config_with_allowed_commands(
    base_config_name: str,
    allowed_commands: List[str],
    source_path: str,
) -> str:
    """Generate a config that extends the base config with additional allowed commands.

    Creates a deterministic temp config file that extends the base config and
    adds ``allowed_commands`` entries for the shell tool. The file is written to
    a ``_generated/`` subdirectory alongside the base config.

    Note: This function assumes the base config uses ``execute_command:`` as the
    shell tool key (matching ``freeform_agentic.md``). If a base config uses a
    different key (e.g., ``shell:``), the generated config's ``execute_command:``
    entry would create a separate tool instead of merging.

    Args:
        base_config_name: Config name/path (e.g., DevmateConfig.AGENT_FOUNDATION_AGENTIC).
            Should NOT include ``.md`` extension.
        allowed_commands: List of command executables to allow (e.g.,
            ``["nvidia-smi", "nvcc"]``).
        source_path: fbsource root where configs are authored.

    Returns:
        Config path in the same format as ``DevmateConfig`` values (relative to
        fbsource root, without ``.md`` extension).
    """
    # Deterministic hash includes base config name to prevent collisions
    hash_input = Path(base_config_name).name + "|" + ",".join(sorted(allowed_commands))
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    base_stem = Path(base_config_name).stem
    generated_name = f"_auto_{base_stem}_{short_hash}"

    # Resolve base config directory and create _generated/ subdir
    base_config_dir = str(Path(base_config_name).parent)
    generated_dir = base_config_dir + "/_generated"
    generated_config_name = generated_dir + "/" + generated_name

    # Build the extends path (relative from _generated/ back to parent dir)
    base_filename = Path(base_config_name).name + ".md"
    extends_path = "../" + base_filename

    # Build allowed_commands YAML entries
    commands_yaml = "\n".join(
        f"          - executable: '{cmd}'" for cmd in allowed_commands
    )

    config_content = (
        "---\n"
        "# Auto-generated config. Do not edit manually.\n"
        "# Extends base config with additional allowed shell commands.\n"
        f"extends: '{extends_path}'\n"
        "mcp_servers:\n"
        "  tools:\n"
        "    execute_command:\n"
        "      config:\n"
        "        allowed_commands:\n"
        f"{commands_yaml}\n"
        "---\n"
        "${{ prompt:str }}\n"
    )

    # Write to _generated/ subdirectory (skip if identical content already exists).
    generated_file_path = Path(source_path) / (generated_config_name + ".md")
    generated_file_path.parent.mkdir(parents=True, exist_ok=True)
    if not generated_file_path.exists():
        generated_file_path.write_text(config_content)

    _logger.info(
        "Generated config with allowed_commands %s -> %s",
        allowed_commands,
        generated_file_path,
    )

    return generated_config_name


# ---------------------------------------------------------------------------
# Devmate server-side output token limit.
#
# Source: fbcode/devai/config/llm_config.py (LLMConfig.max_output_tokens field)
#   max_output_tokens: int | None = Field(
#       default=None,
#       description="max_output_tokens",
#       le=64000,  # plugboard vertex errors for > 64000 output tokens
#   )
#
# This limit is enforced by the Plugboard/Vertex AI backend. Setting
# max_tokens higher than this value will cause server-side errors.
# If this limit changes upstream, update this constant accordingly.
# ---------------------------------------------------------------------------
DEVMATE_MAX_OUTPUT_TOKENS: int = 64000


# Explicit mapping for Anthropic API / ClaudeModels values and Claude Code
# short aliases whose Devmate ModelName cannot be derived by simple regex.
# Covers legacy 3.x naming where the Devmate enum omits the dash between
# "claude" and the major version (e.g. "claude3.5-sonnet").
_KNOWN_ALIASES: dict[str, str] = {
    # Legacy 3.x full API names → Devmate plugboard names
    # NOTE: claude-3-opus-20240229 (Claude 3 Opus) has no Devmate equivalent;
    # do NOT map it to gcp-claude-4-opus which is Claude 4 Opus.
    "claude-3-haiku-20240307": "claude3-haiku",
    "claude-3-5-sonnet-20241022": "claude3.5-sonnet",
    "claude-3-7-sonnet-20250219": "claude3.7-sonnet",
    # Legacy 3.x short aliases → Devmate plugboard names
    "claude-3-5-sonnet": "claude3.5-sonnet",
    "claude-3-7-sonnet": "claude3.7-sonnet",
    "claude-3-5-haiku": "claude3.5-haiku",
    "claude-3-haiku": "claude3-haiku",
    # 4.0 single-version (different ordering in Devmate)
    "claude-sonnet-4-20250514": "claude4-sonnet",
    "claude-sonnet-4": "claude4-sonnet",
}

# Pattern: trailing -YYYYMMDD (8-digit date suffix)
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")

# Pattern: digit-digit NOT followed by 3+ more digits (version separator)
_DASH_VERSION_RE = re.compile(r"(\d)-(\d)(?!\d{3,})")


def resolve_model_tag(model_tag: str) -> str:
    """Normalize a model tag for Devmate CLI / SDK.

    Handles three input formats:

    1. **Anthropic API / ClaudeModels** (full date-qualified)::

        claude-opus-4-6-20260204   → claude-opus-4.6
        claude-sonnet-4-5-20250929 → claude-sonnet-4.5
        claude-3-5-sonnet-20241022 → claude3.5-sonnet

    2. **Claude Code CLI / dash-separated versions**::

        claude-opus-4-6            → claude-opus-4.6
        claude-sonnet-4-5          → claude-sonnet-4.5
        claude-3-5-sonnet          → claude3.5-sonnet

    3. **Already-correct Devmate format** (no-op)::

        claude-opus-4.6            → claude-opus-4.6
        claude3.5-sonnet           → claude3.5-sonnet

    Args:
        model_tag: Model tag string in any format.

    Returns:
        Model tag normalized for Devmate server (dot-separated versions,
        no date suffix).
    """
    # 1. Check explicit alias table first (handles legacy naming)
    if model_tag in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[model_tag]

    # 2. Strip trailing date suffix (-YYYYMMDD)
    result = _DATE_SUFFIX_RE.sub("", model_tag)

    # 3. Check alias table again after stripping date
    if result != model_tag and result in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[result]

    # 4. Convert dash-separated version digits to dots
    result = _DASH_VERSION_RE.sub(r"\1.\2", result)

    return result
