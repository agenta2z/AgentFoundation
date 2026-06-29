
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


class SessionMode(Enum):
    """Controls when DevmateCliInferencer starts a fresh devmate session.

    Trade-off: session reuse preserves tool-use context (model "remembers"
    cached file reads, prior tool results) but accumulates per-session caps
    (``max_iterations``, ``max_time_mins``) and risks state corruption.
    """

    SAME_SESSION_ACROSS_ROUNDS = "same"
    """Reuse active_session_id across all calls."""

    NEW_SESSION_PER_CALL = "per_call"
    """Fresh session for every infer/ainfer call."""

    NEW_SESSION_ON_ERROR = "on_error"
    """Reuse session on success; reset on any InferencerExecutionError."""

    NEW_SESSION_ON_CONSECUTIVE_ERRORS = "on_consecutive_errors"
    """Reuse normally; reset after N consecutive errors (default N=2)."""


_logger = logging.getLogger(__name__)


def _detect_fbsource_root_for(cls) -> "str | None":
    """Detect the Sapling/Mercurial repo root containing ``cls``'s source file.

    Walks up from ``cls``'s source file looking for an ``.hg/`` marker.
    Used by Devmate inferencers to set ``source_path`` to the fbsource
    root (Devmate's custom configs at
    ``fbsource/fbcode/tools/devmate/configs/...`` live ABOVE
    AgentFoundation's project root, so the standard src-layout detector
    returns the wrong dir).

    Returns ``None`` when no ``.hg/`` marker is found (test stubs,
    pip-installed packages, non-Sapling environments). Callers chain
    with ``super()._detect_source_root()`` for the fallback.
    """
    import inspect

    try:
        source_file = inspect.getfile(cls)
    except (TypeError, OSError):
        return None
    current = os.path.dirname(os.path.abspath(source_file))
    while True:
        if os.path.exists(os.path.join(current, ".hg")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


# Sapling / EdenSCM / Mercurial repo-root markers. Devmate's Rust server
# REQUIRES its CWD to resolve to one of these (else it aborts with
# "Failed to find repo root"). ``~/fbsource`` has ``.hg`` + ``.eden``.
_REPO_ROOT_MARKERS = (".hg", ".sl", ".eden")


def cwd_repo_root(path: "str | None") -> "str | None":
    """Walk up from ``path`` to the nearest Sapling/EdenSCM/Mercurial repo root.

    Returns the directory containing a ``.hg`` / ``.sl`` / ``.eden`` marker, or
    ``None`` if ``path`` is not inside such a repo. Unlike
    ``_detect_fbsource_root_for`` (which walks the *inferencer class's* source
    file to locate ``source_path`` for config sync), this walks an arbitrary
    filesystem ``path`` — used to decide whether ``devmate`` can start with that
    CWD and, if not, to reroot the server (see
    ``DevmateCliInferencer._resolve_subprocess_cwd``).
    """
    if not path:
        return None
    current = os.path.abspath(os.path.expanduser(path))
    while True:
        if any(os.path.exists(os.path.join(current, m)) for m in _REPO_ROOT_MARKERS):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


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
# "claude" and the major version (e.g. "claude3.5-sonnet"), the bracket
# form for 1M-context variants ("[1m]" in Anthropic SDK → "-1m" in Devmate),
# and the Anthropic "latest" alias ("opus" → most recent Claude Opus).
#
# These are verified against Devmate's server-side ``ModelName`` enum in
# fbcode/devai/config/llm_config.py.
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
    # 1M-context bracket form ("[1m]") → Devmate dash-form ("-1m").
    # Anthropic SDK uses "claude-opus-4-7[1m]" while Devmate's ModelName
    # enum uses "claude-opus-4.7-1m" (CLAUDE_OPUS_4_7_1M_PLUGBOARD).
    "claude-opus-4-6[1m]": "claude-opus-4.6-1m",
    "claude-opus-4-7[1m]": "claude-opus-4.7-1m",
    "claude-opus-4.6[1m]": "claude-opus-4.6-1m",  # already-dotted variant
    "claude-opus-4.7[1m]": "claude-opus-4.7-1m",
    # Anthropic "latest" short aliases → current latest Devmate model.
    # Update these when a newer Claude generation supersedes 4.7.
    "opus": "claude-opus-4.7",
    "opus[1m]": "claude-opus-4.7-1m",
}

# Pattern: trailing -YYYYMMDD (8-digit date suffix)
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")

# Pattern: digit-dash-digit that's a VERSION SEPARATOR (e.g. ``4-7`` in
# ``claude-opus-4-7``), NOT a context-window suffix (e.g. ``7-1`` in
# ``claude-opus-4.7-1m``) or a date prefix.
#
# Restrictions:
#   ``(?!\d{3,})`` - reject when followed by 3+ more digits (date suffix)
#   ``(?![a-zA-Z])`` - reject when followed by a letter, which would
#     indicate a suffix like ``-1m`` (context-window) or ``-sonnet`` etc.
#     where the dash is meaningful and must be preserved.
_DASH_VERSION_RE = re.compile(r"(\d)-(\d)(?!\d{3,})(?![a-zA-Z])")


def resolve_model_tag(model_tag: str) -> str:
    """Normalize a model tag for Devmate CLI / SDK.

    Handles four input formats:

    1. **Anthropic API / ClaudeModels** (full date-qualified)::

        claude-opus-4-6-20260204   → claude-opus-4.6
        claude-sonnet-4-5-20250929 → claude-sonnet-4.5
        claude-3-5-sonnet-20241022 → claude3.5-sonnet

    2. **Claude Code CLI / dash-separated versions**::

        claude-opus-4-6            → claude-opus-4.6
        claude-sonnet-4-5          → claude-sonnet-4.5
        claude-3-5-sonnet          → claude3.5-sonnet

    3. **1M-context bracket form** (Anthropic SDK style) ::

        claude-opus-4-7[1m]        → claude-opus-4.7-1m
        claude-opus-4-6[1m]        → claude-opus-4.6-1m
        opus[1m]                   → claude-opus-4.7-1m

    4. **Anthropic "latest" short aliases**::

        opus                       → claude-opus-4.7   (current latest)
        opus[1m]                   → claude-opus-4.7-1m

    5. **Already-correct Devmate format** (no-op)::

        claude-opus-4.6            → claude-opus-4.6
        claude3.5-sonnet           → claude3.5-sonnet
        claude-opus-4.7-1m         → claude-opus-4.7-1m

    Args:
        model_tag: Model tag string in any format.

    Returns:
        Model tag normalized for Devmate server (dot-separated versions,
        no date suffix, ``-1m`` suffix for 1M-context variants). Returns
        the input unchanged if no rule matches — callers should validate
        against Devmate's server-side ``ModelName`` enum if they want to
        catch unmapped values.
    """
    # 1. Check explicit alias table first (handles legacy naming + brackets)
    if model_tag in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[model_tag]

    # 2. Strip trailing date suffix (-YYYYMMDD)
    result = _DATE_SUFFIX_RE.sub("", model_tag)

    # 3. Check alias table again after stripping date
    if result != model_tag and result in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[result]

    # 4. Convert dash-separated version digits to dots
    result = _DASH_VERSION_RE.sub(r"\1.\2", result)

    # 5. Re-check alias table once more after dash→dot conversion. This
    #    catches inputs like ``claude-opus-4-7[1m]`` where the regex
    #    transforms the version portion to ``4.7`` and the resulting
    #    ``claude-opus-4.7[1m]`` is in the alias table.
    if result in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[result]

    return result
