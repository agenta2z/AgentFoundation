# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Submission script subprocess runner for the Experiment Hub.

``SubmissionRunner`` wraps the user's generated ``submit_v<n>.py`` as a
subprocess so the Hub can: (1) stream output live into the existing task-
subtab UI via the same file-tailer pipeline PTI uses, (2) parse
``FLOW_URI:`` / ``MAST_JOB:`` lines and emit them back to the WebUI as
soon as they appear (NOT at exit), (3) be cancelled cleanly when the
user clicks Cancel run.

Architectural notes (Section 5 of the design doc, Option 2-B refinement):

- Output goes to ``<workspace>/_runtime/inferencer_cache/submission/stream_run.txt``
  matching the EXACT pattern the inferencer bridges use, so the existing
  ``WorkspaceStreamTailer`` + forwarding pipeline can pick it up with no new
  machinery.
- Launch command is loaded from the script's sibling ``launch.json`` (the
  agent's PTI run wrote it). The runner handles placeholder substitution
  (``${ENABLE_FLAGS}`` / ``${EXP_NAME}`` / ``${CODEBASE_ROOT}``) and
  shell-metachar scrubbing on script_args, then DELEGATES command
  construction to a launcher strategy looked up in
  ``submission_launcher._LAUNCHER_REGISTRY`` by the optional
  ``"launcher"`` field (defaults to ``"buck_run_auto"`` for backward
  compat). Adding new launcher types is a launcher-side concern; this
  module never needs to change.
- ``stream_run.txt`` is written by THIS runner; the WebUI's
  stream-from-cache-folder logic instantiates the tailer + send-callback in
  the WebUI process where the WebSocket handle lives. The runner only
  produces the file.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent_foundation.experiment_hub.hub_markers import (
    STREAM_DONE_MARKER,
    STREAM_FAIL_MARKER,
)
from agent_foundation.experiment_hub.submission_launcher import (
    default_launcher_name,
    get_launcher,
    LaunchValidationError,
)

logger: logging.Logger = logging.getLogger(__name__)


# Shell-metacharacter blacklist applied to substituted ``script_args``
# values. ``cwd`` is the proper channel for working-directory changes;
# embedded ``cd ... && ...`` chains are explicitly rejected. The check
# defends against a malicious ``experiment_name`` value (the user
# ultimately controls it) injecting metachars after substitution.
#
# Option 3 schema: launch.json no longer carries ``cmd``; the runner
# always spawns ``buck run <auto_target> -- *script_args``. So we
# only need to scrub the per-arg values, not a full cmd[] template.
_SHELL_METACHARS: tuple[str, ...] = (";", "|", "&&", "||", ">", "<", "`", "$(")

# Per-line buffer for asyncio's subprocess StreamReader (default is 64 KB).
# Pushed to 16 MB so a single ultra-long line from a training run (e.g. a
# big bracketed JSON checkpoint dump) doesn't raise ``LimitOverrunError``
# and abort the run mid-stream. Defensive — most training stdout is line-
# oriented, but boundaries occasionally exceed 64 KB.
_MAX_STREAM_LINE_BYTES: int = 16 * 1024 * 1024  # 16 MB

# Regexes anchored at start-of-line so they tolerate the surrounding
# decoded UTF-8 line that ``_stream_pipe`` produces. ``\S+`` deliberately
# excludes whitespace — Windows ``\r`` is stripped before matching so it
# never sneaks into the captured group.
_FLOW_URI_RE: re.Pattern[str] = re.compile(r"^FLOW_URI:\s*(\S+)\s*$")
_MAST_JOB_RE: re.Pattern[str] = re.compile(r"^MAST_JOB:\s*(\S+)\s*$")

# Mode B local-runner contract markers. The runner emits these:
#
#   LOCAL_RUN: pid=<int> host=<hostname> workspace=<abs_path> [attempt=<int>]
#   STATUS: <key>=<value> [<key>=<value> ...]
#
# LOCAL_RUN appears once per attempt at process start and lets the UI
# render the local-run framing block (runHost, runLogPath). STATUS lines
# appear per epoch (epoch=N ndcg10=X hr10=Y mrr=Z) AND on terminal
# transitions (result=success epochs=N | result=watchdog_timeout etc.).
# Lenient parsing: emit runMode='local' even if pid/host extraction fails.
_LOCAL_RUN_RE: re.Pattern[str] = re.compile(
    r"^LOCAL_RUN:\s*"
    r"(?:pid=(\d+))?\s*"
    r"(?:host=(\S+))?\s*"
    r"(?:workspace=(\S+))?\s*"
    r"(?:attempt=(\d+))?"
)
_STATUS_RE: re.Pattern[str] = re.compile(r"^STATUS:\s+(.+)$")
_STATUS_KV_RE: re.Pattern[str] = re.compile(r"(\w+)=(\S+)")

# Literal placeholder string. Used both as the substitution target and
# as a presence check in launch.json (skip resolution entirely if no
# launch.json element contains it — preserves backward compat with
# v1 launch.json files that have hardcoded paths).
#
# Pattern semantics (the pattern itself is REQUIRED at the caller level
# — see ``SubmissionRunner.__init__`` — so different teams / future UI
# overrides can configure their own without a silent "fbsource"
# default):
#   - ``**/`` prefix means "any leading segments" (including zero).
#   - Remaining segments use ``fnmatch.fnmatchcase`` per-segment, so
#     ``*``, ``?``, ``[abc]`` work as in shell globs.
#   - The runner walks ancestors of the session's ``workflow_target_path``
#     and returns the deepest matching one. NO filesystem traversal —
#     purely string-pattern matching.
_CODEBASE_ROOT_PLACEHOLDER: str = "${CODEBASE_ROOT}"


# Auto-build phase: the runner can run the user's reference command
# (e.g. ``app-layer main fire-app -d ... launch_roo.py``) BEFORE spawning
# submit_v1.py, capture the resulting fbpkg version from stdout, and
# substitute it into ``${APP_LAYER_VERSION}``. This eliminates the need
# for the user to manually paste the version into the Submit Confirm
# modal after every ``app-layer`` build.
_BUILD_CMD_ALLOWLIST: tuple[str, ...] = ("app-layer", "buck", "buck2", "fbpkg")
# Capture line format (verified from real app-layer output, which is
# logger-prefixed — NOT bare). Anchoring with ``^`` would miss it.
#   ``I0420 19:40:56.569000 500624 app_layer.py:372] Built FBPKG: fire-app:2941a32``
_BUILT_FBPKG_RE: re.Pattern[str] = re.compile(r"Built FBPKG:\s*(\S+)")
# Match a per-token user-checkout path prefix that needs ``${CODEBASE_ROOT}``
# rewriting. Examples:
#   ``/data/users/zgchen/fbsource/fbcode``         -> strip prefix
#   ``/data/users/linfengliu/fbs_cfr_dev/fbcode``  -> strip prefix
# Pattern: ``/data/users/<user>/<checkout>/fbcode`` followed by ``/`` or end.
_USER_FBCODE_PREFIX_RE: re.Pattern[str] = re.compile(
    r"^/data/users/[^/]+/[^/]+/fbcode(/|$)"
)


class BuildFailed(Exception):
    """Raised by ``SubmissionRunner._run_build_phase`` when the user's
    auto-build command (typically ``app-layer main fire-app ...``) exits
    non-zero or succeeds without producing the expected
    ``Built FBPKG: <version>`` line. Caught by ``run()`` and translated
    into a clean ``submission_state: error`` event so the user sees the
    build failure (not a silent submit failure later)."""


def _tokenize_build_command(cmd_str: str) -> tuple[list[str], str]:
    """Safely parse the user's referenceCommand string into argv + cwd
    WITHOUT shell execution.

    Input shape (verified from the user's actual SetupWizard input):
      ``cd /data/users/<user>/<checkout>/fbcode && app-layer main fire-app -d ~/fbsource/fbcode/dper_lib/slimper_lib ... ~/fbsource/fbcode/.../launch_roo.py``

    Output:
      - ``cmd_tokens``: argv list with each token's user-checkout path
        prefix rewritten to ``${CODEBASE_ROOT}/...`` for portability.
        First token is the executable (validated against an allowlist
        by the caller, NOT here — this function is shape-only).
      - ``cwd_token``: the value to use as cwd. Either the path from the
        leading ``cd <path> &&`` (also rewritten to placeholder if it
        matches a user-checkout pattern) or ``${CODEBASE_ROOT}`` as a
        sensible default.

    Raises ``BuildFailed`` if the input is empty or contains more than
    one ``&&`` chain (multi-step builds belong to a future plan; one
    builder per submission keeps blame attribution clear).
    """
    if not cmd_str or not cmd_str.strip():
        raise BuildFailed("build_command is empty")

    # Split on the literal token ``&&``. ``shlex.split`` would consume
    # ``&&`` as a regular token (its default isn't shell-aware about
    # control operators), so we pre-split here.
    parts = [p.strip() for p in cmd_str.split("&&")]
    if len(parts) > 2:
        raise BuildFailed(
            "build_command contains more than one '&&' — only a single "
            "leading 'cd <path> && <build>' chain is supported"
        )

    cwd_token: str = _CODEBASE_ROOT_PLACEHOLDER  # default
    if len(parts) == 2:
        # Leading ``cd <path>`` provides the cwd; the remainder is the
        # actual build command.
        cd_chunk = parts[0]
        cd_tokens = shlex.split(cd_chunk)
        if len(cd_tokens) == 2 and cd_tokens[0] == "cd":
            cwd_token = _rewrite_user_path_token(cd_tokens[1])
            build_chunk = parts[1]
        else:
            # First chunk isn't a clean ``cd <path>`` — refuse rather
            # than silently misinterpret. The user can simplify their
            # reference command in the SetupWizard.
            raise BuildFailed(
                f"build_command's first '&&' chunk is not 'cd <path>': {cd_chunk!r}"
            )
    else:
        build_chunk = parts[0]

    cmd_tokens = shlex.split(build_chunk)
    if not cmd_tokens:
        raise BuildFailed("build_command resolved to an empty argv")

    cmd_tokens = [_rewrite_user_path_token(t) for t in cmd_tokens]
    return cmd_tokens, cwd_token


def _rewrite_user_path_token(token: str) -> str:
    """Per-token user-checkout path normalization. Applied to each argv
    token independently (NEVER to the joined cmd string — that would be
    brittle on nested ``fbcode`` segments).

    Three patterns are normalized to ``${CODEBASE_ROOT}``-based:
      - ``~/fbsource/fbcode/...``       -> tilde shorthand (most common
        in the user's verified reference command)
      - ``~/<other>/fbcode/...``        -> tilde with non-``fbsource``
        checkout dir
      - ``/data/users/<u>/<dir>/fbcode/...`` -> absolute path to any
        user's checkout
    Anything else (relative paths, non-checkout absolute paths, flags
    like ``-d``) passes through unchanged.
    """
    # Tilde-prefixed paths under any second-level dir (fbsource,
    # fbs_cfr_dev, etc.) ending in fbcode. Use a small pattern rather
    # than os.path.expanduser to avoid binding to the agent server's
    # $HOME (which differs from the running user's checkout root).
    m = re.match(r"^~/([^/]+)/fbcode(/|$)(.*)", token)
    if m:
        rest = m.group(3)
        return _CODEBASE_ROOT_PLACEHOLDER + ("/" + rest if rest else "")
    # Absolute /data/users/<u>/<dir>/fbcode prefix.
    m = _USER_FBCODE_PREFIX_RE.match(token)
    if m:
        # Strip the matched prefix (including the trailing ``/`` or end).
        return (
            _CODEBASE_ROOT_PLACEHOLDER
            + token[m.end() - (1 if m.group(1) == "/" else 0) :]
        )
    return token


def _extract_flow_id(flow_uri: str | None) -> str:
    """Pull the trailing flow ID out of an MLHub URL.

    Examples:
      ``https://mlhub.intern.facebook.com/run/f1234567890`` -> ``f1234567890``
      ``https://www.internalfb.com/mlhub/flow/1070390993/overview`` -> ``f1070390993``
      ``f1234567890`` -> ``f1234567890``

    Returns ``""`` when the input is empty / unparseable so callers can
    treat absence uniformly.

    Delegates to ``integrations.fblearner.utils.parse_experiment_identifier``
    for the heavy lifting (6 known URL forms + bare-ID handling). The
    try/except + falsy-input guard preserves the streaming-hot-path contract:
    callers receive ``""`` for unparseable input rather than a ``ValueError``
    that would crash the live submission-state pipeline.
    """
    if not flow_uri:
        return ""
    # TODO(port): confirm the fblearner.utils import path under AgentFoundation.
    # This is a real Meta SDK import (6-known-URL-form parser); kept at the
    # rankevolve path with `# @manual` per the port plan rather than guessed.
    from rankevolve.src.integrations.fblearner.utils import (  # @manual
        flow_id_to_experiment_id,
        parse_experiment_identifier,
    )

    try:
        ident = parse_experiment_identifier(flow_uri)
    except ValueError:
        return ""
    return flow_id_to_experiment_id(ident.flow_id)


class SubmissionRunner:
    """Run a generated submission script as a subprocess.

    Lifecycle:
      1. ``__init__`` captures workspace + script path + flags + the
         ``emit_event`` callback.
      2. ``run()`` builds the launch command from ``launch.json``, spawns
         the subprocess, streams stdout+stderr concurrently into both the
         workspace logs AND the inferencer-cache stream file, parses
         ``FLOW_URI:`` / ``MAST_JOB:`` lines and emits them via
         ``emit_event`` as soon as they appear, and writes the stream
         completion marker on exit.
      3. ``cancel()`` is invoked from outside (via the task-cancel WS
         handler) and gives the subprocess 5s to terminate gracefully
         before sending SIGKILL.

    The ``emit_event`` callback is called twice during a normal run:
      - once when ``FLOW_URI:`` is parsed (status='running', includes
        flowUri / experimentId / mastJob if known)
      - once at exit (status='completed' or 'error', includes
        runFinishedAt; mastJob may also be carried)
    """

    def __init__(
        self,
        workspace: Path,
        script_path: Path,
        enable_flags: list[str],
        experiment_name: str,
        launch_path: Path,
        emit_event: Callable[[dict[str, Any]], Awaitable[None]],
        workflow_target_path: str,
        codebase_root_pattern: str,
        app_layer_version: str = "",
        build_command: str = "",
    ) -> None:
        self.workspace: Path = workspace
        self.script_path: Path = script_path
        self.enable_flags: list[str] = list(enable_flags or [])
        self.experiment_name: str = experiment_name
        # Per-submission fbpkg version (e.g. "fire-app:2941a32"). Substituted
        # into ${APP_LAYER_VERSION} in launch.json script_args. Empty string
        # is treated as "unset" — _build_launch_command raises a clear
        # LaunchValidationError if the placeholder appears but the value
        # is empty.
        self.app_layer_version: str = app_layer_version
        # Auto-build recipe (the user's referenceCommand from the SetupWizard,
        # e.g. "cd /data/.../fbcode && app-layer main fire-app -d ... launch_roo.py").
        # When ``app_layer_version`` is empty AND this is non-empty, ``run()``
        # invokes ``_run_build_phase()`` first to execute it, capture
        # ``Built FBPKG: <version>`` from stdout, and populate
        # ``self.app_layer_version`` for the subsequent script_args
        # substitution. When the user types a value into the modal it
        # short-circuits the build (fast iteration on a known fbpkg).
        self.build_command: str = build_command
        self.launch_path: Path = launch_path
        # workflow_target_path is the session-side hint for which
        # codebase the user is operating on. May be a file or a
        # directory; the pattern matches against its ancestors to
        # extract the actual codebase root. Empty string is treated as
        # "unset" by _resolve_codebase_root and produces a clear
        # error if the launch.json actually uses ${CODEBASE_ROOT}.
        self._workflow_target_path: str = workflow_target_path
        self._codebase_root_pattern: str = codebase_root_pattern
        # CRITICAL (live-emit fix): FLOW_URI must reach the WebUI as soon
        # as the line is parsed, NOT at subprocess exit. An FBLearner
        # training job runs for hours-to-days after the launcher prints
        # FLOW_URI:; if we only emit on .run() return, the Monitor view
        # shows nothing until the job COMPLETES.
        self._emit_event: Callable[[dict[str, Any]], Awaitable[None]] = emit_event
        self.cmd: list[str] = []  # filled by _build_launch_command
        self.cwd: str = ""  # filled from launch.json["cwd"]
        self._proc: asyncio.subprocess.Process | None = None
        # Build-phase subprocess handle, separate from ``self._proc`` so
        # cancel during the build kills app-layer specifically (not a
        # spawn that hasn't happened yet).
        self._build_proc: asyncio.subprocess.Process | None = None
        self._cancelled: bool = False
        # Serialize stdout/stderr writes — they run concurrently and both
        # write to the same stream_run.txt file. Without serialization,
        # lines from one pipe can interleave inside lines from the other,
        # breaking the FLOW_URI:/MAST_JOB: regex matches.
        self._stream_lock: asyncio.Lock = asyncio.Lock()
        # Rolling per-epoch trajectory window for live UI updates. Capped
        # at 100 to bound payload size; full history still lives in
        # stream_run.txt for forensic.
        self._epoch_trajectory: list[dict[str, Any]] = []
        self._local_run_emitted: bool = False

    # ------------------------------------------------------------------
    # Launch resolution
    # ------------------------------------------------------------------

    def _resolve_codebase_root(self) -> str:
        """Walk ancestors of ``self._workflow_target_path`` and return the
        deepest one whose tail segments match ``self._codebase_root_pattern``.

        Pure string-pattern matching — does NOT touch the filesystem. So
        works regardless of whether the workflow_target_path actually
        exists on disk for the current user.

        Pattern semantics:
          - Leading ``**/`` is stripped and means "any leading segments"
            (including zero).
          - Remaining segments are matched per-segment using
            ``fnmatch.fnmatchcase``, so wildcards (``*``, ``?``,
            ``[abc]``) work within a segment.
          - E.g. ``**/fbsource/fbcode`` matches any path whose last two
            segments are exactly ``fbsource`` and ``fbcode``.
          - E.g. ``**/fbsource_*/fbcode`` matches paths where the
            second-to-last segment starts with ``fbsource_``.

        Raises ``LaunchValidationError`` if:
          - ``workflow_target_path`` is empty / unset (user must run
            ``/set-workflow-target-path`` first), OR
          - no ancestor of ``workflow_target_path`` matches the pattern
            (the path doesn't lie under a matching codebase root).

        Called only when ``${CODEBASE_ROOT}`` actually appears in the
        launch.json (so v1-style hardcoded launch.json files never
        trigger this path — preserves backward compat).
        """
        if not self._codebase_root_pattern:
            raise LaunchValidationError(
                "launch.json uses ${CODEBASE_ROOT} but no "
                "codebase_root_pattern was configured. The caller "
                "(typically the hub controller's submission-run path) must "
                "supply one — there is no built-in default because the "
                "right pattern is team-specific (e.g. **/fbcode, "
                "**/fbsource/fbcode, **/fbs_*/fbcode)."
            )
        if not self._workflow_target_path:
            raise LaunchValidationError(
                f"launch.json uses ${{CODEBASE_ROOT}} but the session has "
                f"no workflow_target_path set. Set it via "
                f"`/set-workflow-target-path <abs path under your "
                f"{self._codebase_root_pattern} checkout>` and re-submit."
            )

        # Strip leading "**/" — it means "match at any depth", which is
        # equivalent to walking ancestors anyway (so the segments after
        # "**/" are what we actually compare against each ancestor's tail).
        pattern = self._codebase_root_pattern
        if pattern.startswith("**/"):
            tail_pattern = pattern[3:]
        else:
            tail_pattern = pattern
        pattern_segments = tail_pattern.split("/")

        path = Path(self._workflow_target_path)
        # Check the path itself + every ancestor (deepest first).
        for ancestor in (path, *path.parents):
            ancestor_segments = ancestor.parts
            if len(ancestor_segments) < len(pattern_segments):
                continue
            tail_segments = ancestor_segments[-len(pattern_segments) :]
            if all(
                fnmatch.fnmatchcase(seg, pat)
                for seg, pat in zip(tail_segments, pattern_segments)
            ):
                logger.info(
                    "_resolve_codebase_root: pattern %r matched ancestor %s "
                    "of workflow_target_path %s",
                    self._codebase_root_pattern,
                    ancestor,
                    self._workflow_target_path,
                )
                return str(ancestor)

        raise LaunchValidationError(
            f"launch.json uses ${{CODEBASE_ROOT}} but no ancestor of "
            f"workflow_target_path {self._workflow_target_path!r} matches "
            f"the configured pattern {self._codebase_root_pattern!r}. "
            f"Either set workflow_target_path to a path under a matching "
            f"codebase root (via `/set-workflow-target-path`) or — once "
            f"the future UI pattern override ships — configure a pattern "
            f"that matches your layout."
        )

    def _build_launch_command(self) -> tuple[list[str], str]:
        """Read ``launch.json``, resolve ``${CODEBASE_ROOT}``, and
        delegate command construction to the declared launcher.

        New launch.json schema (Option 2-B):
        ```
        {
          "launcher": "buck_run_auto",  // OPTIONAL — defaults to buck_run_auto
          "script_args": ["--enable-flags", "${ENABLE_FLAGS}",
                          "--experiment-name", "${EXP_NAME}"],
          "cwd": "${CODEBASE_ROOT}"  // OPTIONAL — defaults to codebase root
        }
        ```
        The runner stays a thin process supervisor — it handles
        placeholder substitution + shell-metachar scrubbing, then hands
        the cleaned (script_args, cwd) to the launcher selected from
        ``submission_launcher._LAUNCHER_REGISTRY``. The launcher owns
        any wrapper-specific side effects (e.g. writing a BUCK file).

        Backward compat: launch.json without ``launcher`` falls back to
        ``buck_run_auto``, matching pre-2-B behavior.

        Raises ``LaunchValidationError`` on schema violation, unknown
        launcher, or launcher-specific install failure.
        """
        if not self.launch_path.is_file():
            raise LaunchValidationError(
                f"launch.json not found at {self.launch_path} — "
                "PTI did not produce a launch invocation; Re-generate "
                "the setup or use the script editor drawer to fix."
            )
        try:
            data = json.loads(self.launch_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise LaunchValidationError(f"launch.json is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise LaunchValidationError(
                "launch.json must be a JSON object with script_args[], "
                "optional cwd, and optional launcher fields"
            )

        # New schema: ``script_args`` is the contract.
        script_args_raw = data.get("script_args")
        if script_args_raw is None:
            # Soft-explain old schema for users hitting this after a
            # Submit on a v1-style launch.json. They can fix via the
            # script editor drawer or Re-generate the setup.
            if "cmd" in data:
                raise LaunchValidationError(
                    "launch.json uses the old `cmd`/`cwd` schema. "
                    "Re-generate the setup or edit launch.json via the "
                    "script editor drawer to use the new schema: "
                    '`{"script_args": [...], "cwd": "${CODEBASE_ROOT}"}` '
                    "(the runner now dispatches via the `launcher` "
                    "field, defaulting to `buck_run_auto`)."
                )
            raise LaunchValidationError(
                "launch.json missing required `script_args` (list of strings)"
            )
        if not isinstance(script_args_raw, list) or not all(
            isinstance(c, str) for c in script_args_raw
        ):
            raise LaunchValidationError(
                "launch.json `script_args` must be a list of strings"
            )

        # ``cwd`` is optional. Empty string after `data.get(...)` ->
        # launcher-decides default (typically codebase_root).
        cwd_raw = data.get("cwd", "")
        if cwd_raw is not None and not isinstance(cwd_raw, str):
            raise LaunchValidationError(
                "launch.json `cwd` (when present) must be a string"
            )
        cwd_raw = cwd_raw or ""

        # ``launcher`` is optional — defaults to buck_run_auto so v1-era
        # launch.json files continue to work without migration.
        launcher_name = data.get("launcher", default_launcher_name())
        if not isinstance(launcher_name, str) or not launcher_name:
            raise LaunchValidationError(
                "launch.json `launcher` (when present) must be a "
                "non-empty string identifying a known launcher (e.g. "
                "'buck_run_auto')"
            )
        launcher = get_launcher(launcher_name)

        # Resolve ${CODEBASE_ROOT} once, but ONLY if launch.json actually
        # uses it. Skipping resolution preserves backward compat with
        # v1-era launch.json files that have hardcoded paths AND avoids
        # spurious "no codebase root" errors when the launcher wouldn't
        # need one (e.g. a future passthrough launcher with no script
        # template needs).
        needs_root = _CODEBASE_ROOT_PLACEHOLDER in cwd_raw or any(
            _CODEBASE_ROOT_PLACEHOLDER in a for a in script_args_raw
        )
        codebase_root = self._resolve_codebase_root() if needs_root else ""

        # Validate ${APP_LAYER_VERSION} usage BEFORE substitution: if the
        # placeholder appears anywhere in script_args but no value was
        # provided, surface a clear actionable error rather than spawning
        # the script with a literal placeholder string (which would fail
        # opaquely deep inside FBLearner's package_version validation).
        needs_app_layer = any("${APP_LAYER_VERSION}" in a for a in script_args_raw)
        if needs_app_layer and not self.app_layer_version:
            raise LaunchValidationError(
                "launch.json uses ${APP_LAYER_VERSION} but no "
                "app_layer_version was provided. Type a fbpkg version "
                "(e.g. fire-app:2941a32) in the Submit Confirm modal, or "
                "edit launch.json via the script editor drawer to remove "
                "the placeholder."
            )

        # Substitute placeholders in script_args + scrub for metachars.
        substituted_args: list[str] = []
        enable_flags_str = ",".join(self.enable_flags)
        for arg in script_args_raw:
            v = (
                arg.replace("${ENABLE_FLAGS}", enable_flags_str)
                .replace("${EXP_NAME}", self.experiment_name)
                .replace("${APP_LAYER_VERSION}", self.app_layer_version)
                .replace(_CODEBASE_ROOT_PLACEHOLDER, codebase_root)
            )
            for meta in _SHELL_METACHARS:
                if meta in v:
                    raise LaunchValidationError(
                        f"launch.json script_args contains shell metachar "
                        f"{meta!r} in {v!r}; use cwd / separate args "
                        "instead of cd / pipe / redirect."
                    )
            if _CODEBASE_ROOT_PLACEHOLDER in v:
                raise LaunchValidationError(
                    f"launch.json script_args still contains "
                    f"{_CODEBASE_ROOT_PLACEHOLDER} after substitution: "
                    f"{v!r}. Internal error — _resolve_codebase_root "
                    "did not produce a value."
                )
            substituted_args.append(v)

        cwd_substituted = cwd_raw.replace(_CODEBASE_ROOT_PLACEHOLDER, codebase_root)
        if _CODEBASE_ROOT_PLACEHOLDER in cwd_substituted:
            raise LaunchValidationError(
                f"launch.json cwd still contains "
                f"{_CODEBASE_ROOT_PLACEHOLDER} after substitution: "
                f"{cwd_substituted!r}. Internal error."
            )

        # AUTO-INJECT --app-layer-version when launch.json doesn't carry the
        # ${APP_LAYER_VERSION} placeholder.
        #
        # The substitution loop above only writes a value INTO an existing
        # ${APP_LAYER_VERSION} token; if launch.json's script_args predates
        # the placeholder (legacy v1 setups), the captured-from-build or
        # user-typed value would be silently dropped. The script's argparse
        # would see ``app_layer_version=None`` and fall back to libfb's
        # ``fbpkg.get_metadata("fire-app")`` lookup — which returns whatever
        # registered fbpkg libfb knows about (observed: a stale ``:707``).
        # FBLearner then runs the WRONG fbpkg without ever raising.
        #
        # This branch closes the gap: when we have a value AND it's not
        # already present in script_args, append the explicit pair so the
        # spawn always carries the user's intended version. New launch.json
        # files that already include the placeholder skip this branch.
        if self.app_layer_version and "--app-layer-version" not in substituted_args:
            substituted_args.extend(["--app-layer-version", self.app_layer_version])
            logger.info(
                "Auto-injected --app-layer-version=%s (launch.json had no "
                "${APP_LAYER_VERSION} placeholder; legacy-setup compat path)",
                self.app_layer_version,
            )

        # Delegate to the chosen launcher. It owns any wrapper-specific
        # install side effects + cmd construction + cwd defaulting.
        cmd, cwd = launcher.build_command(
            script_path=self.script_path,
            codebase_root=codebase_root,
            script_args=substituted_args,
            cwd_override=cwd_substituted,
        )
        return cmd, cwd

    # ------------------------------------------------------------------
    # Streaming + parsing
    # ------------------------------------------------------------------

    async def _stream_pipe(
        self,
        pipe: asyncio.StreamReader,
        log_file: Path,
        stream_file: Path,
        channel: str,
        result: dict[str, Any],
    ) -> None:
        """Pipe a single subprocess stream into log + stream + parser.

        Open both files ONCE per pipe — opening per-line would be O(N)
        syscalls for training output that emits thousands of lines.
        """
        with (
            log_file.open("a", encoding="utf-8") as fh,
            stream_file.open("a", encoding="utf-8") as sf,
        ):
            async for line in pipe:
                # Robust decode + line-ending normalization — strip both
                # \r and \n so FLOW_URI's `\S+` group never captures a
                # stray \r when the script uses Windows line endings.
                raw = line.decode("utf-8", errors="replace")
                line_text = raw.rstrip("\r\n") + "\n"
                fh.write(line_text)
                fh.flush()
                async with self._stream_lock:
                    sf.write(line_text)
                    sf.flush()

                # FLOW_URI: only printed on stdout per the prompt
                # contract, but parse on both pipes defensively in case
                # the script mis-routes; first match wins. Log a WARN
                # when a contract marker arrives on stderr —
                # observability for prompt drift.
                m = _FLOW_URI_RE.match(line_text)
                if m and not result.get("flow_uri"):
                    result["flow_uri"] = m.group(1)
                    if channel == "stderr":
                        logger.warning("FLOW_URI on stderr (script contract drift)")
                    await self._emit_event(
                        {
                            "type": "submission_state",
                            "status": "running",
                            "flowUri": result["flow_uri"],
                            "experimentId": _extract_flow_id(result["flow_uri"]),
                        }
                    )
                m2 = _MAST_JOB_RE.match(line_text)
                if m2 and not result.get("mast_job"):
                    result["mast_job"] = m2.group(1)
                    if channel == "stderr":
                        logger.warning("MAST_JOB on stderr (script contract drift)")
                    await self._emit_event(
                        {
                            "type": "submission_state",
                            "mastJob": result["mast_job"],
                        }
                    )

                # Mode B local-runner LOCAL_RUN marker. Stamps
                # runMode='local' (+ runHost/runLogPath when extractable)
                # so JobMonitorView's local-run framing block lights up.
                # Lenient — emit even if pid/host extraction fails.
                m3 = _LOCAL_RUN_RE.match(line_text)
                if m3 and not self._local_run_emitted:
                    _pid, host, ws_path, _attempt = m3.groups()
                    payload: dict[str, Any] = {
                        "type": "submission_state",
                        "runMode": "local",
                    }
                    if host:
                        payload["runHost"] = host
                    if ws_path:
                        payload["runLogPath"] = f"{ws_path}/logs/training.log"
                    await self._emit_event(payload)
                    self._local_run_emitted = True

                # STATUS marker — per-epoch live trajectory updates.
                # Forgiving key=value parser; only `epoch=N` triggers an
                # emit (other STATUS lines are forensic-only).
                m4 = _STATUS_RE.match(line_text)
                if m4:
                    kv = dict(_STATUS_KV_RE.findall(m4.group(1)))
                    if "epoch" in kv:
                        try:
                            epoch_int = int(kv["epoch"])
                        except ValueError:
                            continue
                        epoch_row: dict[str, Any] = {"epoch": epoch_int}
                        for k in ("ndcg10", "hr10", "mrr"):
                            if k in kv:
                                try:
                                    epoch_row[k] = float(kv[k])
                                except ValueError:
                                    pass
                        # Sliding window: keep last 100 rows to bound payload
                        self._epoch_trajectory.append(epoch_row)
                        if len(self._epoch_trajectory) > 100:
                            self._epoch_trajectory = self._epoch_trajectory[-100:]
                        await self._emit_event(
                            {
                                "type": "submission_state",
                                "epochsCompleted": epoch_int,
                                "epochTrajectory": list(self._epoch_trajectory),
                            }
                        )

    # ------------------------------------------------------------------
    # Auto-build phase (runs BEFORE the main spawn when configured)
    # ------------------------------------------------------------------

    async def _run_build_phase(self, stream_file: Path) -> str:
        """Execute ``self.build_command`` (the user's referenceCommand)
        and return the captured fbpkg version (e.g. ``fire-app:2941a32``).

        Streams app-layer's stdout+stderr into ``stream_file`` so the
        user sees build progress live in the run subtab — same channel
        the main spawn writes to. Tail buffers per pipe (~4 KB) feed
        actionable error messages on failure.

        Cancellation matches the main-spawn pattern: the build subprocess
        is held in ``self._build_proc`` so ``self.cancel()`` can SIGTERM
        / SIGKILL it; ``run()`` wraps the call with the same
        ``CancelledError -> asyncio.shield(cancel())`` block.

        Raises ``BuildFailed`` on any of:
          - empty / unparseable ``build_command``
          - cmd[0] not in ``_BUILD_CMD_ALLOWLIST``
          - shell metachar in any substituted token
          - subprocess exit code != 0
          - subprocess exit code 0 but no ``Built FBPKG: <ver>`` match.
        Raises ``LaunchValidationError`` for codebase-root resolution
        failures (matches the main spawn's behavior).
        """
        # Tokenize first — this is shape validation only, no execution.
        # ``_tokenize_build_command`` itself raises ``BuildFailed`` on
        # empty / multi-&& / non-cd inputs, which we propagate.
        raw_cmd, raw_cwd = _tokenize_build_command(self.build_command)

        # cmd[0] allowlist BEFORE substitution — catches ``rm`` / ``sh``
        # / etc. in the bare command, which is the kind of thing a user
        # might paste accidentally and which would never be a legitimate
        # builder.
        if raw_cmd[0] not in _BUILD_CMD_ALLOWLIST:
            raise LaunchValidationError(
                f"build cmd[0] {raw_cmd[0]!r} not in allowlist "
                f"{set(_BUILD_CMD_ALLOWLIST)}. Edit the SetupWizard's "
                "Reference command to start with one of these binaries, "
                "or supply --app-layer-version manually in the modal."
            )

        # Resolve ${CODEBASE_ROOT} once (only if any token actually uses
        # it — same backward-compat reasoning as the main spawn path).
        needs_root = _CODEBASE_ROOT_PLACEHOLDER in raw_cwd or any(
            _CODEBASE_ROOT_PLACEHOLDER in t for t in raw_cmd
        )
        codebase_root = self._resolve_codebase_root() if needs_root else ""

        # Substitute + per-token metachar scrub. We never invoke a shell,
        # but a user-typed reference command containing ``$()`` /
        # backticks / pipes is almost certainly an error we should
        # surface rather than silently exec.
        substituted_cmd: list[str] = []
        for t in raw_cmd:
            v = t.replace(_CODEBASE_ROOT_PLACEHOLDER, codebase_root)
            for meta in _SHELL_METACHARS:
                if meta in v:
                    raise LaunchValidationError(
                        f"build cmd token contains shell metachar "
                        f"{meta!r} after substitution: {v!r}. Reference "
                        "command must use plain argv tokens — no pipes, "
                        "redirects, or substitutions."
                    )
            if _CODEBASE_ROOT_PLACEHOLDER in v:
                raise LaunchValidationError(
                    f"build cmd token still contains "
                    f"{_CODEBASE_ROOT_PLACEHOLDER} after substitution: "
                    f"{v!r}. Set workflow_target_path on the session."
                )
            substituted_cmd.append(v)
        substituted_cwd = raw_cwd.replace(_CODEBASE_ROOT_PLACEHOLDER, codebase_root)
        if _CODEBASE_ROOT_PLACEHOLDER in substituted_cwd:
            raise LaunchValidationError(
                f"build cwd still contains {_CODEBASE_ROOT_PLACEHOLDER} "
                f"after substitution: {substituted_cwd!r}"
            )

        # Surface a "Building" status BEFORE the subprocess starts so
        # the Monitor view's pill changes immediately (the build can
        # take ~49s on warm cache, minutes on cold; "Running" would
        # mislead the user into thinking the FBLearner job is up).
        await self._emit_event({"type": "submission_state", "status": "building"})

        # Header line so the user sees "what's about to happen" before
        # the (potentially noisy) app-layer output starts.
        async with self._stream_lock:
            with stream_file.open("a", encoding="utf-8") as sf:
                sf.write(
                    f"[runner] Build phase: {' '.join(substituted_cmd)}\n"
                    f"[runner] cwd: {substituted_cwd}\n"
                )
                sf.flush()

        self._build_proc = await asyncio.create_subprocess_exec(
            *substituted_cmd,
            cwd=substituted_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_MAX_STREAM_LINE_BYTES,
        )

        # Tail buffers (~4 KB each) for error-message construction. We
        # also stream both pipes to ``stream_file`` for live UI display.
        tail_bytes = 4096
        stdout_tail = bytearray()
        stderr_tail = bytearray()
        captured_version: str | None = None

        async def _drain(
            pipe: asyncio.StreamReader,
            tail: bytearray,
            channel: str,
        ) -> None:
            nonlocal captured_version
            assert pipe is not None
            async for line in pipe:
                raw = line.decode("utf-8", errors="replace")
                line_text = raw.rstrip("\r\n") + "\n"
                # Tail buffer (rolling, byte-bounded — drop from the
                # FRONT to keep the most recent bytes for error msg).
                encoded = line_text.encode("utf-8", errors="replace")
                tail.extend(encoded)
                if len(tail) > tail_bytes:
                    del tail[: len(tail) - tail_bytes]
                # Stream to the same file the main spawn writes to.
                async with self._stream_lock:
                    with stream_file.open("a", encoding="utf-8") as sf:
                        sf.write(line_text)
                        sf.flush()
                # Capture version. Last-match-wins (the line is near the
                # END of app-layer's output; some failures emit it
                # earlier as a previous-run echo).
                m = _BUILT_FBPKG_RE.search(line_text)
                if m:
                    captured_version = m.group(1)

        assert self._build_proc.stdout is not None
        assert self._build_proc.stderr is not None
        await asyncio.gather(
            _drain(self._build_proc.stdout, stdout_tail, "stdout"),
            _drain(self._build_proc.stderr, stderr_tail, "stderr"),
        )
        exit_code = await self._build_proc.wait()

        if exit_code != 0:
            tail_text = stderr_tail.decode("utf-8", errors="replace").strip()
            if not tail_text:
                tail_text = stdout_tail.decode("utf-8", errors="replace").strip()
            # Cap the surfaced tail to keep error events readable.
            if len(tail_text) > 1000:
                tail_text = "…" + tail_text[-1000:]
            raise BuildFailed(
                f"app-layer exited {exit_code}.\n--- last output ---\n{tail_text}"
            )
        if not captured_version:
            raise BuildFailed(
                "app-layer succeeded (exit 0) but produced no "
                f"'Built FBPKG: <version>' line in stdout. Captured "
                f"regex: {_BUILT_FBPKG_RE.pattern!r}"
            )
        # Header line so the user can see the captured value in-context.
        async with self._stream_lock:
            with stream_file.open("a", encoding="utf-8") as sf:
                sf.write(f"[runner] Build phase OK: {captured_version}\n")
                sf.flush()
        return captured_version

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        """Spawn the subprocess and drive it to completion.

        Returns ``{flow_uri, mast_job, exit_code}``. Raises
        ``LaunchValidationError`` if launch.json is invalid (caught by
        the caller's wrapper, which marks the run 'error' with the
        validation message).
        """
        logs_dir = self.workspace / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = logs_dir / "stdout.log"
        stderr_log = logs_dir / "stderr.log"

        cache_dir = self.workspace / "_runtime" / "inferencer_cache" / "submission"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stream_file = cache_dir / "stream_run.txt"

        # Auto-build phase: when the user left the App-layer version blank
        # AND the setup carries a referenceCommand, run the build first
        # and capture the freshly-built ``fire-app:<sha>`` version.
        # Build failures short-circuit before any submit_v1.py spawn —
        # blame attribution stays sharp ("Build failed: ..." vs an
        # opaque FBLearner-side error from a stale fbpkg).
        if not self.app_layer_version and self.build_command:
            try:
                self.app_layer_version = await self._run_build_phase(stream_file)
            except (BuildFailed, LaunchValidationError) as e:
                # Surface the failure as a terminal submission_state. The
                # caller reads the result dict and emits its own
                # task_status terminal event; we only own the
                # submission_state here.
                err_msg = f"Build failed: {e}"
                async with self._stream_lock:
                    with stream_file.open("a", encoding="utf-8") as sf:
                        sf.write(f"[runner] {err_msg}\n")
                        sf.write(STREAM_FAIL_MARKER + "\n")
                        sf.flush()
                await self._emit_event(
                    {
                        "type": "submission_state",
                        "status": "error",
                        "error": err_msg,
                    }
                )
                return {
                    "flow_uri": None,
                    "mast_job": None,
                    "exit_code": 1,
                    "error": err_msg,
                }
            except asyncio.CancelledError:
                # User clicked Cancel during the build — kill app-layer,
                # never proceed to spawn submit_v1.py. Same shield
                # pattern as the main spawn's CancelledError handler.
                await asyncio.shield(self.cancel())
                raise
        elif self.app_layer_version and self.build_command:
            # Override path: user typed a value, skip the build. Logged
            # so the run subtab shows what happened (otherwise users
            # wonder why "Building" never appeared).
            logger.info("app_layer_version override provided; skipping build phase")
            async with self._stream_lock:
                with stream_file.open("a", encoding="utf-8") as sf:
                    sf.write(
                        f"[runner] Skipping build phase — using user-typed "
                        f"--app-layer-version {self.app_layer_version!r}\n"
                    )
                    sf.flush()

        # Validate launch BEFORE the workspace tailer attaches — early
        # failure must surface in the run task subtab, not silently
        # leave a half-spawned process behind.
        self.cmd, self.cwd = self._build_launch_command()

        self._proc = await asyncio.create_subprocess_exec(
            *self.cmd,
            cwd=self.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_MAX_STREAM_LINE_BYTES,
        )

        result: dict[str, Any] = {
            "flow_uri": None,
            "mast_job": None,
            "exit_code": None,
        }
        try:
            assert self._proc.stdout is not None
            assert self._proc.stderr is not None
            await asyncio.gather(
                self._stream_pipe(
                    self._proc.stdout,
                    stdout_log,
                    stream_file,
                    "stdout",
                    result,
                ),
                self._stream_pipe(
                    self._proc.stderr,
                    stderr_log,
                    stream_file,
                    "stderr",
                    result,
                ),
            )
            result["exit_code"] = await self._proc.wait()
            # Surface the latest STATUS:-derived metrics row to the caller
            # so the submission-run path can run the post-completion
            # auto-analysis without re-loading hub_<mid>_submissions.json.
            # final_metrics is the LAST entry of self._epoch_trajectory
            # (built in _stream_pipe) — i.e. the metrics from the final
            # epoch the subprocess emitted. None when no STATUS: lines
            # were parsed (e.g. FBLearner runs that don't emit them).
            if self._epoch_trajectory:
                result["final_metrics"] = dict(self._epoch_trajectory[-1])
                result["epoch_trajectory"] = list(self._epoch_trajectory)
            # Marker is a CONTENT BOUNDARY: tells the tailer "everything
            # before this is real content; truncate after it" so any
            # producer-written debug bytes after stream-end don't leak
            # into the frontend. The tailer's loop only exits on an
            # explicit ``tailer.stop()`` call (handled by the caller in
            # the submission-run finally) — both are needed.
            marker = (
                STREAM_DONE_MARKER if result["exit_code"] == 0 else STREAM_FAIL_MARKER
            )
            async with self._stream_lock:
                with stream_file.open("a", encoding="utf-8") as sf:
                    sf.write(marker + "\n")
                    sf.flush()
        except asyncio.CancelledError:
            # The task-cancel handler calls .cancel() on the queue
            # entry's asyncio.Task; that surfaces here as
            # CancelledError. Without this except block the subprocess
            # would be orphaned — CancelledError would unwind back
            # through the caller before any termination call.
            #
            # Wrap with asyncio.shield so the cancel() coroutine itself
            # cannot be cancelled mid-cleanup (which would re-orphan
            # the subprocess in the very case the block exists to
            # prevent).
            await asyncio.shield(self.cancel())
            raise
        return result

    async def cancel(self) -> None:
        """Terminate any running subprocess (build or main spawn).
        Idempotent. Both ``self._build_proc`` and ``self._proc`` are
        candidates — at most one is alive at any moment because the
        build phase runs to completion before the main spawn begins,
        but cancel during either path must work.
        """
        self._cancelled = True
        for proc in (self._build_proc, self._proc):
            if proc is None or proc.returncode is not None:
                continue
            try:
                proc.terminate()
            except ProcessLookupError:
                continue
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2)
                except asyncio.TimeoutError:
                    logger.error(
                        "SubmissionRunner.cancel: subprocess did not exit "
                        "within 7s after SIGTERM+SIGKILL"
                    )
