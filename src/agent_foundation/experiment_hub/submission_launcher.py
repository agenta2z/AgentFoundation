# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Launcher strategies for ``SubmissionRunner``.

Decouples the runner from launcher-specific knowledge (currently Buck)
so future launcher types — e.g. plain-python passthrough, fbpkg-based
deploys, k8s job submitters — can be added by registering a new class
without changing the runner.

``launch.json`` declares which launcher to use via the optional
``"launcher"`` field. When omitted, the runner uses
``default_launcher_name()`` (currently ``"buck_run_auto"``), preserving
backward compat with v1-era setups that predate the field.

Architectural rationale (Option 2-B in the design doc):
- Runner stays a thin process supervisor (spawn / stream / cancel).
- Launcher owns the "translate (script + args + cwd) -> spawnable cmd"
  decision plus any side effects (writing BUCK files, copying scripts,
  etc.) that that decision requires.
- New launcher = new file + one registry entry. No runner surgery.

NOT a generic strategy framework: launchers are explicitly enumerated
in ``_LAUNCHER_REGISTRY``. Adding "passthrough" or "k8s" later is a
~50-line patch to this file alone.

Meta-infra commands (``buck``/``buck2``/``fbpkg``/``app-layer``,
``meta mast.job submit``) are isolated behind the launcher registry. A
``LocalMockLauncher`` (``"local_mock"``) is provided so the hub is
exercisable without that infra: it spawns the user's script via plain
``python3`` so the runner's streaming / marker-parsing / cancel paths can
be tested end-to-end without Buck or FBLearner.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Protocol


logger: logging.Logger = logging.getLogger(__name__)


class LaunchValidationError(Exception):
    """Raised when ``launch.json`` or the chosen launcher cannot be
    validated. The runner surfaces the message to the user via the run
    task subtab so they can edit the script (drawer Save) and re-submit."""


# ----------------------------------------------------------------------
# Strategy Protocol
# ----------------------------------------------------------------------


class LauncherStrategy(Protocol):
    """Translate (script_path + script_args + cwd_override) -> spawnable
    subprocess command.

    Implementations MUST be safe to call concurrently (the runner may
    look up the same instance from multiple in-flight runs). The
    canonical implementation is stateless; share via the singleton
    pattern in ``_LAUNCHER_REGISTRY``.

    To add a new launcher:
      1. Implement this Protocol (any class with matching ``name`` +
         ``build_command`` signature satisfies it).
      2. Register a singleton instance in ``_LAUNCHER_REGISTRY`` below.
    No runner changes required.
    """

    name: str

    def build_command(
        self,
        script_path: Path,
        codebase_root: str,
        script_args: list[str],
        cwd_override: str,
    ) -> tuple[list[str], str]:
        """Return ``(cmd, cwd)`` to pass to ``asyncio.create_subprocess_exec``.

        Args:
          script_path: absolute path to the user's submission script
            (PTI's ``outputs/submit_v<n>.py``).
          codebase_root: resolved absolute path of the codebase root.
            ``""`` if launch.json had no ``${CODEBASE_ROOT}`` usage AND
            no ``cwd`` override (the runner skips resolution in that
            case to preserve backward compat with v1 hardcoded
            launch.json files).
          script_args: already-substituted, already-scrubbed script args
            (the runner handles ``${ENABLE_FLAGS}`` / ``${EXP_NAME}`` /
            ``${CODEBASE_ROOT}`` substitution + shell-metachar rejection
            BEFORE handing to the launcher).
          cwd_override: if non-empty, the substituted cwd from
            launch.json. If empty, the launcher chooses its own default
            (typically ``codebase_root``).

        Raises:
          LaunchValidationError on launcher-specific validation failures
          (e.g., script unreadable, target dir not writable).
        """
        ...


# ----------------------------------------------------------------------
# Built-in launcher: BuckRunAutoLauncher
# ----------------------------------------------------------------------

# Module-import regex: ``from a.b.c import x`` -> captures ``a.b.c``.
# Anchored to start-of-line (with optional leading whitespace) so we
# don't pick up imports inside docstrings or commented-out code.
_IMPORT_RE: re.Pattern[str] = re.compile(
    r"^\s*from\s+([a-zA-Z_][\w\.]*)\s+import\s", re.MULTILINE
)

# Heuristic allowlist: only convert imports under these top-level
# packages into buck deps. Anything else (stdlib, pypi) doesn't need a
# buck dep — fbcode python_binary auto-resolves third-party via
# autodeps. This list is intentionally small; ``arc lint -a`` on the
# generated BUCK can fill the rest if the buck build complains.
_FBCODE_DEP_PREFIXES: frozenset[str] = frozenset(
    {
        "minimal_viable_ai",
        "fblearner",
        "agent_foundation",
        "configerator",
        "libfb",
    }
)


class BuckRunAutoLauncher:
    """Auto-installs a ``python_binary`` BUCK target alongside a copy
    of the user's script under ``<codebase_root>/_agent_foundation_runtime/<hash>/``,
    then spawns ``buck run fbcode//_agent_foundation_runtime/<hash>:submit -- *args``.

    Why "auto": PTI's generated script lives in the agent's task
    workspace, NOT in fbcode. So PTI can't honestly emit a "real" buck
    target spec — ``fbcode//<pkg>:<target>`` requires ``<pkg>`` to be a
    path with an existing ``python_binary`` target, and the script's
    workspace location has neither. This launcher creates one on
    demand, keyed by SHA-1 of the script contents (so identical
    scripts hit the buck cache).

    Idempotency: same content -> same dir -> buck-out cache hit. Different
    content -> different dir -> no stale-cache risk.
    """

    name: str = "buck_run_auto"

    # Where auto-installed buck runtime artifacts live, RELATIVE to the
    # resolved codebase root. Clearly namespaced so users can hgignore
    # the whole tree without affecting anything else.
    _BUCK_RUNTIME_PREFIX: str = "_agent_foundation_runtime"

    def _extract_buck_deps(self, script_source: str) -> list[str]:
        """Best-effort: derive ``//<pkg>:<lib>`` deps from script imports.

        Convention: ``from a.b.c.d import x`` -> ``//a/b/c:d`` (the most
        common Meta python_library naming). If the actual target layout
        differs, the buck build fails with a clear "no such target"
        error and the user can fix the BUCK file via the editor drawer.

        Special case — ``libfb.py`` namespace:
          ``libfb.py`` is a *directory* (``libfb/py/``) not a module,
          so the standard last-dot-split produces ``//libfb:py`` which
          doesn't exist. Actual targets live at ``//libfb/py:<module>``.
        """
        deps: set[str] = set()
        for match in _IMPORT_RE.finditer(script_source):
            module = match.group(1)
            top = module.split(".", 1)[0]
            if top not in _FBCODE_DEP_PREFIXES:
                continue
            parts = module.split(".")
            if len(parts) < 2:
                continue

            if parts[0] == "libfb" and len(parts) >= 2 and parts[1] == "py":
                if len(parts) >= 3:
                    # ``from libfb.py.fbpkg import X`` -> ``//libfb/py:fbpkg``
                    deps.add(f"//libfb/py:{parts[2]}")
                else:
                    # ``from libfb.py import fbpkg`` -> need the imported name.
                    # Extract from the text AFTER ``import`` on the same line.
                    tail = script_source[match.end() : match.end() + 300]
                    first_line = tail.split("\n", 1)[0]
                    for name_token in first_line.split(","):
                        name = name_token.strip().split(" as ")[0].strip()
                        if name and name.isidentifier():
                            deps.add(f"//libfb/py:{name}")
                continue

            pkg = "/".join(parts[:-1])
            tgt = parts[-1]
            deps.add(f"//{pkg}:{tgt}")
        return sorted(deps)

    def _install(self, script_path: Path, codebase_root: str) -> tuple[str, Path]:
        """Copy script + emit BUCK target into a namespaced runtime dir.

        Returns ``(buck_target_spec, runtime_dir)``. Idempotent (writes
        skipped when content already matches).
        """
        try:
            script_source = script_path.read_text(encoding="utf-8")
        except OSError as e:
            raise LaunchValidationError(
                f"buck_run_auto: cannot read script {script_path}: {e}"
            ) from e

        content_hash = hashlib.sha1(script_source.encode("utf-8")).hexdigest()[:12]
        runtime_dir = Path(codebase_root) / self._BUCK_RUNTIME_PREFIX / content_hash
        try:
            runtime_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise LaunchValidationError(
                f"buck_run_auto: cannot create {runtime_dir}: {e} "
                "(check codebase root is writable + not in a read-only "
                "sparse profile)"
            ) from e

        # Write script copy. Use ``submit.py`` (fixed name) so the
        # python_binary target name is also fixed (``submit``).
        submit_py = runtime_dir / "submit.py"
        if (
            not submit_py.is_file()
            or submit_py.read_text(encoding="utf-8") != script_source
        ):
            submit_py.write_text(script_source, encoding="utf-8")

        # Build python_binary BUCK content with auto-derived deps.
        deps = self._extract_buck_deps(script_source)
        deps_block = "".join(f'        "{d}",\n' for d in deps)
        runtime_module = f"{self._BUCK_RUNTIME_PREFIX}.{content_hash}.submit"
        buck_content = (
            "# (c) Meta Platforms, Inc. and affiliates.\n"
            "# Auto-generated by BuckRunAutoLauncher. Do not edit\n"
            "# by hand; regenerate the submission setup if you need\n"
            "# different deps.\n"
            'load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")\n'
            "\n"
            'oncall("mrs_algorithms_ai4p")\n'
            "\n"
            "python_binary(\n"
            '    name = "submit",\n'
            '    srcs = ["submit.py"],\n'
            f'    main_module = "{runtime_module}",\n'
            "    deps = [\n"
            f"{deps_block}"
            "    ],\n"
            ")\n"
        )
        buck_file = runtime_dir / "BUCK"
        if (
            not buck_file.is_file()
            or buck_file.read_text(encoding="utf-8") != buck_content
        ):
            buck_file.write_text(buck_content, encoding="utf-8")

        target_spec = f"fbcode//{self._BUCK_RUNTIME_PREFIX}/{content_hash}:submit"
        logger.info(
            "buck_run_auto: target=%s deps=%d (%s)",
            target_spec,
            len(deps),
            ", ".join(deps) if deps else "<no fbcode deps detected>",
        )
        return target_spec, runtime_dir

    def build_command(
        self,
        script_path: Path,
        codebase_root: str,
        script_args: list[str],
        cwd_override: str,
    ) -> tuple[list[str], str]:
        if not codebase_root:
            # buck needs ``cwd`` inside the cell so it can find
            # .buckconfig. Without a resolved codebase_root we have
            # nowhere honest to point.
            raise LaunchValidationError(
                "buck_run_auto requires a resolved codebase_root, but "
                "launch.json had no ${CODEBASE_ROOT} usage AND no `cwd` "
                "override. Set ${CODEBASE_ROOT} in launch.json (and "
                "/set-workflow-target-path on the session) so the runner "
                "can locate the buck cell."
            )
        target_spec, runtime_dir = self._install(script_path, codebase_root)
        cmd = ["buck", "run", target_spec, "--", *script_args]
        # cwd defaults to codebase_root (where buck finds .buckconfig).
        cwd = cwd_override or codebase_root
        logger.info(
            "buck_run_auto: spawning %s (runtime_dir=%s, cwd=%s, %d args)",
            " ".join(cmd[:4]),  # buck run <target> --
            runtime_dir,
            cwd,
            len(script_args),
        )
        return cmd, cwd


# ----------------------------------------------------------------------
# FBLearner launcher
# ----------------------------------------------------------------------


class _FBLearnerLauncher:
    """Build the launch command for an FBLearner job.

    The submit script (PTI's ``outputs/submit_v<n>.py``) is expected to
    use the FBLearner SDK to define & launch a flow internally; this
    launcher just spawns the script with the same ``script_args`` the
    runner already substituted. The MAST job orchestrator picks up
    ``FLOW_URI:`` and ``MAST_JOB:`` from the script's stdout exactly the
    same way ``BuckRunAutoLauncher`` does for the local case — so the
    monitor stays backend-agnostic.

    Distinguished from ``BuckRunAutoLauncher`` only in the *cwd* default
    (FBLearner setups commonly want the script's own dir as cwd, not
    the codebase root). If a launch.json sets ``cwd`` explicitly that
    wins.

    To use: set ``"launcher": "fblearner"`` in launch.json.
    """

    name: str = "fblearner"

    def build_command(
        self,
        script_path: Path,
        codebase_root: str,
        script_args: list[str],
        cwd_override: str,
    ) -> tuple[list[str], str]:
        """Build a ``python3 <script> <args>`` invocation. The script
        itself owns the FBLearner SDK calls (mirror of the local case
        where the script owns the actual training loop).

        For team-specific MAST configs, the script's ``argparse``
        surface is the right place to add knobs — keep this launcher
        thin so it stays universal.
        """
        if not script_path.is_file():
            raise LaunchValidationError(
                f"FBLearner launcher: script_path {script_path!r} not "
                "readable. Re-generate the setup or check the path."
            )
        cwd = cwd_override or str(script_path.parent)
        # Use python3 directly so the FBLearner SDK imports resolve
        # against the calling environment's PYTHONPATH.
        cmd = ["python3", str(script_path), *script_args]
        return cmd, cwd


# ----------------------------------------------------------------------
# Local mock launcher (no Meta infra required)
# ----------------------------------------------------------------------


class LocalMockLauncher:
    """Spawn the user's script with plain ``python3`` — NO Buck/FBLearner.

    Exists so the Experiment Hub is exercisable without Meta build/launch
    infra: the runner's streaming pipeline, ``FLOW_URI:`` / ``MAST_JOB:`` /
    ``STATUS epoch=`` parsing, auto-build short-circuit, and cancel
    (SIGTERM->SIGKILL) paths can all be tested end-to-end against a small
    local Python script that prints the contract markers.

    Does NOT require a resolved ``codebase_root`` (unlike
    ``buck_run_auto``). ``cwd`` defaults to the script's own directory
    unless ``cwd_override`` is supplied; ``codebase_root`` is used as a
    fallback cwd when present.

    Select with ``"launcher": "local_mock"`` in launch.json.
    """

    name: str = "local_mock"

    def build_command(
        self,
        script_path: Path,
        codebase_root: str,
        script_args: list[str],
        cwd_override: str,
    ) -> tuple[list[str], str]:
        if not script_path.is_file():
            raise LaunchValidationError(
                f"local_mock launcher: script_path {script_path!r} not "
                "readable. Re-generate the setup or check the path."
            )
        # cwd preference: explicit override > codebase_root (if resolved) >
        # the script's own directory. No Buck cell needed.
        cwd = cwd_override or codebase_root or str(script_path.parent)
        cmd = ["python3", str(script_path), *script_args]
        logger.info(
            "local_mock: spawning python3 %s (cwd=%s, %d args)",
            script_path.name,
            cwd,
            len(script_args),
        )
        return cmd, cwd


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

# Default launcher used when launch.json omits the ``launcher`` field.
# Backward compat: existing v1-era launch.json files (no ``launcher``)
# continue to use buck_run_auto without code or schema migration.
_DEFAULT_LAUNCHER_NAME: str = "buck_run_auto"

# Add new launcher implementations here. Each entry is a singleton —
# launchers are stateless, safe to share across runs.
_LAUNCHER_REGISTRY: dict[str, LauncherStrategy] = {
    "buck_run_auto": BuckRunAutoLauncher(),
    # FBLearner backend. The launcher itself is a stateless command-
    # builder; the actual MAST/Flow submission lives inside the generated
    # submit_v<n>.py via the FBLearner SDK (just like the local case uses
    # buck run). The launcher's contribution is the right invocation + cwd.
    "fblearner": _FBLearnerLauncher(),
    # Local mock backend — exercises the runner without Meta infra.
    "local_mock": LocalMockLauncher(),
}


def get_launcher(name: str) -> LauncherStrategy:
    """Look up a launcher by name. Raises ``LaunchValidationError`` on
    unknown name (with the available set in the message so the user
    knows what to fix in the editor drawer)."""
    launcher = _LAUNCHER_REGISTRY.get(name)
    if launcher is None:
        available = sorted(_LAUNCHER_REGISTRY.keys())
        raise LaunchValidationError(
            f"unknown launcher {name!r} in launch.json. "
            f"Available launchers: {available}. Edit launch.json via "
            "the script editor drawer or Re-generate the setup."
        )
    return launcher


def default_launcher_name() -> str:
    """Default launcher used when launch.json omits the ``launcher`` field.

    Preserves backward compat with v1-era launch.json files that
    predate the field; they continue to behave as if
    ``"launcher": "buck_run_auto"`` were declared.
    """
    return _DEFAULT_LAUNCHER_NAME


def known_launcher_names() -> list[str]:
    """Return the sorted list of registered launcher names — useful for
    validators and for surfacing ``Available launchers: ...`` in error
    messages without forcing a ``LaunchValidationError`` raise+catch first."""
    return sorted(_LAUNCHER_REGISTRY.keys())
