# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Shared validation helpers for `launch.json` + `submit.py` runner contracts.

Ported from RankEvolve ``launch_validation.py`` (no FastAPI in the original;
this is a faithful copy with the ``submission_launcher`` import rewritten to
the AF path).

Used by:
  - The direct-upload endpoint (`POST /submission-setup`, mode='import'):
    runs full validation pre-flight; returns a structured `ValidationReport`
    with severity-graded findings.
  - PTI's `_setup_completion_hook` (in `tool_executor.py`) — TODO migration:
    currently has duplicated inline validation; should be replaced with a
    call to `validate_launch_json` to ensure both paths gate identically.
    Until then, both implementations must stay in sync (verified by
    `test_validate_launch_json_helper.py`).
"""

from __future__ import annotations

import ast
import json
from typing import Any


# Severity ordering — used to roll up the worst finding into the top-level.
_SEVERITY_RANK: dict[str, int] = {"ok": 0, "info": 1, "warning": 2, "error": 3}


def _make_finding(severity: str, category: str, message: str) -> dict[str, Any]:
    return {"severity": severity, "category": category, "message": message}


def _rollup_severity(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "ok"
    worst = max(_SEVERITY_RANK.get(f.get("severity", "info"), 0) for f in findings)
    for k, v in _SEVERITY_RANK.items():
        if v == worst:
            return k
    return "ok"


def validate_launch_json(content: str) -> dict[str, Any]:
    """Validate launch.json content per the runner contract.

    Returns a `ValidationReport` shape:
      {"severity": "ok|info|warning|error", "findings": [{...}, ...]}

    Mirrors the inline checks in `tool_executor.py:_setup_completion_hook`
    (~lines 1185-1318). KEEP IN SYNC.
    """
    findings: list[dict[str, Any]] = []

    try:
        data = json.loads(content)
    except Exception as e:
        findings.append(
            _make_finding("error", "json_parse", f"launch.json is not valid JSON: {e}")
        )
        return {"severity": "error", "findings": findings}

    if not isinstance(data, dict):
        findings.append(
            _make_finding(
                "error",
                "schema",
                "launch.json must be a JSON object with `script_args` (list[str]) and optional `cwd` (str)",
            )
        )
        return {"severity": "error", "findings": findings}

    # script_args required + must be list of strings.
    script_args = data.get("script_args")
    if script_args is None:
        if "cmd" in data:
            findings.append(
                _make_finding(
                    "error",
                    "schema",
                    "launch.json uses the legacy `cmd`/`cwd` schema. Use the new schema: "
                    '`{"script_args": [...], "cwd": "${CODEBASE_ROOT}"}`',
                )
            )
        else:
            findings.append(
                _make_finding(
                    "error",
                    "schema",
                    "launch.json missing required `script_args` (list of strings)",
                )
            )
    elif not isinstance(script_args, list) or not all(
        isinstance(c, str) for c in script_args
    ):
        findings.append(
            _make_finding(
                "error",
                "schema",
                "launch.json `script_args` must be a list of strings",
            )
        )

    # cwd optional but if present must be non-empty string.
    cwd = data.get("cwd")
    if cwd is not None and (not isinstance(cwd, str) or not cwd):
        findings.append(
            _make_finding(
                "error",
                "schema",
                "launch.json `cwd` (when present) must be a non-empty string",
            )
        )

    # launcher field — if present must be a non-empty string + registered name.
    launcher_field = data.get("launcher")
    if launcher_field is not None:
        if not isinstance(launcher_field, str) or not launcher_field:
            findings.append(
                _make_finding(
                    "error",
                    "schema",
                    "launch.json `launcher` (when present) must be a non-empty string "
                    "identifying a known launcher (e.g. 'buck_run_auto')",
                )
            )
        else:
            try:
                from agent_foundation.experiment_hub.submission_launcher import (
                    known_launcher_names,
                )

                available = known_launcher_names()
                if launcher_field not in available:
                    findings.append(
                        _make_finding(
                            "error",
                            "launcher_unknown",
                            f"launch.json declares unknown launcher {launcher_field!r}. "
                            f"Available: {sorted(available)}",
                        )
                    )
            except Exception:
                # Lazy import failure — don't block validation; the runtime
                # will surface this if the launcher really is unknown.
                pass
    else:
        findings.append(
            _make_finding(
                "info",
                "launcher_default",
                "launch.json omits `launcher` field; runtime defaults to 'buck_run_auto'",
            )
        )

    return {"severity": _rollup_severity(findings), "findings": findings}


def validate_runner_script(content: str) -> dict[str, Any]:
    """Validate `submit.py` runner-script content for direct-upload pre-flight.

    Errors block upload; warnings are advisory ("script may not work as
    expected; the user explicitly chose to bypass PTI"). Returns a
    `ValidationReport`.

    Checks:
      - Python syntax via `ast.parse` — ERROR with line/col on failure
      - Substring presence of `FLOW_URI:` and `MAST_JOB:` print markers — WARNING
        if absent (runner won't capture flow URL / MAST job; downstream cancel +
        observability won't work)
      - Substring presence of expected CLI args (`--enable-flags`,
        `--experiment-name`, `--app-layer-version`) — WARNING if absent
    """
    findings: list[dict[str, Any]] = []

    # 1. Python syntax check.
    try:
        ast.parse(content)
    except SyntaxError as e:
        findings.append(
            _make_finding(
                "error",
                "python_syntax",
                f"Python syntax error at line {e.lineno}, col {e.offset}: {e.msg}",
            )
        )
        # Continue with substring checks — useful even if syntax fails at one site.

    # 2. Runner-contract substring presence.
    for marker in ("FLOW_URI:", "MAST_JOB:"):
        if marker not in content:
            findings.append(
                _make_finding(
                    "warning",
                    "runner_contract",
                    f"Script body does not appear to print `{marker}`. Runner won't "
                    f"capture this marker; downstream cancel + observability for it will not work.",
                )
            )

    # 3. CLI args substring presence (best-effort; doesn't AST-walk argparse).
    for arg in ("--enable-flags", "--experiment-name", "--app-layer-version"):
        if arg not in content:
            findings.append(
                _make_finding(
                    "warning",
                    "cli_arg",
                    f"Script body does not reference `{arg}`. The hub's Submit Confirm "
                    f"flow may forward this arg; the script should accept it (or accept-and-ignore).",
                )
            )

    return {"severity": _rollup_severity(findings), "findings": findings}


def merge_reports(*reports: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple `ValidationReport`s into one (combined findings + severity rollup)."""
    findings: list[dict[str, Any]] = []
    for r in reports:
        if r and isinstance(r.get("findings"), list):
            findings.extend(r["findings"])
    return {"severity": _rollup_severity(findings), "findings": findings}
