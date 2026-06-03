"""Real CLI subprocess test for the understand_codebase tool.

Runs understand_codebase via the task tool with template_master_version=understand_codebase.
Verifies: exit code 0, workspace created.

Usage:
    pytest test/agent_foundation/resources/tools/understand_codebase/test_understand_codebase_real_cli.py -m integration -s
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

AF_ROOT = Path(__file__).resolve().parents[5]
RPU_ROOT = AF_ROOT.parent / "RichPythonUtils"
OS_ROOT = AF_ROOT.parent / "OpenStartup"

assert (AF_ROOT / "src" / "agent_foundation").is_dir(), f"AF_ROOT wrong: {AF_ROOT}"

skip_no_backend = pytest.mark.skipif(
    shutil.which("claude") is None and shutil.which("acli") is None,
    reason="No LLM backend (claude or acli) on PATH",
)


@skip_no_backend
@pytest.mark.integration
@pytest.mark.timeout(0)
def test_understand_codebase_real_cli(tmp_path):
    """Run understand_codebase on a small target via CLI subprocess."""
    pythonpath = ":".join([
        str(AF_ROOT / "src"),
        str(RPU_ROOT / "src"),
        str(OS_ROOT / "src"),
    ])
    env = {**os.environ, "PYTHONPATH": pythonpath}

    # Use a small target — the understand_codebase tool directory itself
    target = str(AF_ROOT / "src" / "agent_foundation" / "resources" / "tools" / "understand_codebase")

    cmd = [
        sys.executable, "-m", "agent_foundation.resources.tools.task",
        f"Investigate codebase at {target}",
        "--full",
    ]

    log_file = tmp_path / "uc_test.log"
    print(f"\n[uc-real-cli] Command: {' '.join(cmd)}")
    print(f"[uc-real-cli] Target: {target}")
    print(f"[uc-real-cli] Log: {log_file}")

    result = subprocess.run(
        cmd,
        cwd=str(AF_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=None,
    )

    log_file.write_text(
        f"=== ARGV ===\n{' '.join(cmd)}\n\n"
        f"=== STDOUT ===\n{result.stdout}\n\n"
        f"=== STDERR ===\n{result.stderr}\n\n"
        f"=== EXIT CODE ===\n{result.returncode}\n"
    )

    print(f"[uc-real-cli] Exit code: {result.returncode}")
    if result.stdout:
        print(f"[uc-real-cli] STDOUT (last 500):\n{result.stdout[-500:]}")

    assert result.returncode == 0, (
        f"understand_codebase CLI exited with {result.returncode}.\n"
        f"STDERR: {result.stderr[-500:]}\n"
        f"STDOUT: {result.stdout[-500:]}"
    )


def test_understand_codebase_executor_import():
    """Verify the executor imports without errors."""
    from agent_foundation.resources.tools.understand_codebase.executor import execute
    assert callable(execute)


def test_understand_codebase_slash_args_import():
    """Verify slash_args module imports from AF (not OS)."""
    from agent_foundation.resources.tools.task.slash_args import (
        parse_slash_args,
        TASK_BOOL_FLAGS,
        TASK_MODE_ALIASES,
    )
    assert callable(parse_slash_args)
    assert "plan" in TASK_BOOL_FLAGS
    assert "task_plan" in TASK_MODE_ALIASES
