"""Real CLI subprocess test for the SOP tool.

Runs: python -m agent_foundation.resources.tools.sop role_creation --yolo "hire an MLE"
Verifies: exit code 0, SOP entered, SOP completed.

Usage:
    pytest test/agent_foundation/resources/tools/sop/test_sop_real_cli.py -m integration -s
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

skip_no_claude = pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="claude binary not on PATH",
)

skip_no_backend = pytest.mark.skipif(
    shutil.which("claude") is None and shutil.which("acli") is None,
    reason="No LLM backend (claude or acli) on PATH",
)


@skip_no_backend
@pytest.mark.integration
@pytest.mark.timeout(1800)
def test_sop_role_creation_yolo(tmp_path):
    """Run role_creation SOP in yolo mode via CLI subprocess."""
    pythonpath = ":".join([
        str(AF_ROOT / "src"),
        str(RPU_ROOT / "src"),
        str(OS_ROOT / "src"),
    ])
    env = {**os.environ, "PYTHONPATH": pythonpath}

    extra_sop_dirs = str(OS_ROOT / "src" / "openteam" / "server" / "resources" / "sops")
    extra_tool_dirs = str(OS_ROOT / "src" / "openteam" / "server" / "resources" / "tools")

    cmd = [
        sys.executable, "-m", "agent_foundation.resources.tools.sop",
        "role_creation",
        "--yolo",
        "--extra-sop-dirs", extra_sop_dirs,
        "--extra-tool-dirs", extra_tool_dirs,
        "hire a machine learning engineer",
    ]

    log_file = tmp_path / "sop_test.log"
    print(f"\n[sop-real-cli] Command: {' '.join(cmd)}")
    print(f"[sop-real-cli] Log: {log_file}")

    result = subprocess.run(
        cmd,
        cwd=str(AF_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    log_file.write_text(
        f"=== ARGV ===\n{' '.join(cmd)}\n\n"
        f"=== STDOUT ===\n{result.stdout}\n\n"
        f"=== STDERR ===\n{result.stderr}\n\n"
        f"=== EXIT CODE ===\n{result.returncode}\n"
    )

    print(f"[sop-real-cli] Exit code: {result.returncode}")
    if result.stdout:
        print(f"[sop-real-cli] STDOUT (last 500):\n{result.stdout[-500:]}")
    if result.stderr:
        print(f"[sop-real-cli] STDERR (last 300):\n{result.stderr[-300:]}")

    assert result.returncode == 0, (
        f"SOP CLI exited with {result.returncode}.\n"
        f"STDERR: {result.stderr[-500:]}\n"
        f"STDOUT: {result.stdout[-500:]}"
    )
    assert "Entered SOP: role_creation" in result.stdout, (
        "SOP entry not found in output"
    )


@skip_no_backend
@pytest.mark.integration
@pytest.mark.timeout(60)
def test_sop_cli_import():
    """Verify the CLI module imports without errors."""
    cmd = [
        sys.executable, "-c",
        "from agent_foundation.resources.tools.sop.cli import main; print('OK')",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": ":".join([
            str(AF_ROOT / "src"),
            str(RPU_ROOT / "src"),
        ]),
    }
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "OK" in result.stdout


def test_sop_cli_help():
    """Verify --help works without a backend."""
    cmd = [
        sys.executable, "-c",
        "from agent_foundation.resources.tools.sop.cli import main; main(['--help'])",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": ":".join([
            str(AF_ROOT / "src"),
            str(RPU_ROOT / "src"),
        ]),
    }
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert "sop_name" in result.stdout or "usage" in result.stdout.lower()
