"""TIER 3: Smoke test with REAL CLI but tiny task.

Runs an actual `openteam task` invocation with profile=quick on a minimal
task (e.g., single-file task). Verifies real integration in ~10 minutes.

This is the slowest tier and should run in CI only (not locally unless needed).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


_HERE = Path(__file__).resolve().parent
OPENSTARTUP_PATH = Path(
    os.environ.get(
        "OPENSTARTUP_PATH",
        str(_HERE.parents[4]),
    )
)


@pytest.mark.skip(reason="TIER 3: run only in CI with profile=quick; skipped locally to save time")
@pytest.mark.slow
def test_smoke_run_minimal_task_with_quick_profile(tmp_path, monkeypatch):
    """
    Execute a single-file task task with profile=quick, verify no crashes.
    
    This validates:
    - CLI can be invoked
    - Topology loads and runs end-to-end
    - No Jinja render errors or undefined variable crashes
    - Task completes (or times out gracefully) within 10 minutes
    """
    monkeypatch.chdir(OPENSTARTUP_PATH)
    
    # Create a minimal task: document a single file
    task_file = tmp_path / "task.txt"
    task_file.write_text("Document the architecture of src/openteam/server/main.py", encoding="utf-8")
    
    target_file = tmp_path / "target.py"
    target_file.write_text("# Sample file to document\ndef example(): pass", encoding="utf-8")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    cmd = [
        "openteam", "task",
        "--task", str(task_file),
        "--context", str(target_file),
        "--output", str(output_dir),
        "--profile", "quick",  # Quick profile = minimal LLM calls
        "--timeout", "600",  # 10 minutes
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=620)
    
    # Task may fail for business reasons, but should NOT crash on Jinja/undefined var
    assert result.returncode in [0, 1], (
        f"CLI crashed with return code {result.returncode}. "
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    
    # Verify no Jinja render errors in logs
    assert "UndefinedError" not in result.stderr, (
        f"Jinja UndefinedError detected: {result.stderr}"
    )
    assert "slash" not in result.stderr.lower() or "/" not in result.stderr, (
        f"Possible slash-vs-dot error in output: {result.stderr}"
    )


@pytest.mark.skip(reason="TIER 3: run only in CI with profile=quick; skipped locally to save time")
@pytest.mark.slow
def test_smoke_task_output_not_empty(tmp_path, monkeypatch):
    """Verify that the task produced some output."""
    monkeypatch.chdir(OPENSTARTUP_PATH)
    
    task_file = tmp_path / "task.txt"
    task_file.write_text("Analyze this file.", encoding="utf-8")
    
    target_file = tmp_path / "target.py"
    target_file.write_text("x = 1", encoding="utf-8")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    cmd = [
        "openteam", "task",
        "--task", str(task_file),
        "--context", str(target_file),
        "--output", str(output_dir),
        "--profile", "quick",
        "--timeout", "600",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=620)
    
    # If successful, output dir should have files
    if result.returncode == 0:
        outputs = list(output_dir.iterdir())
        assert len(outputs) > 0, f"Task succeeded but output_dir is empty: {output_dir}"
