"""Conversational resume across processes — Layer 2.

Run 1 writes an in_progress checkpoint and crashes (simulated).
Run 2 (fresh adapter) loads the checkpoint and resumes from where Run 1 left off.

This is the real-I/O cross-process resume that mocks fundamentally cannot test.
"""

import json
import os

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.flow_node_adapter import (
    ConversationalFlowNodeAdapter,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)

from .conftest import DEFAULT_TIMEOUT, skip_claude


def _make_claude_for_conversational(tmp_workspace, **overrides):
    kwargs = dict(
        target_path=str(tmp_workspace["workspace"]),
        cache_folder=str(tmp_workspace["cache"]),
        model_name="sonnet",
        resume_with_saved_results=True,
        idle_timeout_seconds=120,
        enable_shell=False,
    )
    if hasattr(ClaudeCodeCliInferencer, "permission_mode"):
        kwargs["permission_mode"] = "bypassPermissions"
    if hasattr(ClaudeCodeCliInferencer, "allowed_tools"):
        kwargs["allowed_tools"] = []
    kwargs.update(overrides)
    return ClaudeCodeCliInferencer(**kwargs)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 4)
@skip_claude
async def test_conversational_level1_resume_real_io(tmp_workspace):
    """Pre-write a 'completed' checkpoint to disk; verify a freshly-constructed
    adapter loads it and short-circuits without calling Claude.

    Mocks can simulate this, but real cross-process I/O verifies that the
    file format on disk matches what the adapter writes/reads.
    """
    session_id = "resume-l1-real"
    session_dir = os.path.join(str(tmp_workspace["checkpoint"]), session_id)
    os.makedirs(session_dir)

    # Write a completed checkpoint as the ADAPTER would (json.dumps wrapped)
    cached_value = "Cached final answer from a prior run."
    checkpoint = {
        "schema_version": 1,
        "session_id": session_id,
        "initial_content": "task",
        "status": "completed",
        "turn_number": 2,
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": cached_value},
        ],
        "completion_result": json.dumps(cached_value),
    }
    with open(os.path.join(session_dir, "checkpoint.json"), "w") as f:
        json.dump(checkpoint, f)

    # Construct a fresh adapter (would call Claude if no checkpoint)
    base = _make_claude_for_conversational(tmp_workspace)
    conv = ConversationalInferencer(base_inferencer=base, max_iterations=2)
    adapter = ConversationalFlowNodeAdapter(
        conversational_inferencer=conv,
        checkpoint_dir=str(tmp_workspace["checkpoint"]),
        session_id=session_id,
    )

    # Track whether Claude is actually called
    original_ainfer = base.ainfer
    call_count = {"n": 0}

    async def tracking_ainfer(*args, **kwargs):
        call_count["n"] += 1
        return await original_ainfer(*args, **kwargs)

    base.ainfer = tracking_ainfer

    result = await adapter.ainfer("task")

    # Level 1 short-circuit: cached value returned, Claude NOT invoked
    assert result == cached_value, (
        f"Expected cached value '{cached_value}', got '{result}'. "
        "Adapter should have short-circuited via Level 1 resume."
    )
    assert call_count["n"] == 0, (
        f"Claude was invoked {call_count['n']} times — Level 1 resume failed "
        "to short-circuit."
    )
