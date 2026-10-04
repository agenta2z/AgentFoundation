"""Agent runs its reasoner at a ``reasoner`` child of the caller's ctx."""

import pytest
from agent_foundation.agents.agent import Agent, AgentAction, AgentResponse
from agent_foundation.agents.agent_state import AgentStateItem
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    RunContext,
)
from agent_foundation.common.inferencers.run_context.bridge import enter_run, exit_run
from agent_foundation.ui.interactive_base import InteractionFlags, InteractiveBase

PATHS = []


@pytest.fixture(autouse=True)
def _fresh_paths():
    PATHS.clear()
    yield
    PATHS.clear()


class _Interactive(InteractiveBase):
    def __init__(self, inputs):
        super().__init__()
        self._inputs = list(inputs)

    def _get_input(self):
        return self._inputs.pop(0) if self._inputs else ""

    def reset_input(self, flag: InteractionFlags) -> None:
        pass

    def _send_response(
        self, response, flag: InteractionFlags = InteractionFlags.TurnCompleted
    ) -> None:
        pass


class _PathRecordingReasoner:
    """First call fans out into ``fanout`` parallel actions; later calls complete."""

    def __init__(self, fanout=1):
        self._fanout = fanout
        self._calls = 0

    def __call__(self, reasoner_input, reasoner_config):
        ctx = active_run_context()
        PATHS.append(ctx.path if ctx else None)
        self._calls += 1
        if self._calls > 1:
            return AgentResponse(instant_response="done", next_actions=[])
        return AgentResponse(
            instant_response="acting",
            next_actions=[
                [AgentAction(type=f"Act{i}", target="t") for i in range(self._fanout)]
            ],
        )


class _Agent(Agent):
    def _parse_raw_response(self, raw_response):
        return raw_response, AgentStateItem()


def _run_agent(fanout=1):
    agent = _Agent(
        reasoner=_PathRecordingReasoner(fanout),
        actor=lambda action_type, action_target, **kwargs: f"{action_type} ok",
        interactive=_Interactive(["go"]),
        log_time=False,
        logger=None,
    )
    agent({"user_input": "go"})


def test_reasoner_runs_at_reasoner_child_of_caller_ctx():
    token = enter_run(RunContext.root(workspace=None).child("x"))
    try:
        _run_agent()
    finally:
        exit_run(token)

    assert PATHS and set(PATHS) == {"/x/reasoner"}


def test_parallel_branch_reasoner_calls_run_at_the_child_path():
    token = enter_run(RunContext.root(workspace=None).child("x"))
    try:
        _run_agent(fanout=3)
    finally:
        exit_run(token)

    assert len(PATHS) >= 4
    assert set(PATHS) == {"/x/reasoner"}


def test_caller_ctx_is_restored_after_the_agent_call():
    caller = RunContext.root(workspace=None).child("x")
    token = enter_run(caller)
    try:
        _run_agent()
        assert active_run_context() is caller
    finally:
        exit_run(token)


def test_reasoner_without_a_caller_ctx_runs_ctx_less():
    _run_agent()

    assert PATHS and set(PATHS) == {None}
    assert active_run_context() is None
