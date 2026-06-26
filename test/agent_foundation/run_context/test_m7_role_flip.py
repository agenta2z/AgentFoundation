"""M7 read-flip: under a context, switch_role does NOT mutate self; the role flows
through ctx.node.call and the render pipeline reads it via _effective_role()."""

from attr import attrs

from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from agent_foundation.common.inferencers.run_context import (
    RoleState,
    RunContext,
    enter_run,
    exit_run,
)


@attrs
class _Templated(TemplatedInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kw):
        return "x"


def test_switch_role_under_context_does_not_mutate_self():
    t = _Templated(template_key="default")
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        t.switch_role("reviewer", template_key="review")
        # Separation: the instance definition is NOT mutated under a context...
        assert t.template_key == "default"
        # ...the role flows through the context node instead.
        call = root._store.node("/").call
        assert isinstance(call, RoleState)
        assert call.template_key == "review"
        # ...and the render pipeline resolves the EFFECTIVE role from the context.
        eff_key, _root, _master = t._effective_role()
        assert eff_key == "review"
    finally:
        exit_run(tok)
    # Outside the context, _effective_role falls back to the instance default.
    assert t._effective_role()[0] == "default"


def test_switch_role_without_context_is_byte_identical_legacy():
    t = _Templated(template_key="default")
    # No active context -> legacy behavior: self IS mutated.
    t.switch_role("reviewer", template_key="review")
    assert t.template_key == "review"
    assert t._effective_role()[0] == "review"


def test_two_branches_get_isolated_roles_on_one_instance():
    """One shared instance, two child contexts -> independent roles (the M7 payoff)."""
    t = _Templated(template_key="default")
    root = RunContext.root(workspace=None)
    for ctx, role_key in ((root.child("review"), "review"), (root.child("fix"), "followup")):
        tok = enter_run(ctx)
        try:
            t.switch_role(role_key, template_key=role_key)
            assert t._effective_role()[0] == role_key
        finally:
            exit_run(tok)
    # The instance was never mutated; each branch kept its own role.
    assert t.template_key == "default"
    for ctx, role_key in ((root.child("review"), "review"), (root.child("fix"), "followup")):
        tok = enter_run(ctx)
        try:
            assert t._effective_role()[0] == role_key
        finally:
            exit_run(tok)
