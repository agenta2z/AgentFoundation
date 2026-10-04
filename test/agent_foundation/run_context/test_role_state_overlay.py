"""A ctx-scoped ``switch_role`` overlays every template attribute it sets (plan v8
§5.9, B16 / B16b, P5 c1).

``RoleState`` carries one typed field per attribute ``switch_role`` can set, and
``_effective_role_state`` resolves each from the active node's ``RoleState``, else
the definition. A role under a host ctx therefore renders exactly like the same
role switched with no ctx, without touching the definition; the role's template
*version* never lands in the *master* slot (B16b). Old stores carrying those
attributes only in ``changes`` load into the typed fields. A host ``switch_role``
audits into the node's bounded provenance; with no ctx or under a legacy root
the instance audit trail applies.
"""

from __future__ import annotations

import json

import pytest
from agent_foundation.common.inferencers.run_context import (
    decode_state,
    encode_state,
    enter_run,
    exit_run,
    mint_root,
    RoleState,
    RunContext,
    RunStateStore,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrs


class _Templates:
    """Renders the key, root, master version and the feed it was given."""

    def __call__(
        self, key, *, active_template_root_space=None, master_version=None, **feed
    ):
        shown = {k: v for k, v in sorted(feed.items()) if not k.startswith("enable_")}
        return f"{active_template_root_space}|{key}|{master_version}|{shown}"


@attrs(slots=False)
class _Leaf(TemplatedInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return inference_input


def _leaf():
    leaf = _Leaf()
    leaf.template_manager = _Templates()
    leaf.template_root_space = "plan"
    leaf.template_key = "initial"
    leaf.template_master_version = "own_master"
    leaf.template_version = "own_version"
    leaf.modes = {}
    return leaf


ROLE = {
    "template_key": "review",
    "template_root_space": "review",
    "template_master_version": "research_propose",
    "template_version": "v2",
    "template_variables": {"task_instructions": "research_propose"},
    "template_extra_feed": {"panel": "A"},
    "modes": {"deep_mode": True},
}


def _under(ctx, fn):
    token = enter_run(ctx)
    try:
        return fn()
    finally:
        exit_run(token)


def _definition(leaf):
    return {name: getattr(leaf, name) for name in ROLE}


def test_a_host_role_renders_like_the_same_role_switched_with_no_ctx():
    bare, host_leaf, ctx = _leaf(), _leaf(), RunContext.root().child("review")
    bare.switch_role("reviewer", **ROLE)
    before = _definition(host_leaf)
    _under(ctx, lambda: host_leaf.switch_role("reviewer", **ROLE))
    expected = bare._render_prompt("the request")
    assert _under(ctx, lambda: host_leaf._render_prompt("the request")) == expected
    assert "research_propose" in expected and "'panel': 'A'" in expected
    assert _definition(host_leaf) == before
    assert host_leaf._render_prompt("the request") != expected


def test_the_role_version_never_lands_in_the_master_slot():
    leaf, ctx = _leaf(), RunContext.root().child("review")
    _under(ctx, lambda: leaf.switch_role("reviewer", template_version="v2"))
    key, root, master = _under(ctx, leaf._effective_role)
    role = _under(ctx, leaf._effective_role_state)
    assert (key, root, master) == ("initial", "plan", "own_master")
    assert role.template_version == "v2"


def test_unset_role_fields_keep_the_definition():
    leaf, ctx = _leaf(), RunContext.root().child("review")
    _under(ctx, lambda: leaf.switch_role("reviewer", template_key="review"))
    role = _under(ctx, leaf._effective_role_state)
    assert (role.template_key, role.template_root_space) == ("review", "plan")
    assert role.template_master_version == "own_master"


# -- the store ----------------------------------------------------------------


def test_role_state_round_trips_every_typed_field():
    state = RoleState(new_role="reviewer", **ROLE)
    assert decode_state(json.loads(json.dumps(encode_state(state)))) == state


def test_an_old_store_with_the_attributes_in_changes_loads_into_typed_fields():
    saved = {
        "_state_class": "RoleState",
        "new_role": "reviewer",
        "template_key": "review",
        "template_root_space": None,
        "template_version": None,
        "modes": None,
        "changes": {
            "template_key": "review",
            "template_master_version": "research_propose",
            "template_variables": {"task_instructions": "research_propose"},
            "template_extra_feed": {"panel": "A"},
        },
    }
    store = RunStateStore.from_json(
        {"nodes": {"/review": {"path": "/review", "role_state": saved}}}
    )
    state = store.peek("/review").role_state
    assert state.template_master_version == "research_propose"
    assert state.template_variables == {"task_instructions": "research_propose"}
    assert state.template_extra_feed == {"panel": "A"}


# -- the audit ------------------------------------------------------------------


def test_host_role_audit_goes_to_the_bounded_node_provenance():
    leaf, ctx = _leaf(), RunContext.root().child("review")
    limit = leaf._ROLE_PROVENANCE_LIMIT
    for i in range(limit + 3):
        _under(ctx, lambda i=i: leaf.switch_role(f"role{i}", template_key="review"))
    provenance = ctx.store.peek("/review").provenance
    assert len(provenance) == limit
    assert provenance[-1]["to_role"] == f"role{limit + 2}"
    assert provenance[-1]["changed"] == ["template_key"]
    assert "_role_history" not in vars(leaf)
    json.dumps(provenance)


@pytest.mark.parametrize("mode", ("legacy", "no_ctx"))
def test_bare_role_switches_keep_the_instance_audit_trail(mode):
    leaf = _leaf()
    if mode == "legacy":
        _under(mint_root(), lambda: leaf.switch_role("reviewer", template_key="review"))
        assert leaf.template_key == "initial"
    else:
        leaf.switch_role("reviewer", template_key="review")
        assert leaf.template_key == "review"
    (entry,) = leaf._role_history
    assert (entry["to_role"], entry["changes"]) == (
        "reviewer",
        {"template_key": "review"},
    )
