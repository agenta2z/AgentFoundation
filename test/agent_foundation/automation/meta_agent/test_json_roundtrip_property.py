"""Property test for ActionGraph JSON serialization round-trip (Property 18).

**Validates: Requirements 7.1, 7.3**

For any valid synthesized ActionGraph, serializing to JSON then deserializing
SHALL produce an equivalent ActionGraph (same number of nodes, same action
types in each node, same number of actions per node).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from agent_foundation.automation.schema.action_graph import ActionGraph
from agent_foundation.automation.schema.action_metadata import ActionMetadataRegistry


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Action types that are registered by default and safe for serialization
SERIALIZABLE_ACTION_TYPES = ["click", "input_text", "visit_url", "scroll", "wait"]

# Simple string targets for click/input_text/scroll
simple_target_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "Nd"), whitelist_characters="-_#."),
    min_size=1,
    max_size=20,
)

# Args strategies per action type
args_for_type = {
    "click": st.just(None),
    "input_text": st.fixed_dictionaries({"text": st.text(min_size=1, max_size=30)}),
    "visit_url": st.just(None),
    "scroll": st.one_of(
        st.just(None),
        st.fixed_dictionaries({"direction": st.sampled_from(["up", "down"])}),
    ),
    "wait": st.just(None),
}


@st.composite
def action_spec(draw):
    """Generate a single (action_type, target, args, wait) tuple for graph.action()."""
    action_type = draw(st.sampled_from(SERIALIZABLE_ACTION_TYPES))

    if action_type == "wait":
        # wait actions use a numeric target (seconds) and no element target
        wait_val = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False))
        return action_type, None, None, wait_val
    elif action_type == "visit_url":
        url = draw(st.sampled_from([
            "https://example.com",
            "https://test.org/page",
            "https://demo.io",
        ]))
        return action_type, url, None, None
    else:
        target = draw(simple_target_strategy)
        args = draw(args_for_type[action_type])
        return action_type, target, args, None


@st.composite
def action_graph_actions(draw):
    """Generate a list of action specs to build an ActionGraph."""
    return draw(st.lists(action_spec(), min_size=1, max_size=8))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_graph(action_specs, executor):
    """Build an ActionGraph from a list of action specs."""
    graph = ActionGraph(
        action_executor=executor,
        action_metadata=ActionMetadataRegistry(),
    )
    for action_type, target, args, wait in action_specs:
        graph.action(action_type, target=target, args=args, wait=wait)
    return graph


def _collect_actions(graph):
    """Collect (action_type, target, args) from all nodes in a graph."""
    result = []
    for node in graph._nodes:
        for action in node._actions:
            result.append((action.type, action.target, action.args))
    return result


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(specs=action_graph_actions())
def test_json_round_trip_preserves_node_count(specs):
    """Property 18: JSON serialize → deserialize preserves the number of nodes.

    **Validates: Requirements 7.1, 7.3**
    """
    executor = MagicMock()
    original = _build_graph(specs, executor)

    json_str = original.serialize(output_format="json")
    restored = ActionGraph.deserialize(
        json_str,
        output_format="json",
        action_executor=executor,
        action_metadata=ActionMetadataRegistry(),
    )

    assert len(restored._nodes) == len(original._nodes)


@settings(max_examples=100)
@given(specs=action_graph_actions())
def test_json_round_trip_preserves_action_types(specs):
    """Property 18: JSON serialize → deserialize preserves action types per node.

    **Validates: Requirements 7.1, 7.3**
    """
    executor = MagicMock()
    original = _build_graph(specs, executor)

    json_str = original.serialize(output_format="json")
    restored = ActionGraph.deserialize(
        json_str,
        output_format="json",
        action_executor=executor,
        action_metadata=ActionMetadataRegistry(),
    )

    original_actions = _collect_actions(original)
    restored_actions = _collect_actions(restored)

    assert len(restored_actions) == len(original_actions)
    for (orig_type, _, _), (rest_type, _, _) in zip(original_actions, restored_actions):
        assert rest_type == orig_type


@settings(max_examples=100)
@given(specs=action_graph_actions())
def test_json_round_trip_produces_valid_json(specs):
    """Property 18: Serialized output is always valid JSON with expected structure.

    **Validates: Requirements 7.1, 7.3**
    """
    executor = MagicMock()
    graph = _build_graph(specs, executor)

    json_str = graph.serialize(output_format="json")
    parsed = json.loads(json_str)

    assert "nodes" in parsed
    assert "version" in parsed
    assert isinstance(parsed["nodes"], list)
    assert len(parsed["nodes"]) == len(graph._nodes)


@settings(max_examples=100)
@given(specs=action_graph_actions())
def test_json_round_trip_preserves_action_count_per_node(specs):
    """Property 18: Each node retains the same number of actions after round-trip.

    **Validates: Requirements 7.1, 7.3**
    """
    executor = MagicMock()
    original = _build_graph(specs, executor)

    json_str = original.serialize(output_format="json")
    restored = ActionGraph.deserialize(
        json_str,
        output_format="json",
        action_executor=executor,
        action_metadata=ActionMetadataRegistry(),
    )

    for orig_node, rest_node in zip(original._nodes, restored._nodes):
        assert len(rest_node._actions) == len(orig_node._actions)
