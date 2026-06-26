"""Example (M4): state_factory populates per-call state in the context node.

A real inferencer with ``state_factory`` set: each ``ainfer`` call builds a typed
state into ``ctx.node.call`` — the run-state lives in the context, never on the
instance. Mock leaf, no credentials. CI-gated by ``test_runcontext_examples.py``.
"""

from __future__ import annotations

import asyncio

from attr import attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    BTAState,
    RunContext,
)


@attrs
class _MockLeaf(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kw):
        return f"echo:{inference_input}"

    async def _ainfer(self, inference_input, inference_config=None, **kw):
        return f"echo:{inference_input}"


def main() -> None:
    leaf = _MockLeaf(
        # state_factory builds the node's per-call state from the input.
        state_factory=lambda inp: BTAState(effective_sub_queries=[str(inp)]),
    )
    root = RunContext.root(workspace=None)

    result = asyncio.run(leaf.ainfer("decompose this", run_context=root))
    assert result == "echo:decompose this"

    # The per-call state was built INTO the context node, not onto the instance.
    node = root._store.node("/")
    assert isinstance(node.call, BTAState)
    assert node.call.effective_sub_queries == ["decompose this"]
    assert "state_factory" not in node.to_json()  # only the state value is stored
    print("state_factory example OK:", node.call.effective_sub_queries)


if __name__ == "__main__":
    main()
