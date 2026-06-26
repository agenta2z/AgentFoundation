"""CI gate for the run_context examples (plan §5 DoD #3: promote examples to CI).

Four mock/no-credential examples, each exercising a milestone end-to-end:
  * example_runcontext_explicit          — host-mint + three tiers + provenance
  * example_runcontext_state_factory     — M4 state_factory -> ctx.node.call
  * example_runcontext_resume            — M9 persist/rehydrate
  * example_runcontext_parallel_isolation — M3/M6 per-branch isolation (V8)
"""

import importlib.util
import os

import pytest

_EXAMPLES_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(
            __import__(
                "agent_foundation.common.inferencers.inferencer_base", fromlist=["x"]
            ).__file__
        ),
        "..", "..", "..", "..",
        "examples", "agent_foundation", "common", "inferencers",
    )
)

_EXAMPLES = [
    "example_runcontext_explicit",
    "example_runcontext_state_factory",
    "example_runcontext_resume",
    "example_runcontext_parallel_isolation",
]


@pytest.mark.parametrize("name", _EXAMPLES)
def test_runcontext_example_runs(name):
    path = os.path.join(_EXAMPLES_DIR, f"{name}.py")
    assert os.path.exists(path), path
    spec = importlib.util.spec_from_file_location(f"_ex_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()  # asserts internally; raises on any failure
