"""
Test script to verify that the reasoner property setter properly sets parent_debuggable.
"""
import sys
from pathlib import Path

# Add ScienceModelingTools and SciencePythonUtils to path
project_root = Path(__file__).parent.parent.parent.parent.parent
rich_python_utils_src = project_root / "SciencePythonUtils" / "src"
agent_foundation_src = project_root / "ScienceModelingTools" / "src"

for path_item in [agent_foundation_src, rich_python_utils_src]:
    if path_item.exists() and str(path_item) not in sys.path:
        sys.path.insert(0, str(path_item))

from agent_foundation.agents.agent import Agent
from agent_foundation.common.inferencers.api_inferencers.claude_api_inferencer import ClaudeApiInferencer
from rich_python_utils.common_objects.debuggable import Debuggable

def test_reasoner_property():
    """Test that setting reasoner via property properly sets parent_debuggable."""

    # Create a simple agent subclass for testing
    class TestAgent(Agent):
        def _parse_raw_response(self, raw_response):
            return raw_response, None

    # Create an agent with initial reasoner (ClaudeApiInferencer is Debuggable)
    initial_reasoner = ClaudeApiInferencer()
    agent = TestAgent(
        reasoner=initial_reasoner
    )

    print("=" * 80)
    print("TEST 1: Initial reasoner parent_debuggable")
    print("=" * 80)
    print(f"Agent ID: {agent.id}")
    print(f"Initial reasoner ID: {initial_reasoner.id}")
    print(f"Initial reasoner parent_debuggables: {initial_reasoner.parent_debuggables}")

    # Check that parent_debuggable was set during __attrs_post_init__
    assert len(initial_reasoner.parent_debuggables) > 0, "Initial reasoner should have parent_debuggable set"
    assert agent in initial_reasoner.parent_debuggables, "Agent should be in initial reasoner's parent_debuggables"
    print("[PASS] Initial reasoner has parent_debuggable set correctly")

    # Create a new reasoner and assign it via property setter
    new_reasoner = ClaudeApiInferencer()
    print("\n" + "=" * 80)
    print("TEST 2: Setting new reasoner via property")
    print("=" * 80)
    print(f"New reasoner ID: {new_reasoner.id}")
    print(f"New reasoner parent_debuggables (before assignment): {new_reasoner.parent_debuggables}")

    # This should trigger the property setter
    agent.reasoner = new_reasoner

    print(f"New reasoner parent_debuggables (after assignment): {new_reasoner.parent_debuggables}")

    # Check that parent_debuggable was set via property setter
    assert len(new_reasoner.parent_debuggables) > 0, "New reasoner should have parent_debuggable set"
    assert agent in new_reasoner.parent_debuggables, "Agent should be in new reasoner's parent_debuggables"
    print("[PASS] New reasoner has parent_debuggable set correctly via property setter")

    # Verify that the agent's reasoner property returns the new reasoner
    print("\n" + "=" * 80)
    print("TEST 3: Verify reasoner property getter")
    print("=" * 80)
    assert agent.reasoner is new_reasoner, "Agent.reasoner should return the new reasoner"
    print(f"Agent.reasoner ID: {agent.reasoner.id}")
    print("[PASS] Property getter returns correct reasoner")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe reasoner property setter correctly maintains the debugging chain")
    print("when swapping inferencers in web_agent_service.py")

if __name__ == '__main__':
    test_reasoner_property()
