r"""
Example: Loop Construct

Demonstrates the loop() construct for iterative computation.

Loop structure:
    [set {start}]
         |
    [loop: while condition is True]
         |
         +-- advance: update state each iteration
         |
    (loop exits when condition returns False)

The loop has three components:
    - condition: callable checked before each iteration, return True to continue
    - advance: callable executed each iteration to update state
    - max_loop: safety limit prevents infinite loops

Example 1: Count to target
    Start at 0, add 1 each iteration until reaching target

Example 2: Double until threshold
    Start at 1, double each iteration until exceeding threshold
"""

import sys
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resolve_path  # noqa: F401 - Must be first import for path setup

from agent_foundation.automation.schema.action_graph import ActionGraph
from arithmetic_executor import ArithmeticExecutor
from arithmetic_registry import create_arithmetic_registry


# ============================================================
# Setup
# ============================================================
executor = ArithmeticExecutor()
registry = create_arithmetic_registry()


# ============================================================
# Example 1: Count to Target
# ============================================================
def build_count_to_target(target: int):
    """
    Build graph that counts from 0 to target.

    Graph: [set 0] -> [loop: add 1 until accumulator >= target]

    Args:
        target: Value to count up to
    """
    graph = ActionGraph(action_executor=executor, action_metadata=registry)

    def not_reached_target(result, **kwargs):
        """Continue while accumulator < target."""
        return executor.accumulator < target

    def increment(result, **kwargs):
        """Add 1 to accumulator."""
        executor.accumulator += 1
        return result

    (graph
        .set(value=0)
        .loop(
            condition=not_reached_target,
            max_loop=target + 10,  # Safety margin
            advance=increment,
        ))

    return graph


# ============================================================
# Example 2: Double Until Threshold
# ============================================================
def build_double_until_threshold(threshold: int):
    """
    Build graph that doubles value until it exceeds threshold.

    Sequence: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> ...

    Graph: [set 1] -> [loop: double until accumulator > threshold]

    Args:
        threshold: Stop when accumulator exceeds this value
    """
    graph = ActionGraph(action_executor=executor, action_metadata=registry)

    def below_threshold(result, **kwargs):
        """Continue while accumulator <= threshold."""
        return executor.accumulator <= threshold

    def double_value(result, **kwargs):
        """Double the accumulator."""
        executor.accumulator *= 2
        return result

    (graph
        .set(value=1)
        .loop(
            condition=below_threshold,
            max_loop=50,
            advance=double_value,
        ))

    return graph


# ============================================================
# Example 3: Variable Start with Counter
# ============================================================
def build_configurable_counter(target: int):
    """
    Build graph with variable start value that counts to target.

    Graph: [set {start}] -> [loop: add 1 until accumulator >= target]

    Args:
        target: Value to count up to
    """
    graph = ActionGraph(action_executor=executor, action_metadata=registry)

    def not_reached_target(result, **kwargs):
        """Continue while accumulator < target."""
        return executor.accumulator < target

    def step_forward(result, **kwargs):
        """Add 1 to accumulator."""
        executor.accumulator += 1
        return result

    (graph
        .set(value="{start}")
        .loop(
            condition=not_reached_target,
            max_loop=100,
            advance=step_forward,
        ))

    return graph


# ============================================================
# Execute Examples
# ============================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # Suppress retry warnings for cleaner output

    print("=" * 60)
    print("Example 1: Count to Target")
    print("=" * 60)
    print("Loop: start at 0, add 1 each iteration until reaching target")
    print()

    test_cases_1 = [
        (5, 5),    # count 0->1->2->3->4->5 = 5
        (10, 10),  # count to 10 = 10
        (0, 0),    # count to 0 (no iterations) = 0
        (1, 1),    # count 0->1 = 1
    ]

    for target, expected in test_cases_1:
        graph = build_count_to_target(target)
        executor.reset()
        graph()
        actual = executor.accumulator
        status = "OK" if actual == expected else "FAIL"
        print(f"  Count to {target:2}: result={actual:5.1f} (expected {expected}) [{status}]")

    print()
    print("=" * 60)
    print("Example 2: Double Until Threshold")
    print("=" * 60)
    print("Loop: start at 1, double each iteration until exceeding threshold")
    print("  1 -> 2 -> 4 -> 8 -> 16 -> 32 -> ...")
    print()

    test_cases_2 = [
        (10, 16),   # 1->2->4->8->16 (16 > 10, stop)
        (20, 32),   # 1->2->4->8->16->32 (32 > 20, stop)
        (1, 2),     # 1->2 (2 > 1, stop)
        (0, 1),     # 1 (1 > 0, condition false immediately)
    ]

    for threshold, expected in test_cases_2:
        graph = build_double_until_threshold(threshold)
        executor.reset()
        graph()
        actual = executor.accumulator
        status = "OK" if actual == expected else "FAIL"
        print(f"  Threshold={threshold:2}: result={actual:5.1f} (expected {expected}) [{status}]")

    print()
    print("=" * 60)
    print("Example 3: Variable Start Counter")
    print("=" * 60)
    print("Loop: start at {start}, add 1 until reaching target=10")
    print()

    graph3 = build_configurable_counter(target=10)
    print(f"Required variables: {graph3.required_variables}")
    print()

    test_cases_3 = [
        (0, 10),   # start=0, count to 10 = 10
        (5, 10),   # start=5, count to 10 = 10
        (10, 10),  # start=10 (already at target, no iterations) = 10
        (15, 15),  # start=15 (above target, no iterations) = 15
    ]

    for start, expected in test_cases_3:
        graph3 = build_configurable_counter(target=10)
        executor.reset()
        graph3(start=start)
        actual = executor.accumulator
        status = "OK" if actual == expected else "FAIL"
        print(f"  Start={start:2}: result={actual:5.1f} (expected {expected}) [{status}]")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("The loop() construct enables iterative computation in ActionGraph:")
    print("  - condition: checked before each iteration (return True to continue)")
    print("  - advance: executed each iteration to update state")
    print("  - max_loop: safety limit prevents infinite loops")
    print()
    print("Usage:")
    print("  graph.loop(")
    print("      condition=lambda result, **kw: <bool>,  # continue while True")
    print("      advance=lambda result, **kw: result,    # update state")
    print("      max_loop=100,                           # safety limit")
    print("  )")
