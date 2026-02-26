r"""
Example: Diamond DAG with Variable Passing

Diamond structure:
       [a: set {initial}]
              |
       [b: add {first_add}]
              |
       [c: multiply {multiplier}]
             / \
    (>50)  /     \ (<=50)
          /       \
   [d: +100]    [e: -50]
          \       /
           \     /
       [g: add {final_add}]

Variables flow from graph(...) call through all nodes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import resolve_path  # noqa: F401 - Must be first import for path setup

from science_modeling_tools.automation.schema.action_graph import ActionGraph
from arithmetic_executor import ArithmeticExecutor
from arithmetic_registry import create_arithmetic_registry


def get_last_value(result):
    """Extract last action's result value."""
    if not result.context or not result.context.results:
        return 0
    last_id = list(result.context.results.keys())[-1]
    r = result.context.results[last_id].value
    return r.value if hasattr(r, 'value') else 0


# ============================================================
# Setup
# ============================================================
executor = ArithmeticExecutor()
registry = create_arithmetic_registry()


# ============================================================
# Style 1: Verbose API (graph.action with args dict)
# ============================================================
def build_verbose_style():
    """Build graph using verbose action() API."""
    graph = ActionGraph(
        action_executor=executor,
        action_metadata=registry,
    )

    graph.action("set", args={"value": "{initial}"})
    graph.action("add", args={"value": "{first_add}"})
    graph.action("multiply", args={"value": "{multiplier}"})

    graph.branch(
        condition=lambda r, **kw: get_last_value(r) > 50,
        if_true=lambda g: g.action("add", args={"value": 100}),
        if_false=lambda g: g.action("subtract", args={"value": 50}),
    )

    graph.action("add", args={"value": "{final_add}"})
    return graph


# ============================================================
# Style 2: Fluent API with chaining and if-condition
# ============================================================
def build_fluent_style():
    """Build graph using fluent API with method chaining."""
    graph = ActionGraph(
        action_executor=executor,
        action_metadata=registry,
    )

    # Fluent chaining: graph.action_name(value=...).action_name(...)
    (graph
        .set(value="{initial}")
        .add(value="{first_add}")
        .multiply(value="{multiplier}"))

    # Pythonic condition with context manager syntax
    with graph.condition(lambda r, **kw: get_last_value(r) > 50) as branch:
        with branch.if_true():
            graph.add(value=100)
        with branch.if_false():
            graph.subtract(value=50)

    graph.add(value="{final_add}")
    return graph


# ============================================================
# Style 3: Fully chained fluent API
# ============================================================
def build_fully_chained_style():
    """Build graph using fully chained fluent API."""
    return (
        ActionGraph(action_executor=executor, action_metadata=registry)
        .set(value="{initial}")
        .add(value="{first_add}")
        .multiply(value="{multiplier}")
        .branch(
            condition=lambda r, **kw: get_last_value(r) > 50,
            if_true=lambda g: g.add(value=100),
            if_false=lambda g: g.subtract(value=50),
        )
        .add(value="{final_add}")
    )


# ============================================================
# Execute with Variables
# ============================================================
if __name__ == "__main__":
    # Test case 1: True branch (value > 50)
    true_branch_vars = dict(initial=10, first_add=5, multiplier=5, final_add=7)
    # Test case 2: False branch (value <= 50)
    false_branch_vars = dict(initial=5, first_add=5, multiplier=2, final_add=10)

    def test_graph(name, graph):
        """Test a graph with both branch paths."""
        print(f"Required variables: {graph.required_variables}")
        executor.reset()
        graph(**true_branch_vars)
        true_result = executor.accumulator
        print(f"  True branch:  10 -> +5 -> *5 = 75 (>50) -> +100 -> +7 = {true_result}")
        executor.reset()
        graph(**false_branch_vars)
        false_result = executor.accumulator
        print(f"  False branch: 5 -> +5 -> *2 = 20 (<=50) -> -50 -> +10 = {false_result}")
        return true_result, false_result

    print("="*60)
    print("Style 1: Verbose API (graph.action with args dict)")
    print("="*60)
    r1 = test_graph("Style 1", build_verbose_style())

    print("\n" + "="*60)
    print("Style 2: Fluent API with context manager")
    print("="*60)
    r2 = test_graph("Style 2", build_fluent_style())

    print("\n" + "="*60)
    print("Style 3: Fully chained fluent API")
    print("="*60)
    r3 = test_graph("Style 3", build_fully_chained_style())

    print("\n" + "="*60)
    print("Verification: All styles produce identical results")
    print("="*60)
    all_match = r1 == r2 == r3
    print(f"  Style 1: True={r1[0]}, False={r1[1]}")
    print(f"  Style 2: True={r2[0]}, False={r2[1]}")
    print(f"  Style 3: True={r3[0]}, False={r3[1]}")
    print(f"  All match: {all_match}")
