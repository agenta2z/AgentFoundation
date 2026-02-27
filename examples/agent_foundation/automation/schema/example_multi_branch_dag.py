r"""
Example: Multi-Branch DAG with elseif Support

This example shows how to create more than 2 branches using the elseif parameter.

Multi-branch structure (grade classification):
    [set {score}]
         |
    [branch with elseif]
       /   |   \   \
      A    B    C    D
    >=90  80-89 70-79 <70

Equivalent to:
    if score >= 90: add 100 (A)
    elif score >= 80: add 80 (B)
    elif score >= 70: add 70 (C)
    else: add 60 (D)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import resolve_path  # noqa: F401 - Must be first import for path setup

from agent_foundation.automation.schema.action_graph import ActionGraph
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
# Style 1: Multi-branch using elseif parameter (lambda callbacks)
# ============================================================
def build_multi_branch_graph():
    """
    Multi-way branching using elseif parameter.

    branch(
        condition=...,      # if
        if_true=...,        # then
        elseif=[            # elif cases
            (cond, callback),
            (cond, callback),
        ],
        if_false=...,       # else
    )
    """
    return (
        ActionGraph(action_executor=executor, action_metadata=registry)
        .set(value="{score}")
        .branch(
            condition=lambda r, **kw: get_last_value(r) >= 90,
            if_true=lambda g: g.add(value=100),  # Grade A
            elseif=[
                (lambda r, **kw: get_last_value(r) >= 80, lambda g: g.add(value=80)),  # Grade B
                (lambda r, **kw: get_last_value(r) >= 70, lambda g: g.add(value=70)),  # Grade C
            ],
            if_false=lambda g: g.add(value=60),  # Grade D (else)
        )
    )


# ============================================================
# Style 2: Multi-branch using context manager with elseif()
# ============================================================
def build_context_manager_style():
    """
    Multi-way branching using context manager syntax with elseif().

    with graph.condition(...) as branch:
        with branch.if_true():
            ...
        with branch.elseif(...):
            ...
        with branch.if_false():
            ...
    """
    graph = ActionGraph(action_executor=executor, action_metadata=registry)
    graph.set(value="{score}")

    with graph.condition(lambda r, **kw: get_last_value(r) >= 90) as branch:
        with branch.if_true():
            graph.add(value=100)  # Grade A
        with branch.elseif(lambda r, **kw: get_last_value(r) >= 80):
            graph.add(value=80)   # Grade B
        with branch.elseif(lambda r, **kw: get_last_value(r) >= 70):
            graph.add(value=70)   # Grade C
        with branch.if_false():
            graph.add(value=60)   # Grade D

    return graph


# ============================================================
# Style 3: Convenience comparison methods with value_extractor
# ============================================================
def build_convenience_style():
    """
    Multi-way branching using convenience comparison methods.

    with graph.condition(value_extractor=...) as branch:
        with branch.if_gte(90):
            ...
        with branch.elseif_gte(80):
            ...
        with branch.else_():
            ...
    """
    graph = ActionGraph(action_executor=executor, action_metadata=registry)
    graph.set(value="{score}")

    with graph.condition(value_extractor=get_last_value) as branch:
        with branch.if_gte(90):
            graph.add(value=100)  # Grade A
        with branch.elseif_gte(80):
            graph.add(value=80)   # Grade B
        with branch.elseif_gte(70):
            graph.add(value=70)   # Grade C
        with branch.else_():
            graph.add(value=60)   # Grade D

    return graph


# ============================================================
# Execute with Variables
# ============================================================
if __name__ == "__main__":
    test_cases = [
        (95, "A", 195),  # 95 + 100 = 195
        (85, "B", 165),  # 85 + 80 = 165
        (75, "C", 145),  # 75 + 70 = 145
        (65, "D", 125),  # 65 + 60 = 125
        (90, "A", 190),  # boundary: 90 + 100 = 190
        (80, "B", 160),  # boundary: 80 + 80 = 160
        (70, "C", 140),  # boundary: 70 + 70 = 140
    ]

    def test_graph(name, graph):
        """Test a graph with all test cases."""
        print(f"Required variables: {graph.required_variables}")
        print()
        results = []
        all_pass = True
        for score, grade, expected in test_cases:
            executor.reset()
            graph(score=score)
            actual = executor.accumulator
            status = "OK" if actual == expected else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  Score {score:3} -> Grade {grade}: {score} + bonus = {actual:6.1f} (expected {expected}) [{status}]")
            results.append((score, actual))
        return results, all_pass

    # Style 1: Lambda callback style
    print("="*60)
    print("Style 1: Lambda Callback with elseif=[]")
    print("="*60)
    print("  graph.branch(")
    print("      condition=lambda r: score >= 90,")
    print("      if_true=lambda g: g.add(100),")
    print("      elseif=[")
    print("          (lambda r: score >= 80, lambda g: g.add(80)),")
    print("          (lambda r: score >= 70, lambda g: g.add(70)),")
    print("      ],")
    print("      if_false=lambda g: g.add(60),")
    print("  )")
    print()
    r1, p1 = test_graph("Style 1", build_multi_branch_graph())

    # Style 2: Context manager style
    print()
    print("="*60)
    print("Style 2: Context Manager with branch.elseif()")
    print("="*60)
    print("  with graph.condition(lambda r: score >= 90) as branch:")
    print("      with branch.if_true():")
    print("          graph.add(value=100)")
    print("      with branch.elseif(lambda r: score >= 80):")
    print("          graph.add(value=80)")
    print("      with branch.elseif(lambda r: score >= 70):")
    print("          graph.add(value=70)")
    print("      with branch.if_false():")
    print("          graph.add(value=60)")
    print()
    r2, p2 = test_graph("Style 2", build_context_manager_style())

    # Style 3: Convenience comparison methods
    print()
    print("="*60)
    print("Style 3: Convenience Methods (if_gte, elseif_gte, else_)")
    print("="*60)
    print("  with graph.condition(value_extractor=get_last_value) as branch:")
    print("      with branch.if_gte(90):")
    print("          graph.add(value=100)")
    print("      with branch.elseif_gte(80):")
    print("          graph.add(value=80)")
    print("      with branch.elseif_gte(70):")
    print("          graph.add(value=70)")
    print("      with branch.else_():")
    print("          graph.add(value=60)")
    print()
    r3, p3 = test_graph("Style 3", build_convenience_style())

    # Verify all styles produce identical results
    print()
    print("="*60)
    print("Verification: All styles produce identical results")
    print("="*60)
    all_match = r1 == r2 == r3
    print(f"  Style 1 results: {[r[1] for r in r1]}")
    print(f"  Style 2 results: {[r[1] for r in r2]}")
    print(f"  Style 3 results: {[r[1] for r in r3]}")
    print(f"  Results match: {all_match}")
    print(f"  All tests passed: {p1 and p2 and p3}")
    print("="*60)
