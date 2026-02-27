"""
Dummy WorkGraph executor for generating hierarchical logs.
"""
import sys
from pathlib import Path
from functools import partial

# Add paths for imports
# Navigate up from utils/ -> dash_interactive/ -> ui/ -> agent_foundation/ -> src/ -> ScienceModelingTools/ -> PythonProjects/
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent.parent.parent
rich_python_utils_src = project_root / "SciencePythonUtils" / "src"
if rich_python_utils_src.exists() and str(rich_python_utils_src) not in sys.path:
    sys.path.insert(0, str(rich_python_utils_src))
elif not rich_python_utils_src.exists():
    # Fallback: try to find it relative to current working directory
    fallback_path = Path.cwd().parent / "SciencePythonUtils" / "src"
    if fallback_path.exists() and str(fallback_path) not in sys.path:
        sys.path.insert(0, str(fallback_path))

from rich_python_utils.common_objects.workflow.workgraph import WorkGraphNode, WorkGraph
from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import ResultPassDownMode
from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector


def dummy_task_1(x: int) -> int:
    """Simple increment task."""
    return x + 1


def dummy_task_2(x: int) -> int:
    """Double the input."""
    return x * 2


def dummy_task_3(x: int) -> int:
    """Add 10 to input."""
    return x + 10


def dummy_task_parallel_a(x: int) -> int:
    """Parallel task A: multiply by 3."""
    return x * 3


def dummy_task_parallel_b(x: int) -> int:
    """Parallel task B: subtract 5."""
    return x - 5


def dummy_summarizer(*results) -> int:
    """Summarize parallel results by summing."""
    return sum(results)


def create_sequential_graph(log_collector: LogCollector) -> WorkGraph:
    """
    Create a sequential WorkGraph: Task1 -> Task2 -> Task3.

    Args:
        log_collector: LogCollector instance to capture logs

    Returns:
        Configured WorkGraph
    """
    # Create nodes with hierarchical logging
    node1 = WorkGraphNode(
        dummy_task_1,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    node2 = WorkGraphNode(
        dummy_task_2,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        parent_log_group_id=node1,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    node3 = WorkGraphNode(
        dummy_task_3,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        parent_log_group_id=node2,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    # Link nodes
    node1.add_next(node2)
    node2.add_next(node3)

    # Create graph
    graph = WorkGraph(
        start_nodes=[node1],
        parent_log_group_id=None,
        log_group_id="SequentialGraph",
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    return graph


def create_parallel_graph(log_collector: LogCollector) -> WorkGraph:
    """
    Create a parallel WorkGraph with branching and merging.

    Structure:
        Start -> ParallelA \\
                           -> Summarizer
              -> ParallelB /

    Args:
        log_collector: LogCollector instance to capture logs

    Returns:
        Configured WorkGraph
    """
    # Create start node
    start_node = WorkGraphNode(
        dummy_task_1,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    # Create parallel branches
    parallel_a = WorkGraphNode(
        dummy_task_parallel_a,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        parent_log_group_id=start_node,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    parallel_b = WorkGraphNode(
        dummy_task_parallel_b,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        parent_log_group_id=start_node,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    # Create summarizer node
    summary_node = WorkGraphNode(
        dummy_summarizer,
        parent_log_group_id=start_node,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    # Link nodes
    start_node.add_next(parallel_a)
    start_node.add_next(parallel_b)
    parallel_a.add_next(summary_node)
    parallel_b.add_next(summary_node)

    # Create graph
    graph = WorkGraph(
        start_nodes=[start_node],
        parent_log_group_id=None,
        log_group_id="ParallelGraph",
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    return graph


def create_complex_graph(log_collector: LogCollector) -> WorkGraph:
    """
    Create a complex WorkGraph with multiple levels and branches.

    Structure:
        Level1 -> Level2A -> Level3A
               -> Level2B -> Level3B -> Level4
                          -> Level3C /

    Args:
        log_collector: LogCollector instance to capture logs

    Returns:
        Configured WorkGraph
    """
    # Level 1
    level1 = WorkGraphNode(
        dummy_task_1,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    # Level 2
    level2a = WorkGraphNode(
        dummy_task_2,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        parent_log_group_id=level1,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    level2b = WorkGraphNode(
        dummy_task_3,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        parent_log_group_id=level1,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    # Level 3
    level3a = WorkGraphNode(
        dummy_task_parallel_a,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        parent_log_group_id=level2a,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    level3b = WorkGraphNode(
        dummy_task_parallel_b,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        parent_log_group_id=level2b,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    level3c = WorkGraphNode(
        dummy_task_1,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
        parent_log_group_id=level2b,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    # Level 4 - merge
    level4 = WorkGraphNode(
        dummy_summarizer,
        parent_log_group_id=level2b,
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    # Link nodes
    level1.add_next(level2a)
    level1.add_next(level2b)
    level2a.add_next(level3a)
    level2b.add_next(level3b)
    level2b.add_next(level3c)
    level3b.add_next(level4)
    level3c.add_next(level4)

    # Create graph
    graph = WorkGraph(
        start_nodes=[level1],
        parent_log_group_id=None,
        log_group_id="ComplexGraph",
        logger=log_collector,
        always_add_logging_based_logger=False,
        debug_mode=True,
        log_time=False
    )

    return graph


def simulate_graph_logs(graph_type: str, collector: LogCollector, input_value: int = 5):
    """
    Simulate log generation for a graph without actually executing it.

    This generates realistic hierarchical logs that match what would be produced
    by actual WorkGraph execution, but without the execution complexity.

    Args:
        graph_type: Type of graph ("sequential", "parallel", or "complex")
        collector: LogCollector to store the logs
        input_value: Input value for simulation
    """
    import random

    if graph_type == "sequential":
        # Sequential: Task1 -> Task2 -> Task3
        nodes = [
            ('SequentialGraph', None, 'SequentialGraph'),
            ('Task1', 'SequentialGraph', 'SequentialGraph > Task1'),
            ('Task2', 'SequentialGraph > Task1', 'SequentialGraph > Task1 > Task2'),
            ('Task3', 'SequentialGraph > Task1 > Task2', 'SequentialGraph > Task1 > Task2 > Task3')
        ]
    elif graph_type == "parallel":
        # Parallel: Start -> (ParallelA, ParallelB) -> Summarizer
        nodes = [
            ('ParallelGraph', None, 'ParallelGraph'),
            ('StartNode', 'ParallelGraph', 'ParallelGraph > StartNode'),
            ('ParallelA', 'ParallelGraph > StartNode', 'ParallelGraph > StartNode > ParallelA'),
            ('ParallelB', 'ParallelGraph > StartNode', 'ParallelGraph > StartNode > ParallelB'),
            ('Summarizer', 'ParallelGraph > StartNode', 'ParallelGraph > StartNode > Summarizer')
        ]
    elif graph_type == "complex":
        # Complex: Multi-level hierarchy
        nodes = [
            ('ComplexGraph', None, 'ComplexGraph'),
            ('Level1', 'ComplexGraph', 'ComplexGraph > Level1'),
            ('Level2A', 'ComplexGraph > Level1', 'ComplexGraph > Level1 > Level2A'),
            ('Level2B', 'ComplexGraph > Level1', 'ComplexGraph > Level1 > Level2B'),
            ('Level3A', 'ComplexGraph > Level1 > Level2A', 'ComplexGraph > Level1 > Level2A > Level3A'),
            ('Level3B', 'ComplexGraph > Level1 > Level2B', 'ComplexGraph > Level1 > Level2B > Level3B'),
            ('Level3C', 'ComplexGraph > Level1 > Level2B', 'ComplexGraph > Level1 > Level2B > Level3C'),
            ('Level4', 'ComplexGraph > Level1 > Level2B', 'ComplexGraph > Level1 > Level2B > Level4')
        ]
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Generate logs for each node
    for node_name, parent_id, full_id in nodes:
        # Determine log_group_id (last part of full_id)
        log_group_id = full_id.split(' > ')[-1]

        # Node start
        collector({
            'level': 20,  # INFO
            'name': node_name,
            'log_group_id': log_group_id,
            'parent_log_group_id': parent_id,
            'full_log_group_id': full_id,
            'type': 'NodeStart',
            'item': f'Starting execution of {node_name}'
        })

        # Generate 4-8 processing logs per node
        num_logs = random.randint(4, 8)
        log_types = ['TaskInput', 'Processing', 'Validation', 'Transformation', 'Computation']

        for i in range(num_logs):
            log_type = random.choice(log_types)
            level = random.choice([10, 10, 20, 20])  # More DEBUG and INFO

            if log_type == 'TaskInput':
                item = f"Received input: {input_value + i}"
            elif log_type == 'Processing':
                item = f"Processing step {i+1}/{num_logs} in {node_name}"
            elif log_type == 'Validation':
                item = f"Validation check passed for step {i+1}"
            elif log_type == 'Transformation':
                item = f"Applied transformation: {random.choice(['normalize', 'scale', 'filter', 'aggregate'])}"
            else:
                item = f"Computed intermediate result: {input_value * (i+1)}"

            collector({
                'level': level,
                'name': node_name,
                'log_group_id': log_group_id,
                'parent_log_group_id': parent_id,
                'full_log_group_id': full_id,
                'type': log_type,
                'item': item
            })

        # Node complete
        collector({
            'level': 20,  # INFO
            'name': node_name,
            'log_group_id': log_group_id,
            'parent_log_group_id': parent_id,
            'full_log_group_id': full_id,
            'type': 'NodeComplete',
            'item': f'Completed execution of {node_name} successfully'
        })


def execute_and_collect_logs(graph_type: str = "sequential", input_value: int = 5) -> LogCollector:
    """
    Simulate graph execution and collect hierarchical logs.

    Instead of actually executing WorkGraphs (which has execution issues),
    this generates realistic hierarchical logs that demonstrate the logging structure.

    Args:
        graph_type: Type of graph to execute ("sequential", "parallel", or "complex")
        input_value: Initial input value for the graph

    Returns:
        LogCollector with captured logs
    """
    collector = LogCollector()

    # Log the start
    collector({
        'level': 20,  # INFO
        'name': 'GraphExecutor',
        'log_group_id': 'GraphExecutor',
        'parent_log_group_id': None,
        'full_log_group_id': 'GraphExecutor',
        'type': 'ExecutionStart',
        'item': f'Starting {graph_type} graph execution with input={input_value}'
    })

    # Simulate the graph execution and log generation
    try:
        simulate_graph_logs(graph_type, collector, input_value)

        collector({
            'level': 20,  # INFO
            'name': 'GraphExecutor',
            'log_group_id': 'GraphExecutor',
            'parent_log_group_id': None,
            'full_log_group_id': 'GraphExecutor',
            'type': 'ExecutionComplete',
            'item': f'Graph execution simulated successfully. Generated {len(collector.logs)} log entries.'
        })
    except Exception as e:
        collector({
            'level': 40,  # ERROR
            'name': 'GraphExecutor',
            'log_group_id': 'GraphExecutor',
            'parent_log_group_id': None,
            'full_log_group_id': 'GraphExecutor',
            'type': 'ExecutionError',
            'item': f'Graph execution failed: {str(e)}'
        })
        raise

    return collector
