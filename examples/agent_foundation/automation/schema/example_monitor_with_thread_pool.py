"""
Example: MonitorNode with Thread Pool Executor (Parallel Execution)

This example demonstrates MonitorNode with QueuedThreadPoolExecutor,
showing how multiple monitors can run in parallel.

Scenario:
    - Two background threads increment separate counters in files
    - Two MonitorNodes watch different files in parallel
    - Each triggers actions when their respective conditions are met
    - Uses QueuedThreadPoolExecutor for true parallel execution

Key Concepts:
    - QueuedThreadPoolExecutor: Multiple worker threads for parallel execution
    - Multiple start nodes: Both monitors run concurrently
    - Independent monitoring: Each monitor has its own condition

Usage:
    python example_monitor_with_thread_pool.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import resolve_path  # noqa: F401 - Must be first import for path setup

import time
import threading
import tempfile
import uuid

from rich_python_utils.common_objects.workflow.workgraph import WorkGraph, WorkGraphNode
from rich_python_utils.common_objects.workflow.common.worknode_base import NextNodesSelector
from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import ResultPassDownMode
from rich_python_utils.mp_utils.queued_executor import QueuedThreadPoolExecutor
from rich_python_utils.service_utils.queue_service.thread_queue_service import ThreadQueueService

from agent_foundation.automation.schema.monitor import MonitorNode, MonitorResult, MonitorStatus


# =============================================================================
# Counter Thread
# =============================================================================

class CounterThread(threading.Thread):
    """Background thread that increments a counter in a file."""

    def __init__(self, name: str, counter_file: Path, interval: float, max_count: int):
        super().__init__(daemon=True, name=name)
        self.counter_file = counter_file
        self.interval = interval
        self.max_count = max_count
        self._stop_event = threading.Event()

    def run(self):
        count = 0
        while count < self.max_count and not self._stop_event.is_set():
            count += 1
            self.counter_file.write_text(str(count))
            print(f"   [{self.name}] count={count}")
            self._stop_event.wait(self.interval)
        print(f"   [{self.name}] Finished at count={count}")

    def stop(self):
        self._stop_event.set()


# =============================================================================
# Monitor Iteration Factory
# =============================================================================

def create_monitor_iteration(name: str, counter_file: Path, target: int):
    """
    Create a monitor iteration that checks if file counter reaches target.

    Args:
        name: Name for logging
        counter_file: Path to the counter file
        target: Target value to match

    Returns:
        Callable that returns NextNodesSelector wrapping MonitorResult
    """
    check_count = [0]

    def iteration(prev_result=None) -> NextNodesSelector:
        check_count[0] += 1

        try:
            current_count = int(counter_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            current_count = 0

        print(f"   [{name}] Check #{check_count[0]}: count={current_count}, target={target}")

        if current_count >= target:
            print(f"   [{name}] Target reached! count={current_count} >= {target}")

            result = MonitorResult(
                success=True,
                status=MonitorStatus.CONDITION_MET,
                matched_content={'name': name, 'count': current_count, 'target': target},
                check_count=check_count[0]
            )
            return NextNodesSelector(
                include_self=False,
                include_others=True,
                result=result
            )
        else:
            result = MonitorResult(
                success=False,
                status=MonitorStatus.MAX_ITERATIONS,
                check_count=check_count[0]
            )
            return NextNodesSelector(
                include_self=True,
                include_others=False,
                result=result
            )

    return iteration


# =============================================================================
# Action Functions
# =============================================================================

def action_a(prev_result):
    """Action for Monitor A."""
    print(f"   [Action-A] Processing...")
    if isinstance(prev_result, MonitorResult):
        info = prev_result.matched_content
    else:
        info = prev_result
    print(f"   [Action-A] Target reached: {info}")
    time.sleep(0.3)
    return {'action': 'A', 'info': info, 'timestamp': time.time()}


def action_b(prev_result):
    """Action for Monitor B."""
    print(f"   [Action-B] Processing...")
    if isinstance(prev_result, MonitorResult):
        info = prev_result.matched_content
    else:
        info = prev_result
    print(f"   [Action-B] Target reached: {info}")
    time.sleep(0.3)
    return {'action': 'B', 'info': info, 'timestamp': time.time()}


# =============================================================================
# Helper Functions
# =============================================================================

def unique_queue_ids(prefix='threadpool'):
    unique = uuid.uuid4().hex[:8]
    return f'{prefix}_in_{unique}', f'{prefix}_out_{unique}'


def create_thread_pool_executor(num_workers=4):
    """Create a QueuedThreadPoolExecutor."""
    service = ThreadQueueService()
    input_id, output_id = unique_queue_ids()
    executor = QueuedThreadPoolExecutor(
        input_queue_service=service,
        output_queue_service=service,
        input_queue_id=input_id,
        output_queue_id=output_id,
        num_workers=num_workers,
        name='MonitorPool',
        verbose=False
    )
    return executor, service


# =============================================================================
# Main Example
# =============================================================================

def main():
    print("""
==============================================================================
        MonitorNode with Thread Pool Executor (Parallel Monitors)
==============================================================================

This example demonstrates:
1. Two monitors running in parallel with QueuedThreadPoolExecutor
2. Each monitor watches a different file counter
3. Monitor A: waits for count >= 10 (fast counter, 1.5s interval)
4. Monitor B: waits for count >= 15 (slow counter, 2.5s interval)
5. Each triggers its own action when condition is met
""")

    # =========================================================================
    # 1. Setup
    # =========================================================================
    print("1. Setup...")

    # Create temporary files for counters
    counter_file_a = Path(tempfile.gettempdir()) / "monitor_counter_a.txt"
    counter_file_b = Path(tempfile.gettempdir()) / "monitor_counter_b.txt"
    counter_file_a.write_text("0")
    counter_file_b.write_text("0")

    print(f"   Counter A: {counter_file_a}")
    print(f"   Counter B: {counter_file_b}")

    # Create thread pool executor
    executor, service = create_thread_pool_executor(num_workers=4)
    print("   [OK] Thread pool executor created with 4 workers")

    # =========================================================================
    # 2. Build WorkGraph with two parallel monitors
    # =========================================================================
    print("\n2. Building WorkGraph with parallel monitors...")

    # Monitor A: watches counter_a for count >= 10
    iteration_a = create_monitor_iteration("Monitor-A", counter_file_a, target=10)
    monitor_a = MonitorNode(
        name="monitor_a",
        iteration=iteration_a,
        poll_interval=0.3,
        max_repeat=1,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
    )

    # Monitor B: watches counter_b for count >= 15
    iteration_b = create_monitor_iteration("Monitor-B", counter_file_b, target=15)
    monitor_b = MonitorNode(
        name="monitor_b",
        iteration=iteration_b,
        poll_interval=0.3,
        max_repeat=1,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
    )

    # Action nodes
    action_node_a = WorkGraphNode(
        name="action_a",
        value=action_a,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
    )
    action_node_b = WorkGraphNode(
        name="action_b",
        value=action_b,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
    )

    # Build graph structure
    # Monitor A -> Action A, with self-edge
    monitor_a.add_next(action_node_a)
    monitor_a.add_next(monitor_a)

    # Monitor B -> Action B, with self-edge
    monitor_b.add_next(action_node_b)
    monitor_b.add_next(monitor_b)

    # Create WorkGraph with BOTH monitors as start nodes (parallel execution)
    graph = WorkGraph(start_nodes=[monitor_a, monitor_b], executor=executor)

    print("   Monitor A: waiting for count >= 10")
    print("   Monitor B: waiting for count >= 15")
    print("   [OK] Graph built with parallel monitors")

    # =========================================================================
    # 3. Start counter threads
    # =========================================================================
    print("\n3. Starting counter threads...")

    # Counter A: faster (1.5s interval)
    counter_thread_a = CounterThread(
        name="Counter-A",
        counter_file=counter_file_a,
        interval=1.5,
        max_count=20
    )

    # Counter B: slower (2.5s interval)
    counter_thread_b = CounterThread(
        name="Counter-B",
        counter_file=counter_file_b,
        interval=2.5,
        max_count=25
    )

    counter_thread_a.start()
    counter_thread_b.start()
    print("   [OK] Counter threads started")
    print("       Counter-A: 1.5s interval, target=10")
    print("       Counter-B: 2.5s interval, target=15")

    # =========================================================================
    # 4. Run graph
    # =========================================================================
    print("\n4. Running WorkGraph with thread pool executor...")
    print("   (Both monitors run in parallel)")
    print()

    start_time = time.time()
    result = graph.run()
    elapsed = time.time() - start_time

    # =========================================================================
    # 5. Results
    # =========================================================================
    print(f"\n5. Results (completed in {elapsed:.1f}s)...")
    print(f"   Final results: {result}")

    # Analyze timing
    print("\n   Timing analysis:")
    print(f"   - Counter A should reach 10 at ~{10 * 1.5:.1f}s")
    print(f"   - Counter B should reach 15 at ~{15 * 2.5:.1f}s")
    print(f"   - Parallel execution: both completed in {elapsed:.1f}s")

    # =========================================================================
    # 6. Cleanup
    # =========================================================================
    print("\n6. Cleanup...")

    counter_thread_a.stop()
    counter_thread_b.stop()
    counter_thread_a.join(timeout=5.0)
    counter_thread_b.join(timeout=5.0)
    executor.stop()
    service.close()

    for f in [counter_file_a, counter_file_b]:
        if f.exists():
            f.unlink()

    print("   [OK] Complete")

    print("\n" + "=" * 80)
    print("[OK] Example completed successfully!")
    print("=" * 80)
    print("""
Key Takeaways:
- Multiple start nodes enable parallel monitor execution
- QueuedThreadPoolExecutor provides true concurrent execution
- Each monitor independently polls and triggers its own actions
- Self-edge pattern works with multiple monitors simultaneously
- Results are returned as tuple when multiple leaf nodes complete
""")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
