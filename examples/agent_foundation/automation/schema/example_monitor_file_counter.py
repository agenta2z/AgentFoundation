"""
Example: MonitorNode with File Counter (Queue-Based Execution)

This example demonstrates MonitorNode monitoring a file counter using
queue-based execution via SimulatedMultiThreadExecutor.

Scenario:
    - A background thread increments a counter in a file every 2 seconds
    - MonitorNode polls the file and triggers actions when count % 20 == 0
    - Uses WorkGraph with executor for queue-based execution

Key Concepts:
    - MonitorNode: Generic monitor from ScienceModelingTools
    - NextNodesSelector: Controls whether to continue looping or proceed downstream
    - WorkGraph with executor: Queue-based execution
    - Self-loop pattern: Monitor re-executes via self-edge until condition met

Usage:
    python example_monitor_file_counter.py
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
from rich_python_utils.mp_utils.queued_executor import SimulatedMultiThreadExecutor
from rich_python_utils.service_utils.queue_service.thread_queue_service import ThreadQueueService

from agent_foundation.automation.schema.monitor import MonitorNode, MonitorResult, MonitorStatus


# =============================================================================
# Counter Thread (Background task that increments a counter)
# =============================================================================

class CounterThread(threading.Thread):
    """Background thread that increments a counter in a file."""

    def __init__(self, counter_file: Path, interval: float = 2.0, max_count: int = 100):
        super().__init__(daemon=True)
        self.counter_file = counter_file
        self.interval = interval
        self.max_count = max_count
        self._stop_event = threading.Event()

    def run(self):
        count = 0
        while count < self.max_count and not self._stop_event.is_set():
            count += 1
            self.counter_file.write_text(str(count))
            print(f"   [Counter] count={count}")
            self._stop_event.wait(self.interval)
        print(f"   [Counter] Finished at count={count}")

    def stop(self):
        self._stop_event.set()


# =============================================================================
# Monitor Iteration Function
# =============================================================================

def create_file_monitor_iteration(counter_file: Path, target_divisor: int = 20):
    """
    Create a monitor iteration that checks if file counter % target_divisor == 0.

    Returns:
        Callable that returns NextNodesSelector wrapping MonitorResult
    """
    check_count = [0]  # Track number of checks

    def iteration(prev_result=None) -> NextNodesSelector:
        """One iteration of the monitor check."""
        check_count[0] += 1

        # Read current count from file
        try:
            current_count = int(counter_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            current_count = 0

        print(f"   [Monitor] Check #{check_count[0]}: count={current_count}")

        # Check if condition is met
        if current_count > 0 and current_count % target_divisor == 0:
            # Condition met!
            print(f"   [Monitor] Milestone reached: count={current_count} (divisible by {target_divisor})")

            result = MonitorResult(
                success=True,
                status=MonitorStatus.CONDITION_MET,
                matched_content=current_count,
                check_count=check_count[0]
            )

            # include_self=False: Stop looping
            # include_others=True: Run downstream actions
            return NextNodesSelector(
                include_self=False,
                include_others=True,
                result=result
            )
        else:
            # Condition not met, continue polling
            result = MonitorResult(
                success=False,
                status=MonitorStatus.MAX_ITERATIONS,
                check_count=check_count[0]
            )

            # include_self=True: Continue looping via self-edge
            # include_others=False: Don't run downstream yet
            return NextNodesSelector(
                include_self=True,
                include_others=False,
                result=result
            )

    return iteration


# =============================================================================
# Action Node Function
# =============================================================================

def milestone_action(prev_result):
    """Action to run when milestone is reached."""
    print(f"   [Action] Processing milestone...")

    # Extract the count from MonitorResult
    if isinstance(prev_result, MonitorResult):
        count = prev_result.matched_content
    else:
        count = prev_result

    print(f"   [Action] Milestone at count={count}")
    print(f"   [Action] Simulating work (notification, database update, etc.)...")
    time.sleep(0.5)

    result = {
        'action': 'milestone_processed',
        'count': count,
        'timestamp': time.time()
    }
    print(f"   [Action] Completed!")
    return result


# =============================================================================
# Helper Functions
# =============================================================================

def unique_queue_ids(prefix='monitor'):
    """Generate unique queue IDs to avoid contamination."""
    unique = uuid.uuid4().hex[:8]
    return f'{prefix}_in_{unique}', f'{prefix}_out_{unique}'


def create_executor():
    """Create a SimulatedMultiThreadExecutor for testing."""
    service = ThreadQueueService()
    input_id, output_id = unique_queue_ids()
    executor = SimulatedMultiThreadExecutor(
        input_queue_service=service,
        output_queue_service=service,
        input_queue_id=input_id,
        output_queue_id=output_id,
        verbose=False
    )
    return executor, service


# =============================================================================
# Main Example
# =============================================================================

def main():
    print("""
==============================================================================
        MonitorNode with File Counter (Queue-Based Execution)
==============================================================================

This example demonstrates:
1. MonitorNode from ScienceModelingTools for generic monitoring
2. Queue-based execution via SimulatedMultiThreadExecutor
3. Self-loop pattern: monitor continues until condition is met
4. Downstream action triggered when count % 20 == 0
""")

    # =========================================================================
    # 1. Setup
    # =========================================================================
    print("1. Setup...")

    # Create temporary file for counter
    counter_file = Path(tempfile.gettempdir()) / "monitor_node_counter.txt"
    counter_file.write_text("0")
    print(f"   Counter file: {counter_file}")

    # Create executor
    executor, service = create_executor()
    print("   [OK] Executor created")

    # =========================================================================
    # 2. Build WorkGraph with MonitorNode
    # =========================================================================
    print("\n2. Building WorkGraph with MonitorNode...")

    # Create monitor iteration
    iteration = create_file_monitor_iteration(counter_file, target_divisor=20)

    # Create MonitorNode
    monitor_node = MonitorNode(
        name="file_monitor",
        iteration=iteration,
        poll_interval=0.5,  # Check every 0.5 seconds
        max_repeat=1,  # Single iteration per graph execution (self-edge handles looping)
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
    )

    # Create action node for when condition is met
    action_node = WorkGraphNode(
        name="milestone_action",
        value=milestone_action,
        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
    )

    # Build graph: monitor -> action, with self-edge for looping
    monitor_node.add_next(action_node)  # When condition met, run action
    monitor_node.add_next(monitor_node)  # Self-edge for continuous polling

    # Create WorkGraph with executor for queue-based execution
    graph = WorkGraph(start_nodes=[monitor_node], executor=executor)

    print("   Graph structure:")
    print(f"     {monitor_node.str_all_descendants()}")
    print("   [OK] Graph built with self-edge for continuous monitoring")

    # =========================================================================
    # 3. Start counter thread
    # =========================================================================
    print("\n3. Starting counter thread...")

    counter_thread = CounterThread(
        counter_file=counter_file,
        interval=2.0,
        max_count=25  # Will reach 20 milestone
    )
    counter_thread.start()
    print("   [OK] Counter thread started (incrementing every 2 seconds)")

    # =========================================================================
    # 4. Run graph
    # =========================================================================
    print("\n4. Running WorkGraph with executor...")
    print("   (Monitor will poll until count % 20 == 0, then trigger action)")
    print()

    start_time = time.time()
    result = graph.run()
    elapsed = time.time() - start_time

    # =========================================================================
    # 5. Results
    # =========================================================================
    print(f"\n5. Results (completed in {elapsed:.1f}s)...")
    print(f"   Final result: {result}")

    # =========================================================================
    # 6. Cleanup
    # =========================================================================
    print("\n6. Cleanup...")

    counter_thread.stop()
    counter_thread.join(timeout=5.0)
    executor.stop()
    service.close()

    if counter_file.exists():
        counter_file.unlink()

    print("   [OK] Complete")

    print("\n" + "=" * 80)
    print("[OK] Example completed successfully!")
    print("=" * 80)
    print("""
Key Takeaways:
- MonitorNode wraps a polling iteration with NextNodesSelector control
- NextNodesSelector(include_self=True, include_others=False) = continue polling
- NextNodesSelector(include_self=False, include_others=True) = condition met, run downstream
- Self-edge (monitor.add_next(monitor)) enables continuous polling pattern
- Queue-based executor manages task scheduling automatically
- poll_interval controls delay between iterations
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
