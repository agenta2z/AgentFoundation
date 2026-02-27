"""
Multiprocessing agent example using QueueInteractive

Demonstrates real-world cross-process agent communication:
- Separate user and agent processes
- Interactive Q&A session
- Multiple agents with load balancing
- Agent pool pattern

This is the KEY advantage of using QueueServiceBase with QueueInteractive:
TRUE cross-process communication for distributed agent systems.

Prerequisites:
    rich_python_utils package installed

Usage:
    python example_multiprocessing_agent.py
"""

import sys
from pathlib import Path
import tempfile
import shutil
import time
import multiprocessing as mp
from datetime import datetime

# Add src to path
# From: examples/agent_foundation/ui/queue_interactive/example_multiprocessing_agent.py
# To: src/
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Also add rich_python_utils to path if available
rich_python_utils_path = project_root.parent / 'SciencePythonUtils' / 'src'
if rich_python_utils_path.exists():
    sys.path.insert(0, str(rich_python_utils_path))

from agent_foundation.ui.queue_interactive import QueueInteractive

try:
    from rich_python_utils.service_utils.queue_service.storage_based_queue_service import (
        StorageBasedQueueService
    )
    QUEUE_SERVICES_AVAILABLE = True
except ImportError:
    QUEUE_SERVICES_AVAILABLE = False
    print("Error: rich_python_utils not available")
    sys.exit(1)


def simple_agent_process(root_path):
    """
    Simple agent process that answers questions.

    This runs in a SEPARATE process from the user.
    Communication happens through StorageBasedQueueService.
    """
    print("[Agent] Process starting...")

    # Create queue service in THIS process
    queue_service = StorageBasedQueueService(root_path=root_path)

    # Create interactive
    interactive = QueueInteractive(
        system_name="SimpleAgent",
        user_name="User",
        input_queue=queue_service,
        response_queue=queue_service,
        input_queue_id='user_questions',
        response_queue_id='agent_answers',
        blocking=True,
        timeout=30.0
    )

    print("[Agent] Ready to answer questions...")

    # Simple knowledge base
    knowledge = {
        "who are you": "I am SimpleAgent, a helpful Q&A assistant.",
        "what can you do": "I can answer questions about various topics.",
        "what is python": "Python is a high-level programming language.",
        "what is ai": "AI (Artificial Intelligence) is the simulation of human intelligence by machines.",
    }

    question_count = 0

    while True:
        # Get question from user
        question = interactive.get_input()

        if not question:
            print("[Agent] No more questions, shutting down...")
            break

        question_count += 1
        print(f"[Agent] Question {question_count}: {question}")

        # Check for exit
        if question.lower() in ['exit', 'quit', 'bye']:
            interactive.send_response("Goodbye! Have a great day!", flag=False)
            break

        # Look up answer
        question_lower = question.lower().strip('?').strip()
        answer = knowledge.get(question_lower, "I'm not sure about that. Can you ask something else?")

        # Send response
        interactive.send_response(answer, flag=False)
        print(f"[Agent] Answered: {answer[:50]}...")

    queue_service.close()
    print(f"[Agent] Process finished after answering {question_count} questions")


def user_process(root_path, questions):
    """
    User process that asks questions.

    This runs in a SEPARATE process from the agent.
    Communication happens through StorageBasedQueueService.
    """
    print(f"[User] Process starting with {len(questions)} questions...")

    # Create queue service in THIS process
    queue_service = StorageBasedQueueService(root_path=root_path)

    for i, question in enumerate(questions, 1):
        print(f"\n[User] Question {i}: {question}")

        # Send question
        queue_service.put('user_questions', question)

        # Wait for answer
        answer = queue_service.get('agent_answers', blocking=True, timeout=5.0)

        if answer:
            print(f"[User] Answer: {answer}")
        else:
            print("[User] No answer received (timeout)")

        time.sleep(0.3)  # Simulate think time

    queue_service.close()
    print("[User] Process finished")


def example_1_simple_qa():
    """Example 1: Simple Q&A between user and agent in separate processes"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Q&A (Cross-Process)")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Starting agent and user processes...")

        questions = [
            "Who are you?",
            "What can you do?",
            "What is Python?",
            "exit"
        ]

        # Start agent process
        agent_proc = mp.Process(target=simple_agent_process, args=(tmpdir,))
        agent_proc.start()

        # Give agent time to start
        time.sleep(0.5)

        # Start user process
        user_proc = mp.Process(target=user_process, args=(tmpdir, questions))
        user_proc.start()

        # Wait for both to complete
        user_proc.join(timeout=30)
        agent_proc.join(timeout=30)

        print("\n2. Both processes completed")
        print(f"   Agent exit code: {agent_proc.exitcode}")
        print(f"   User exit code: {user_proc.exitcode}")

        print("\n[OK] Example 1 completed!")

    finally:
        time.sleep(0.3)
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def worker_agent(root_path, agent_id, num_tasks):
    """Worker agent that processes tasks from shared queue"""
    queue_service = StorageBasedQueueService(root_path=root_path)

    interactive = QueueInteractive(
        system_name=f"Worker{agent_id}",
        user_name="TaskQueue",
        input_queue=queue_service,
        response_queue=queue_service,
        input_queue_id='task_queue',
        response_queue_id='result_queue',
        blocking=True,
        timeout=3.0
    )

    print(f"[Worker {agent_id}] Started, will process up to {num_tasks} tasks")

    processed = 0
    while processed < num_tasks:
        # Get task
        task = interactive.get_input()

        if not task:
            # Timeout - check if queue is empty
            if queue_service.size('task_queue') == 0:
                print(f"[Worker {agent_id}] No more tasks")
                break
            continue

        processed += 1
        print(f"[Worker {agent_id}] Processing task {processed}: {task}")

        # Simulate work
        time.sleep(0.1)

        # Send result
        result = {
            'worker_id': agent_id,
            'task': task,
            'result': f'Completed by Worker {agent_id}',
            'timestamp': datetime.now().isoformat()
        }
        interactive.send_response(str(result), flag=False)

    queue_service.close()
    print(f"[Worker {agent_id}] Finished, processed {processed} tasks")


def example_2_agent_pool():
    """Example 2: Pool of agents processing tasks (load balancing)"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Agent Pool with Load Balancing")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Creating task queue...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)

        num_tasks = 15
        for i in range(num_tasks):
            task = f"Task-{i+1:02d}"
            queue_service.put('task_queue', task)
            print(f"   [+] Added {task}")

        queue_service.close()

        print(f"\n2. Starting 3 worker agents...")
        num_workers = 3
        tasks_per_worker = num_tasks // num_workers + 1

        workers = []
        for worker_id in range(1, num_workers + 1):
            worker_proc = mp.Process(
                target=worker_agent,
                args=(tmpdir, worker_id, tasks_per_worker)
            )
            worker_proc.start()
            workers.append(worker_proc)

        # Wait for all workers
        for worker_proc in workers:
            worker_proc.join(timeout=20)

        print("\n3. Collecting results...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)

        results_count = queue_service.size('result_queue')
        print(f"   [OK] {results_count} results in queue")

        # Get all results
        results = []
        while queue_service.size('result_queue') > 0:
            result = queue_service.get('result_queue', blocking=False)
            if result:
                results.append(result)

        print(f"\n4. Results summary:")
        print(f"   Total tasks: {num_tasks}")
        print(f"   Results received: {len(results)}")
        print(f"   Success rate: {len(results)/num_tasks*100:.1f}%")

        queue_service.close()

        print("\n[OK] Example 2 completed!")

    finally:
        time.sleep(0.3)
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def specialized_agent(root_path, agent_type, input_queue_id, output_queue_id):
    """Specialized agent for specific task types"""
    queue_service = StorageBasedQueueService(root_path=root_path)

    interactive = QueueInteractive(
        system_name=f"{agent_type}Agent",
        user_name="Router",
        input_queue=queue_service,
        response_queue=queue_service,
        input_queue_id=input_queue_id,
        response_queue_id=output_queue_id,
        blocking=True,
        timeout=5.0
    )

    print(f"[{agent_type}] Agent started")

    task_count = 0
    while task_count < 5:  # Process up to 5 tasks
        task = interactive.get_input()

        if not task:
            break

        task_count += 1
        print(f"[{agent_type}] Processing: {task}")

        # Simulate specialized processing
        time.sleep(0.1)
        result = f"[{agent_type}] Processed: {task}"

        interactive.send_response(result, flag=False)

    queue_service.close()
    print(f"[{agent_type}] Agent finished, processed {task_count} tasks")


def example_3_specialized_agents():
    """Example 3: Multiple specialized agents with routing"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Specialized Agents with Routing")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Setting up task queues...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)

        # Add tasks to different queues based on type
        math_tasks = ["2+2", "5*3", "10/2"]
        text_tasks = ["uppercase: hello", "reverse: world", "length: python"]

        for task in math_tasks:
            queue_service.put('math_queue', task)
            print(f"   [Math] {task}")

        for task in text_tasks:
            queue_service.put('text_queue', task)
            print(f"   [Text] {task}")

        queue_service.close()

        print("\n2. Starting specialized agents...")

        # Math agent
        math_agent = mp.Process(
            target=specialized_agent,
            args=(tmpdir, "Math", "math_queue", "math_results")
        )
        math_agent.start()

        # Text agent
        text_agent = mp.Process(
            target=specialized_agent,
            args=(tmpdir, "Text", "text_queue", "text_results")
        )
        text_agent.start()

        # Wait for completion
        math_agent.join(timeout=15)
        text_agent.join(timeout=15)

        print("\n3. Collecting results...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)

        math_results = []
        while queue_service.size('math_results') > 0:
            result = queue_service.get('math_results', blocking=False)
            if result:
                math_results.append(result)
                print(f"   {result}")

        text_results = []
        while queue_service.size('text_results') > 0:
            result = queue_service.get('text_results', blocking=False)
            if result:
                text_results.append(result)
                print(f"   {result}")

        queue_service.close()

        print(f"\n   Math results: {len(math_results)}")
        print(f"   Text results: {len(text_results)}")

        print("\n[OK] Example 3 completed!")

    finally:
        time.sleep(0.3)
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def main():
    print("""
==============================================================================
        QueueInteractive Multiprocessing Agent Examples
==============================================================================

These examples demonstrate TRUE cross-process agent communication using
StorageBasedQueueService with QueueInteractive.

Key Features:
- Agents run in separate Python processes
- Communication through persistent file-based queues
- No Manager or shared memory required
- Survives process restarts

Use Cases:
- Distributed agent systems
- Multi-agent collaboration
- Load-balanced task processing
- Agent pools and routing
==============================================================================
""")

    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[ERROR] rich_python_utils not available")
        sys.exit(1)

    # Set multiprocessing start method
    if sys.platform == 'win32':
        mp.set_start_method('spawn', force=True)

    examples = [
        example_1_simple_qa,
        example_2_agent_pool,
        example_3_specialized_agents,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n[ERROR] {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("[OK] All multiprocessing examples completed successfully!")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
