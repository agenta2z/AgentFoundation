"""
Multiprocessing tests for QueueInteractive

Demonstrates true cross-process communication using StorageBasedQueueService.
This tests the key advantage of using QueueServiceBase: ability to communicate
across separate Python processes.

Prerequisites:
    rich_python_utils package installed

Usage:
    python test_queue_interactive_multiprocessing.py
"""

import sys
from pathlib import Path
import tempfile
import shutil
import time
import multiprocessing as mp

# Add src to path
# From: test/agent_foundation/ui/test_queue_interactive_multiprocessing.py
# To: src/
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Also add rich_python_utils to path if available
science_python_utils_path = project_root.parent / 'SciencePythonUtils' / 'src'
if science_python_utils_path.exists():
    sys.path.insert(0, str(science_python_utils_path))

from science_modeling_tools.ui.queue_interactive import QueueInteractive

try:
    from rich_python_utils.service_utils.queue_service.storage_based_queue_service import (
        StorageBasedQueueService
    )
    QUEUE_SERVICES_AVAILABLE = True
except ImportError:
    QUEUE_SERVICES_AVAILABLE = False
    print("Warning: rich_python_utils not available, skipping tests")


def user_process(root_path, num_questions):
    """Simulates a user process sending questions"""
    print(f"[User Process] Starting, will ask {num_questions} questions")

    # Create queue service in this process
    queue_service = StorageBasedQueueService(root_path=root_path)

    questions = [
        "What is your name?",
        "What can you do?",
        "How are you today?",
        "What's the meaning of life?",
        "Goodbye!",
    ]

    for i in range(num_questions):
        question = questions[i % len(questions)]
        print(f"[User Process] Asking: {question}")
        queue_service.put('user_input', question)
        time.sleep(0.5)

        # Wait for response
        start_time = time.time()
        response = queue_service.get('agent_response', blocking=True, timeout=5.0)
        elapsed = time.time() - start_time

        if response:
            print(f"[User Process] Got response ({elapsed:.2f}s): {response}")
        else:
            print(f"[User Process] No response (timeout)")

    queue_service.close()
    print(f"[User Process] Finished")


def agent_process(root_path, num_responses):
    """Simulates an agent process using QueueInteractive"""
    print(f"[Agent Process] Starting, will respond to {num_responses} questions")

    # Create queue service in this process
    queue_service = StorageBasedQueueService(root_path=root_path)

    # Create interactive
    interactive = QueueInteractive(
        system_name="TestAgent",
        user_name="User",
        input_queue=queue_service,
        response_queue=queue_service,
        input_queue_id='user_input',
        response_queue_id='agent_response',
        blocking=True,
        timeout=10.0
    )

    responses = {
        "What is your name?": "I am TestAgent, a helpful assistant.",
        "What can you do?": "I can answer questions and help with tasks.",
        "How are you today?": "I'm functioning well, thank you for asking!",
        "What's the meaning of life?": "42, according to Douglas Adams.",
        "Goodbye!": "Goodbye! It was nice talking to you.",
    }

    for i in range(num_responses):
        print(f"[Agent Process] Waiting for question {i+1}...")
        question = interactive.get_input()

        if question:
            print(f"[Agent Process] Received: {question}")
            response = responses.get(question, "I'm not sure how to answer that.")
            print(f"[Agent Process] Responding: {response}")
            interactive.send_response(response, flag=False)
        else:
            print(f"[Agent Process] No question received (timeout)")
            break

    queue_service.close()
    print(f"[Agent Process] Finished")


def test_basic_cross_process():
    """Test 1: Basic cross-process communication"""
    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[SKIP] Test 1: Queue services not available")
        return

    print("\n" + "="*80)
    print("TEST 1: Basic Cross-Process Communication")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Starting user and agent processes...")
        num_exchanges = 3

        # Start agent process
        agent_proc = mp.Process(target=agent_process, args=(tmpdir, num_exchanges))
        agent_proc.start()

        # Start user process
        user_proc = mp.Process(target=user_process, args=(tmpdir, num_exchanges))
        user_proc.start()

        # Wait for both to complete
        user_proc.join(timeout=30)
        agent_proc.join(timeout=30)

        print("\n2. Checking process completion...")
        if user_proc.exitcode == 0:
            print("   [OK] User process completed successfully")
        else:
            print(f"   [WARNING] User process exit code: {user_proc.exitcode}")

        if agent_proc.exitcode == 0:
            print("   [OK] Agent process completed successfully")
        else:
            print(f"   [WARNING] Agent process exit code: {agent_proc.exitcode}")

        # Verify queues are empty
        queue_service = StorageBasedQueueService(root_path=tmpdir)
        input_size = queue_service.size('user_input')
        response_size = queue_service.size('agent_response')
        queue_service.close()

        print(f"\n3. Final queue sizes:")
        print(f"   Input queue: {input_size} items")
        print(f"   Response queue: {response_size} items")

        assert input_size == 0, f"Input queue should be empty, has {input_size} items"
        assert response_size == 0, f"Response queue should be empty, has {response_size} items"

        print("\n[PASS] Test 1 completed successfully!")

    finally:
        time.sleep(0.3)  # Let Windows release lock file
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def monitor_process(root_path, num_expected):
    """Monitor process that tracks conversation"""
    print(f"[Monitor] Starting, expecting {num_expected} exchanges")

    queue_service = StorageBasedQueueService(root_path=root_path)

    exchanges_seen = 0
    start_time = time.time()
    timeout = 30.0  # 30 second total timeout

    while exchanges_seen < num_expected:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"[Monitor] Timeout after {elapsed:.1f}s")
            break

        input_size = queue_service.size('user_input')
        response_size = queue_service.size('agent_response')

        # Count completed exchanges (both queues processed)
        if input_size == 0 and response_size == 0:
            # Check if we've seen any new exchanges
            time.sleep(0.5)
            continue

        print(f"[Monitor] Input queue: {input_size}, Response queue: {response_size}")
        time.sleep(0.5)

    queue_service.close()
    print(f"[Monitor] Finished")


def test_monitored_conversation():
    """Test 2: Monitored cross-process conversation"""
    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[SKIP] Test 2: Queue services not available")
        return

    print("\n" + "="*80)
    print("TEST 2: Monitored Cross-Process Conversation")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Starting user, agent, and monitor processes...")
        num_exchanges = 5

        # Start all processes
        agent_proc = mp.Process(target=agent_process, args=(tmpdir, num_exchanges))
        user_proc = mp.Process(target=user_process, args=(tmpdir, num_exchanges))
        monitor_proc = mp.Process(target=monitor_process, args=(tmpdir, num_exchanges))

        agent_proc.start()
        monitor_proc.start()
        time.sleep(0.5)  # Let agent start first
        user_proc.start()

        # Wait for completion
        user_proc.join(timeout=40)
        agent_proc.join(timeout=40)
        monitor_proc.join(timeout=40)

        print("\n2. All processes completed")
        print(f"   User exit code: {user_proc.exitcode}")
        print(f"   Agent exit code: {agent_proc.exitcode}")
        print(f"   Monitor exit code: {monitor_proc.exitcode}")

        print("\n[PASS] Test 2 completed successfully!")

    finally:
        time.sleep(0.3)  # Let Windows release lock file
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def concurrent_agent(root_path, agent_id, num_questions):
    """An agent that processes questions concurrently with other agents"""
    queue_service = StorageBasedQueueService(root_path=root_path)

    interactive = QueueInteractive(
        system_name=f"Agent{agent_id}",
        user_name="User",
        input_queue=queue_service,
        response_queue=queue_service,
        input_queue_id='shared_input',
        response_queue_id='shared_response',
        blocking=True,
        timeout=2.0
    )

    processed = 0
    while processed < num_questions:
        question = interactive.get_input()
        if question:
            print(f"[Agent {agent_id}] Got: {question}")
            response = f"Agent {agent_id} says: I received '{question}'"
            interactive.send_response(response, flag=False)
            processed += 1
        else:
            # Timeout - check if we're done
            if queue_service.size('shared_input') == 0:
                break

    queue_service.close()
    print(f"[Agent {agent_id}] Processed {processed} questions")


def test_multiple_agents_single_queue():
    """Test 3: Multiple agents processing from same input queue"""
    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[SKIP] Test 3: Queue services not available")
        return

    print("\n" + "="*80)
    print("TEST 3: Multiple Agents, Single Queue (Load Balancing)")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Creating shared queue with questions...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)

        num_questions = 12
        for i in range(num_questions):
            queue_service.put('shared_input', f"Question {i+1}")
        print(f"   [OK] Added {num_questions} questions to shared queue")

        queue_service.close()

        print("\n2. Starting 3 agents to process questions...")
        num_agents = 3
        questions_per_agent = num_questions // num_agents

        agents = []
        for agent_id in range(1, num_agents + 1):
            agent_proc = mp.Process(
                target=concurrent_agent,
                args=(tmpdir, agent_id, questions_per_agent)
            )
            agent_proc.start()
            agents.append(agent_proc)

        # Wait for all agents
        for agent_proc in agents:
            agent_proc.join(timeout=15)

        print("\n3. Checking results...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)

        input_remaining = queue_service.size('shared_input')
        responses_received = queue_service.size('shared_response')

        print(f"   Input remaining: {input_remaining}")
        print(f"   Responses received: {responses_received}")

        # Collect all responses
        responses = []
        while queue_service.size('shared_response') > 0:
            resp = queue_service.get('shared_response', blocking=False)
            if resp:
                responses.append(resp)
                print(f"   - {resp}")

        queue_service.close()

        # Should have processed most/all questions
        assert len(responses) >= num_questions * 0.8, \
            f"Expected at least {int(num_questions * 0.8)} responses, got {len(responses)}"

        print(f"\n   [OK] Processed {len(responses)}/{num_questions} questions")
        print("\n[PASS] Test 3 completed successfully!")

    finally:
        time.sleep(0.3)  # Let Windows release lock file
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def main():
    print("""
==============================================================================
            QueueInteractive Multiprocessing Tests
==============================================================================
""")

    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[ERROR] rich_python_utils not available. Cannot run tests.")
        print("Please ensure rich_python_utils is installed and in PYTHONPATH.")
        sys.exit(1)

    # Set start method for multiprocessing
    if sys.platform == 'win32':
        mp.set_start_method('spawn', force=True)

    tests = [
        test_basic_cross_process,
        test_monitored_conversation,
        test_multiple_agents_single_queue,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*80 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
