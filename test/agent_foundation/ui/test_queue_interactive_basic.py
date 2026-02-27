"""
Basic tests for QueueInteractive with different queue services

Tests QueueInteractive with:
- StorageBasedQueueService (file-based, works across processes)
- ThreadQueueService (in-memory, works within same process)

Prerequisites:
    rich_python_utils package installed

Usage:
    python test_queue_interactive_basic.py
"""

import sys
from pathlib import Path
import tempfile
import shutil
import time

# Add src to path
# From: test/agent_foundation/ui/test_queue_interactive_basic.py
# To: src/
project_root = Path(__file__).parent.parent.parent.parent
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
    from rich_python_utils.service_utils.queue_service.thread_queue_service import (
        ThreadQueueService
    )
    QUEUE_SERVICES_AVAILABLE = True
except ImportError:
    QUEUE_SERVICES_AVAILABLE = False
    print("Warning: rich_python_utils not available, skipping tests")


def test_basic_usage_storage_queue():
    """Test 1: Basic usage with StorageBasedQueueService"""
    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[SKIP] Test 1: Queue services not available")
        return

    print("\n" + "="*80)
    print("TEST 1: Basic Usage with StorageBasedQueueService")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Creating StorageBasedQueueService...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)
        print(f"   [OK] Queue service created at {tmpdir}")

        print("\n2. Creating QueueInteractive...")
        interactive = QueueInteractive(
            system_name="TestAgent",
            user_name="TestUser",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='test_input',
            response_queue_id='test_response'
        )
        print("   [OK] QueueInteractive created")

        print("\n3. Putting input into queue...")
        queue_service.put('test_input', "Hello, agent!")
        print("   [OK] Input queued: 'Hello, agent!'")

        print("\n4. Getting input via interactive...")
        user_input = interactive.get_input()
        print(f"   [OK] Received: '{user_input}'")
        assert user_input == "Hello, agent!", f"Expected 'Hello, agent!', got '{user_input}'"

        print("\n5. Sending response via interactive...")
        interactive.send_response("Hello, user! How can I help?", flag=False)
        print("   [OK] Response sent")

        print("\n6. Getting response from queue...")
        response = queue_service.get('test_response', blocking=False)
        print(f"   [OK] Response received: '{response}'")
        assert response == "Hello, user! How can I help?", f"Unexpected response: {response}"

        queue_service.close()
        print("\n[PASS] Test 1 completed successfully!")

    finally:
        time.sleep(0.2)  # Let Windows release lock file
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def test_basic_usage_thread_queue():
    """Test 2: Basic usage with ThreadQueueService"""
    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[SKIP] Test 2: Queue services not available")
        return

    print("\n" + "="*80)
    print("TEST 2: Basic Usage with ThreadQueueService")
    print("="*80)

    print("\n1. Creating ThreadQueueService...")
    queue_service = ThreadQueueService()
    print("   [OK] Queue service created")

    print("\n2. Creating QueueInteractive...")
    interactive = QueueInteractive(
        system_name="TestAgent",
        user_name="TestUser",
        input_queue=queue_service,
        response_queue=queue_service,
        input_queue_id='test_input',
        response_queue_id='test_response'
    )
    print("   [OK] QueueInteractive created")

    print("\n3. Putting input into queue...")
    queue_service.put('test_input', "What's 2+2?")
    print("   [OK] Input queued: 'What's 2+2?'")

    print("\n4. Getting input via interactive...")
    user_input = interactive.get_input()
    print(f"   [OK] Received: '{user_input}'")
    assert user_input == "What's 2+2?", f"Expected 'What's 2+2?', got '{user_input}'"

    print("\n5. Sending response via interactive...")
    interactive.send_response("The answer is 4.", flag=False)
    print("   [OK] Response sent")

    print("\n6. Getting response from queue...")
    response = queue_service.get('test_response', blocking=False)
    print(f"   [OK] Response received: '{response}'")
    assert response == "The answer is 4.", f"Unexpected response: {response}"

    queue_service.close()
    print("\n[PASS] Test 2 completed successfully!")


def test_multiple_exchanges():
    """Test 3: Multiple input-response exchanges"""
    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[SKIP] Test 3: Queue services not available")
        return

    print("\n" + "="*80)
    print("TEST 3: Multiple Input-Response Exchanges")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Setting up queue service and interactive...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)
        interactive = QueueInteractive(
            system_name="Agent",
            user_name="User",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='input',
            response_queue_id='response'
        )
        print("   [OK] Setup complete")

        # Test multiple exchanges
        exchanges = [
            ("How are you?", "I'm doing well, thank you!"),
            ("What's the weather?", "I don't have access to weather data."),
            ("Goodbye!", "Goodbye! Have a great day!"),
        ]

        for i, (user_msg, agent_msg) in enumerate(exchanges, 1):
            print(f"\n2.{i} Exchange {i}:")
            print(f"   User -> Agent: '{user_msg}'")
            queue_service.put('input', user_msg)

            received_input = interactive.get_input()
            assert received_input == user_msg, f"Input mismatch: {received_input}"
            print(f"   [OK] Agent received: '{received_input}'")

            interactive.send_response(agent_msg, flag=False)
            print(f"   Agent -> User: '{agent_msg}'")

            received_response = queue_service.get('response', blocking=False)
            assert received_response == agent_msg, f"Response mismatch: {received_response}"
            print(f"   [OK] User received: '{received_response}'")

        queue_service.close()
        print("\n[PASS] Test 3 completed successfully!")

    finally:
        time.sleep(0.2)  # Let Windows release lock file
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def test_timeout_behavior():
    """Test 4: Timeout behavior with blocking get"""
    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[SKIP] Test 4: Queue services not available")
        return

    print("\n" + "="*80)
    print("TEST 4: Timeout Behavior")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Creating interactive with timeout=1.0s...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)
        interactive = QueueInteractive(
            system_name="Agent",
            user_name="User",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='input',
            response_queue_id='response',
            blocking=True,
            timeout=1.0  # 1 second timeout
        )
        print("   [OK] Interactive created with 1s timeout")

        print("\n2. Attempting to get input from empty queue (will timeout)...")
        import time
        start_time = time.time()
        result = interactive.get_input()
        elapsed = time.time() - start_time

        print(f"   [OK] Timed out after {elapsed:.2f}s, returned: '{result}'")
        assert result == "", f"Expected empty string on timeout, got: '{result}'"
        assert 0.9 < elapsed < 1.5, f"Timeout should be ~1s, was {elapsed:.2f}s"

        queue_service.close()
        print("\n[PASS] Test 4 completed successfully!")

    finally:
        time.sleep(0.2)  # Let Windows release lock file
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def test_non_blocking_behavior():
    """Test 5: Non-blocking get behavior"""
    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[SKIP] Test 5: Queue services not available")
        return

    print("\n" + "="*80)
    print("TEST 5: Non-blocking Behavior")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Creating interactive with blocking=False...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)
        interactive = QueueInteractive(
            system_name="Agent",
            user_name="User",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='input',
            response_queue_id='response',
            blocking=False
        )
        print("   [OK] Interactive created (non-blocking)")

        print("\n2. Attempting to get input from empty queue (should return immediately)...")
        import time
        start_time = time.time()
        result = interactive.get_input()
        elapsed = time.time() - start_time

        print(f"   [OK] Returned immediately after {elapsed:.3f}s, result: '{result}'")
        assert result == "", f"Expected empty string, got: '{result}'"
        assert elapsed < 0.1, f"Should return immediately, took {elapsed:.3f}s"

        print("\n3. Now adding input and trying again...")
        queue_service.put('input', "Test message")
        result = interactive.get_input()
        print(f"   [OK] Received: '{result}'")
        assert result == "Test message", f"Expected 'Test message', got '{result}'"

        queue_service.close()
        print("\n[PASS] Test 5 completed successfully!")

    finally:
        time.sleep(0.2)  # Let Windows release lock file
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def test_different_queue_ids():
    """Test 6: Using different queue IDs for different interactions"""
    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[SKIP] Test 6: Queue services not available")
        return

    print("\n" + "="*80)
    print("TEST 6: Different Queue IDs")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Creating multiple interactive instances with different queue IDs...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)

        agent1 = QueueInteractive(
            system_name="Agent1",
            user_name="User",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='agent1_input',
            response_queue_id='agent1_response'
        )

        agent2 = QueueInteractive(
            system_name="Agent2",
            user_name="User",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='agent2_input',
            response_queue_id='agent2_response'
        )
        print("   [OK] Created 2 agents with different queue IDs")

        print("\n2. Sending messages to different agents...")
        queue_service.put('agent1_input', "Hello Agent 1")
        queue_service.put('agent2_input', "Hello Agent 2")
        print("   [OK] Messages queued")

        print("\n3. Each agent receives their own message...")
        msg1 = agent1.get_input()
        msg2 = agent2.get_input()
        print(f"   Agent 1 received: '{msg1}'")
        print(f"   Agent 2 received: '{msg2}'")
        assert msg1 == "Hello Agent 1"
        assert msg2 == "Hello Agent 2"

        print("\n4. Each agent sends their own response...")
        agent1.send_response("Response from Agent 1", flag=False)
        agent2.send_response("Response from Agent 2", flag=False)

        resp1 = queue_service.get('agent1_response', blocking=False)
        resp2 = queue_service.get('agent2_response', blocking=False)
        print(f"   Agent 1 response: '{resp1}'")
        print(f"   Agent 2 response: '{resp2}'")
        assert resp1 == "Response from Agent 1"
        assert resp2 == "Response from Agent 2"

        queue_service.close()
        print("\n[PASS] Test 6 completed successfully!")

    finally:
        time.sleep(0.2)  # Let Windows release lock file
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def main():
    print("""
==============================================================================
                QueueInteractive Basic Tests
==============================================================================
""")

    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[ERROR] rich_python_utils not available. Cannot run tests.")
        print("Please ensure rich_python_utils is installed and in PYTHONPATH.")
        sys.exit(1)

    tests = [
        test_basic_usage_storage_queue,
        test_basic_usage_thread_queue,
        test_multiple_exchanges,
        test_timeout_behavior,
        test_non_blocking_behavior,
        test_different_queue_ids,
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
