"""
Simple usage example of QueueInteractive

Demonstrates basic usage patterns:
- Creating QueueInteractive with different queue services
- Sending and receiving messages
- Using different queue IDs
- Timeout and blocking behavior

Prerequisites:
    rich_python_utils package installed

Usage:
    python example_simple_usage.py
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
# From: examples/agent_foundation/ui/queue_interactive/example_simple_usage.py
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
    from rich_python_utils.service_utils.queue_service.thread_queue_service import (
        ThreadQueueService
    )
    QUEUE_SERVICES_AVAILABLE = True
except ImportError:
    QUEUE_SERVICES_AVAILABLE = False
    print("Error: rich_python_utils not available")
    sys.exit(1)


def example_1_basic_usage():
    """Example 1: Basic usage with StorageBasedQueueService"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage with StorageBasedQueueService")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Creating queue service and interactive...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)

        interactive = QueueInteractive(
            system_name="Assistant",
            user_name="User",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='input',
            response_queue_id='response'
        )
        print(f"   [OK] Queue service created at {tmpdir}")
        print(f"   [OK] Interactive created with queues: 'input' -> 'response'")

        print("\n2. Simulating user input...")
        queue_service.put('input', "Hello! Can you help me?")
        print("   [User] Hello! Can you help me?")

        print("\n3. Agent processing input...")
        user_message = interactive.get_input()
        print(f"   [Agent] Received: '{user_message}'")

        print("\n4. Agent sending response...")
        interactive.send_response(
            "Of course! I'm here to help. What do you need assistance with?",
            flag=False
        )
        print("   [Agent] Sent response")

        print("\n5. Getting response...")
        response = queue_service.get('response', blocking=False)
        print(f"   [User] Received: '{response}'")

        queue_service.close()
        print("\n[OK] Example 1 completed!")

    finally:
        import time
        time.sleep(0.2)
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def example_2_conversation_loop():
    """Example 2: Multi-turn conversation"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multi-turn Conversation")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Setting up conversation...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)

        interactive = QueueInteractive(
            system_name="ChatBot",
            user_name="User",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='chat_input',
            response_queue_id='chat_output'
        )

        # Simulate conversation
        conversation = [
            ("Hi there!", "Hello! How can I assist you today?"),
            ("What's the capital of France?", "The capital of France is Paris."),
            ("Thanks!", "You're welcome! Feel free to ask if you need anything else."),
        ]

        for i, (user_msg, bot_response) in enumerate(conversation, 1):
            print(f"\n{i}. Turn {i}:")
            print(f"   User: {user_msg}")
            queue_service.put('chat_input', user_msg)

            # Bot processes
            received = interactive.get_input()
            print(f"   [Bot received: '{received}']")

            interactive.send_response(bot_response, flag=False)
            response = queue_service.get('chat_output', blocking=False)
            print(f"   Bot: {response}")

        queue_service.close()
        print("\n[OK] Example 2 completed!")

    finally:
        import time
        time.sleep(0.2)
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def example_3_timeout_behavior():
    """Example 3: Timeout and non-blocking behavior"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Timeout and Non-blocking Behavior")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Creating interactive with 2-second timeout...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)

        interactive = QueueInteractive(
            system_name="Agent",
            user_name="User",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='input',
            response_queue_id='output',
            blocking=True,
            timeout=2.0
        )
        print("   [OK] Timeout set to 2 seconds")

        print("\n2. Attempting to get input from empty queue (will timeout)...")
        import time
        start = time.time()
        result = interactive.get_input()
        elapsed = time.time() - start
        print(f"   [OK] Timed out after {elapsed:.2f}s")
        print(f"   [OK] Returned: '{result}' (empty string)")

        print("\n3. Switching to non-blocking mode...")
        interactive.blocking = False

        print("\n4. Attempting non-blocking get (should return immediately)...")
        start = time.time()
        result = interactive.get_input()
        elapsed = time.time() - start
        print(f"   [OK] Returned after {elapsed:.3f}s")
        print(f"   [OK] Returned: '{result}' (empty string)")

        queue_service.close()
        print("\n[OK] Example 3 completed!")

    finally:
        import time
        time.sleep(0.2)
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def example_4_multiple_queue_ids():
    """Example 4: Using different queue IDs for different purposes"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Multiple Queue IDs")
    print("="*80)

    tmpdir = tempfile.mkdtemp()

    try:
        print("\n1. Creating multiple interactive instances...")
        queue_service = StorageBasedQueueService(root_path=tmpdir)

        # Customer service agent
        customer_agent = QueueInteractive(
            system_name="CustomerService",
            user_name="Customer",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='customer_input',
            response_queue_id='customer_response'
        )

        # Technical support agent
        tech_agent = QueueInteractive(
            system_name="TechSupport",
            user_name="User",
            input_queue=queue_service,
            response_queue=queue_service,
            input_queue_id='tech_input',
            response_queue_id='tech_response'
        )

        print("   [OK] Created customer service agent (customer_input -> customer_response)")
        print("   [OK] Created tech support agent (tech_input -> tech_response)")

        print("\n2. Sending messages to customer service...")
        queue_service.put('customer_input', "I need help with my order")
        customer_msg = customer_agent.get_input()
        print(f"   [Customer] I need help with my order")
        print(f"   [CustomerService received: '{customer_msg}']")

        customer_agent.send_response(
            "I'd be happy to help with your order. What's your order number?",
            flag=False
        )
        response = queue_service.get('customer_response', blocking=False)
        print(f"   [CustomerService] {response}")

        print("\n3. Sending messages to tech support...")
        queue_service.put('tech_input', "My app is crashing")
        tech_msg = tech_agent.get_input()
        print(f"   [User] My app is crashing")
        print(f"   [TechSupport received: '{tech_msg}']")

        tech_agent.send_response(
            "I'll help you troubleshoot. What error message do you see?",
            flag=False
        )
        response = queue_service.get('tech_response', blocking=False)
        print(f"   [TechSupport] {response}")

        queue_service.close()
        print("\n[OK] Example 4 completed!")

    finally:
        import time
        time.sleep(0.2)
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def example_5_thread_queue_service():
    """Example 5: Using ThreadQueueService (in-process only)"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Using ThreadQueueService")
    print("="*80)

    print("\n1. Creating ThreadQueueService...")
    queue_service = ThreadQueueService()

    interactive = QueueInteractive(
        system_name="Assistant",
        user_name="User",
        input_queue=queue_service,
        response_queue=queue_service,
        input_queue_id='input',
        response_queue_id='output'
    )
    print("   [OK] ThreadQueueService created (in-memory, same-process only)")

    print("\n2. Quick exchange...")
    queue_service.put('input', "Hello!")
    msg = interactive.get_input()
    print(f"   [User] Hello!")
    print(f"   [Assistant received: '{msg}']")

    interactive.send_response("Hi there!", flag=False)
    response = queue_service.get('output', blocking=False)
    print(f"   [Assistant] {response}")

    queue_service.close()
    print("\n[OK] Example 5 completed!")


def main():
    print("""
==============================================================================
                QueueInteractive Simple Usage Examples
==============================================================================
""")

    if not QUEUE_SERVICES_AVAILABLE:
        print("\n[ERROR] rich_python_utils not available")
        print("Please install rich_python_utils first.")
        sys.exit(1)

    examples = [
        example_1_basic_usage,
        example_2_conversation_loop,
        example_3_timeout_behavior,
        example_4_multiple_queue_ids,
        example_5_thread_queue_service,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n[ERROR] {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("[OK] All examples completed successfully!")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
