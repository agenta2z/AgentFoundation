"""
Test to verify DashInteractiveAppWithLogs inherits from DashInteractiveApp correctly.
"""
import sys
from pathlib import Path

# Add src to path
# From: test/agent_foundation/ui/dash_interactive/test_dash_interactive_app_inheritance.py
# To: src/
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from agent_foundation.ui.dash_interactive.dash_interactive_app import DashInteractiveApp
from agent_foundation.ui.dash_interactive.dash_interactive_app_with_logs import DashInteractiveAppWithLogs


def test_inheritance():
    """Test that DashInteractiveAppWithLogs properly inherits from DashInteractiveApp."""

    # Test 1: Verify inheritance
    print("Test 1: Checking inheritance...")
    assert issubclass(DashInteractiveAppWithLogs, DashInteractiveApp), "DashInteractiveAppWithLogs should inherit from DashInteractiveApp"
    print("DashInteractiveAppWithLogs correctly inherits from DashInteractiveApp")

    # Test 2: Verify basic instantiation
    print("\nTest 2: Creating DashInteractiveApp instance...")
    app1 = DashInteractiveApp(title="Test App 1", port=8051, debug=False)
    assert app1.title == "Test App 1"
    assert app1.port == 8051
    assert hasattr(app1, 'sessions')
    assert hasattr(app1, 'chat_history')
    assert hasattr(app1, 'chat_window')
    print("DashInteractiveApp instance created successfully")

    # Test 3: Verify extended class instantiation
    print("\nTest 3: Creating DashInteractiveAppWithLogs instance...")
    app2 = DashInteractiveAppWithLogs(title="Test App 2", port=8052, debug=False)
    assert app2.title == "Test App 2"
    assert app2.port == 8052
    # Should have parent attributes
    assert hasattr(app2, 'sessions')
    assert hasattr(app2, 'chat_history')
    # Should have child-specific attributes
    assert hasattr(app2, 'tabbed_panel')
    assert hasattr(app2, 'log_collector')
    assert hasattr(app2, 'session_agents')
    print("DashInteractiveAppWithLogs instance created successfully")

    # Test 4: Verify inherited methods
    print("\nTest 4: Checking inherited methods...")
    assert hasattr(app2, 'run'), "Should have run() method"
    assert hasattr(app2, 'set_message_handler'), "Should have set_message_handler() method"
    assert hasattr(app2, '_register_session_callbacks'), "Should have _register_session_callbacks() method"
    print("All expected methods are present")

    # Test 5: Verify child-specific methods
    print("\nTest 5: Checking child-specific methods...")
    assert hasattr(app2, 'set_agent_factory'), "Should have set_agent_factory() method"
    assert hasattr(app2, '_register_polling_callback'), "Should have _register_polling_callback() method"
    print("All child-specific methods are present")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_inheritance()
