"""
Basic demo of the Dash Interactive UI with a simple echo chat bot.

This demonstrates:
- Creating a DashInteractiveApp instance
- Setting up a custom message handler
- Running the application

Run this script and navigate to http://localhost:8050 to see the UI.
"""
import sys
from pathlib import Path

# Add source to path if needed
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agent_foundation.ui.dash_interactive import DashInteractiveApp


def simple_echo_handler(message: str) -> str:
    """
    Simple message handler that echoes back with some formatting.

    Args:
        message: User input message

    Returns:
        Formatted response
    """
    return f"You said: '{message}'\n\nMessage length: {len(message)} characters"


def main():
    """Run the basic chat demo."""
    # Create the app
    app = DashInteractiveApp(
        title="Basic Chat Demo",
        port=8050,
        debug=True
    )

    # Set custom message handler
    app.set_message_handler(simple_echo_handler)

    # Run the app
    app.run()


if __name__ == '__main__':
    main()
