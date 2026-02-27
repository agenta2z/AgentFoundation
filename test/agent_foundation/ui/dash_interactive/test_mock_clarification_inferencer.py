"""
Test script for the mock clarification inferencer in the debugger UI.

This demonstrates how to use the Settings tab to test clarification flows
with the MockClarificationInferencer.
"""
import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
src_dir = os.path.join(os.path.dirname(test_dir), 'src')

# Add SciencePythonUtils to path (at same level as ScienceModelingTools)
python_projects_dir = os.path.dirname(os.path.dirname(test_dir))
rich_python_utils_path = os.path.join(python_projects_dir, 'SciencePythonUtils', 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if rich_python_utils_path not in sys.path:
    sys.path.insert(0, rich_python_utils_path)

from agent_foundation.ui.dash_interactive.dash_interactive_app_with_logs import DashInteractiveAppWithLogs
from agent_foundation.common.inferencers.mock_inferencers import MockClarificationInferencer
from agent_foundation.agents.agent_response import AgentResponse, AgentAction

def create_mock_message_handler(inferencer):
    """
    Create a message handler that uses the mock inferencer.

    Args:
        inferencer: MockClarificationInferencer instance

    Returns:
        Callable message handler function
    """
    def mock_message_handler(message: str, session_id: str, all_session_ids: list):
        """Handle messages using mock clarification inferencer."""
        try:
            # Call the inferencer
            response = inferencer(reasoner_input=message, reasoner_config=None)

            # Format the response for display
            if isinstance(response, AgentResponse):
                instant_response = response.instant_response

                # Check for clarification actions
                if response.next_actions:
                    for action_group in response.next_actions:
                        for action in action_group:
                            if isinstance(action, AgentAction) and action.type.startswith('Clarification.'):
                                # Return both instant response and clarification
                                return [instant_response, action.target]

                return instant_response
            else:
                return str(response)

        except Exception as e:
            import traceback
            return f"Error in mock inferencer: {str(e)}\n{traceback.format_exc()}"

    return mock_message_handler


def main():
    """Run the debugger UI with mock clarification inferencer."""
    print("Starting Dash Interactive Debugger with Mock Clarification Inferencer...")
    print("\nInstructions:")
    print("1. Open the app in your browser at http://localhost:8050")
    print("2. Type a message like 'Order pasta, sauce, and garlic bread'")
    print("3. Observe the clarification question flow:")
    print("   - First message: Returns a clarification question with HTML formatting")
    print("   - View the Response Monitor tab (bottom-right panel) to see the structured response")
    print("   - Reply with location and service (e.g., 'Seattle, WA 98121, use Instacart')")
    print("   - Second message: Returns a completion response")
    print("\nWhat to look for:")
    print("- Chat window displays both instant response and clarification question")
    print("- Response Monitor shows the list structure: [instant_response, clarification_html]")
    print("- Clarification HTML is properly formatted with bullet points")
    print()

    # Create the mock inferencer
    mock_inferencer = MockClarificationInferencer()

    # Create the app
    app = DashInteractiveAppWithLogs(
        title="Mock Clarification Inferencer Test",
        port=8050,
        debug=True,
        message_handler=create_mock_message_handler(mock_inferencer)
    )

    # Run the app
    print("Starting server on http://localhost:8050")
    print("Press Ctrl+C to stop\n")
    app.run()


if __name__ == '__main__':
    main()
