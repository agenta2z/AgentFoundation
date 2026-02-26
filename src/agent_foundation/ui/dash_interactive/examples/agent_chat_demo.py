"""
Demo of the Dash Interactive UI with an actual agent integration.

This demonstrates:
- Integrating an agent with the UI
- Using QueueInteractive for async communication
- Connecting agent responses to the chat interface

Run this script and navigate to http://localhost:8050 to interact with an agent.
"""
import sys
from pathlib import Path
from typing import Optional
from queue import Queue, Empty
from threading import Thread
import time

# Add source to path if needed
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from science_modeling_tools.ui.dash_interactive import DashInteractiveApp


class MockAgent:
    """
    Mock agent for demonstration purposes.

    In a real implementation, this would be replaced with an actual
    Agent class from agent_foundation.agents
    """

    def __init__(self):
        self.conversation_history = []

    def process_message(self, message: str) -> str:
        """
        Process user message and generate response.

        Args:
            message: User input

        Returns:
            Agent response
        """
        # Store in history
        self.conversation_history.append(('user', message))

        # Simple rule-based responses for demo
        message_lower = message.lower()

        if 'hello' in message_lower or 'hi' in message_lower:
            response = "Hello! I'm a demo agent. How can I help you today?"
        elif 'help' in message_lower:
            response = (
                "I can help you with:\n"
                "- Answering questions\n"
                "- Processing tasks\n"
                "- Debugging logs\n\n"
                "Just type your question or command!"
            )
        elif 'bye' in message_lower:
            response = "Goodbye! Feel free to start a new chat anytime."
        elif '?' in message:
            response = (
                f"That's an interesting question: '{message}'\n\n"
                f"In a real implementation, I would process this using an LLM "
                f"and provide a detailed response based on my knowledge and tools."
            )
        else:
            response = (
                f"I received your message: '{message}'\n\n"
                f"This is a demo agent. In production, I would:\n"
                f"1. Understand your intent\n"
                f"2. Execute relevant actions\n"
                f"3. Provide helpful responses"
            )

        # Store response
        self.conversation_history.append(('assistant', response))

        return response


def create_agent_handler(agent: MockAgent):
    """
    Create a message handler function that uses the agent.

    Args:
        agent: Agent instance to process messages

    Returns:
        Message handler function
    """
    def handler(message: str) -> str:
        try:
            response = agent.process_message(message)
            return response
        except Exception as e:
            return f"Error processing message: {str(e)}"

    return handler


def main():
    """Run the agent chat demo."""
    print("Initializing agent...")
    agent = MockAgent()

    print("Creating Dash app...")
    app = DashInteractiveApp(
        title="Agent Chat Demo",
        port=8050,
        debug=True
    )

    # Set agent handler
    app.set_message_handler(create_agent_handler(agent))

    print("\nAgent is ready!")
    print("=" * 60)
    print("Try these example messages:")
    print("  - Hello")
    print("  - Help")
    print("  - What can you do?")
    print("  - Goodbye")
    print("=" * 60)

    # Run the app
    app.run()


if __name__ == '__main__':
    main()
