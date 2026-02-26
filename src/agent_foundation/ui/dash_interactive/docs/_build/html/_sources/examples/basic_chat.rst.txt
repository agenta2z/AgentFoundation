Basic Chat Example
==================

This example demonstrates how to create a simple chat interface using
``DashInteractiveApp`` with various message handlers.

.. contents:: On This Page
   :local:
   :depth: 2

Simple Echo Bot
---------------

The simplest example - a bot that echoes back user messages:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveApp

   def echo_handler(message: str) -> str:
       """Simple echo handler."""
       return f"You said: {message}"

   if __name__ == '__main__':
       app = DashInteractiveApp(
           title="Echo Bot",
           port=8050,
           debug=True,
           message_handler=echo_handler
       )
       app.run()

Running the Example
^^^^^^^^^^^^^^^^^^^

1. Save as ``echo_bot.py``
2. Run: ``python echo_bot.py``
3. Open http://localhost:8050 in your browser
4. Type a message and click Send (or Ctrl+Enter)

FAQ Bot
-------

A slightly more sophisticated example with predefined responses:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveApp

   FAQ_RESPONSES = {
       'hello': 'Hi there! How can I help you today?',
       'help': 'I can answer questions about our product. Try asking about features, pricing, or support.',
       'features': 'Our product includes: real-time analytics, custom dashboards, and API access.',
       'pricing': 'We offer three plans: Basic ($9/mo), Pro ($29/mo), and Enterprise (custom).',
       'support': 'You can reach support at support@example.com or call 1-800-EXAMPLE.',
       'bye': 'Goodbye! Have a great day!'
   }

   def faq_handler(message: str) -> str:
       """FAQ bot handler with keyword matching."""
       message_lower = message.lower()

       # Check for keywords
       for keyword, response in FAQ_RESPONSES.items():
           if keyword in message_lower:
               return response

       # Default response
       return "I'm not sure about that. Try asking about features, pricing, or support."

   if __name__ == '__main__':
       app = DashInteractiveApp(
           title="FAQ Bot",
           port=8050,
           message_handler=faq_handler
       )
       app.run()

Stateful Chat Bot
-----------------

Example with conversation state tracking:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveApp

   class ConversationBot:
       """Bot that tracks conversation history."""

       def __init__(self):
           self.message_count = 0
           self.topics_discussed = set()

       def __call__(self, message: str) -> str:
           """Process message with state tracking."""
           self.message_count += 1

           # Track topics
           if 'weather' in message.lower():
               self.topics_discussed.add('weather')
               return "I'd love to discuss the weather! It's a beautiful day."
           elif 'sports' in message.lower():
               self.topics_discussed.add('sports')
               return "Sports are exciting! What's your favorite team?"
           elif 'status' in message.lower():
               topics = ', '.join(self.topics_discussed) if self.topics_discussed else 'none yet'
               return f"Messages exchanged: {self.message_count}\\nTopics discussed: {topics}"

           return f"Message #{self.message_count}: {message}"

   if __name__ == '__main__':
       bot = ConversationBot()
       app = DashInteractiveApp(
           title="Stateful Chat",
           port=8050,
           message_handler=bot
       )
       app.run()

External API Integration
------------------------

Example connecting to an external LLM API:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveApp
   import os

   # Note: Install openai package: pip install openai
   try:
       import openai
       openai.api_key = os.getenv('OPENAI_API_KEY')
       HAS_OPENAI = True
   except ImportError:
       HAS_OPENAI = False

   def llm_handler(message: str) -> str:
       """Handler using OpenAI API."""
       if not HAS_OPENAI:
           return "OpenAI package not installed. Run: pip install openai"

       if not openai.api_key:
           return "Please set OPENAI_API_KEY environment variable"

       try:
           response = openai.ChatCompletion.create(
               model="gpt-3.5-turbo",
               messages=[
                   {"role": "system", "content": "You are a helpful assistant."},
                   {"role": "user", "content": message}
               ],
               max_tokens=500
           )
           return response.choices[0].message.content
       except Exception as e:
           return f"API Error: {str(e)}"

   if __name__ == '__main__':
       app = DashInteractiveApp(
           title="AI Chat (GPT)",
           port=8050,
           message_handler=llm_handler
       )
       app.run()

Calculator Bot
--------------

Example with simple expression evaluation:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveApp
   import re

   def calculator_handler(message: str) -> str:
       """Safe calculator handler."""
       # Look for math expressions
       math_pattern = r'^[\d\s\+\-\*/\(\)\.]+$'

       # Clean the message
       expression = message.strip()

       if re.match(math_pattern, expression):
           try:
               # Safely evaluate (only numbers and operators)
               result = eval(expression, {"__builtins__": {}})
               return f"{expression} = {result}"
           except Exception as e:
               return f"Error calculating: {e}"

       # Instructions
       return """
   Calculator Bot - Enter a math expression!

   Examples:
   - 2 + 2
   - 10 * 5
   - (100 - 20) / 4
   - 3.14 * 2 * 2
   """

   if __name__ == '__main__':
       app = DashInteractiveApp(
           title="Calculator Bot",
           port=8050,
           message_handler=calculator_handler
       )
       app.run()

Multiple Sessions Demo
----------------------

Example showing session awareness (requires 2-parameter handler):

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveAppWithLogs
   from collections import defaultdict

   # Per-session state
   session_data = defaultdict(lambda: {'count': 0, 'name': None})

   def session_aware_handler(message: str, session_id: str) -> str:
       """Handler that tracks per-session state."""
       data = session_data[session_id]
       data['count'] += 1

       # Check for name setting
       if message.lower().startswith('my name is '):
           name = message[11:].strip()
           data['name'] = name
           return f"Nice to meet you, {name}! I'll remember that."

       # Personalized greeting
       greeting = f"Hello {data['name']}!" if data['name'] else "Hello!"
       return f"{greeting} This is message #{data['count']} in session {session_id[:8]}..."

   if __name__ == '__main__':
       app = DashInteractiveAppWithLogs(
           title="Session Demo",
           port=8050,
           message_handler=session_aware_handler
       )
       app.run()

Custom Configuration
--------------------

Example with custom port and host settings:

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive import DashInteractiveApp

   def handler(message: str) -> str:
       return f"Processed: {message}"

   if __name__ == '__main__':
       app = DashInteractiveApp(
           title="Custom Config Demo",
           port=9000,  # Custom port
           debug=False  # Production mode (no hot reload)
       )

       # Access underlying Dash app for advanced config
       app.app.config.suppress_callback_exceptions = True

       # Run with custom host
       app.run(host='127.0.0.1')  # Localhost only

See Also
--------

- :doc:`../applications/dash_interactive_app` - DashInteractiveApp reference
- :doc:`agent_integration` - Connecting AI agents
- :doc:`log_debugging` - Adding log visualization
