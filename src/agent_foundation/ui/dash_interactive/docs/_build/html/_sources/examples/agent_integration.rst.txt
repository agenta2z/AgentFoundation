Agent Integration Example
==========================

This example demonstrates how to integrate AI agents with the chat interface,
including background processing and queue-based communication.

.. contents:: On This Page
   :local:
   :depth: 2

Agent Factory Pattern
---------------------

Using the agent factory pattern for creating per-session agents:

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs
   from threading import Thread
   import time

   class SimpleAgent:
       """Simple agent that processes messages."""

       def __init__(self, name: str = "Agent"):
           self.name = name
           self.interactive = None  # Set by app

       def process(self, message: str) -> str:
           """Process a message (simulated work)."""
           time.sleep(1)  # Simulate processing
           return f"[{self.name}] Processed: {message}"

       def __call__(self, inputs: dict):
           """Main agent loop - reads from input queue, writes to output queue."""
           while True:
               # Get message from input queue (blocking)
               message = self.interactive.input_queue.get()

               if message is None:  # Shutdown signal
                   break

               # Process and respond
               response = self.process(message)
               self.interactive.response_queue.put(response)

   def create_agent():
       """Factory function that creates new agent instances."""
       return SimpleAgent(name="MyAgent")

   if __name__ == '__main__':
       app = DashInteractiveAppWithLogs(
           title="Agent Chat",
           port=8050
       )
       app.set_agent_factory(create_agent)
       app.run()

Queue-Based Agent Service
-------------------------

Using ``QueueBasedDashInteractiveApp`` with a separate agent service:

.. code-block:: python

   from agent_foundation.ui.dash_interactive import QueueBasedDashInteractiveApp
   from queue import Queue
   from threading import Thread
   import time

   # Shared queues
   request_queue = Queue()
   response_queue = Queue()

   def agent_service():
       """Background agent service."""
       while True:
           request = request_queue.get()
           if request is None:
               break

           session_id = request['session_id']
           message = request['message']

           # Simulate agent processing
           time.sleep(2)
           response = f"Agent response to: {message}"

           response_queue.put({
               'session_id': session_id,
               'response': response
           })

   # Start agent service in background
   service_thread = Thread(target=agent_service, daemon=True)
   service_thread.start()

   def send_to_agent(message: str, session_id: str) -> str:
       """Send message to agent service."""
       request_queue.put({
           'session_id': session_id,
           'message': message
       })
       return "⏳ Processing..."

   def check_responses():
       """Check for agent responses."""
       try:
           data = response_queue.get_nowait()
           return (data['session_id'], data['response'], None)
       except:
           return (None, None, None)

   if __name__ == '__main__':
       app = QueueBasedDashInteractiveApp(
           title="Queue Agent",
           message_handler=send_to_agent,
           response_checker=check_responses,
           special_waiting_message="⏳ Processing..."
       )
       app.run()

Multi-Step Agent with Tool Use
------------------------------

Agent that uses multiple tools:

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs
   from agent_foundation.ui.dash_interactive.utils.log_collector import LogCollector
   import time
   import random

   collector = LogCollector()

   class ToolAgent:
       """Agent with multiple tools."""

       TOOLS = {
           'search': lambda q: f"Search results for '{q}': [result1, result2]",
           'calculate': lambda q: f"Calculation: {eval(q, {'__builtins__': {}})}",
           'weather': lambda q: f"Weather for {q}: Sunny, 72°F",
       }

       def process(self, message: str, session_id: str) -> str:
           # Log agent start
           collector({
               'id': f'Agent_{session_id[:8]}',
               'name': 'Tool Agent',
               'type': 'AgentStarted',
               'item': {'query': message}
           })

           # Determine which tool to use
           tool_name = None
           if 'search' in message.lower():
               tool_name = 'search'
           elif any(c in message for c in '+-*/'):
               tool_name = 'calculate'
           elif 'weather' in message.lower():
               tool_name = 'weather'

           if tool_name:
               # Log tool use
               tool_id = f'Tool_{tool_name}_{random.randint(1000,9999)}'
               collector({
                   'id': tool_id,
                   'name': tool_name.capitalize(),
                   'type': 'ToolCall',
                   'item': {'query': message},
                   'parent_ids': [f'Agent_{session_id[:8]}']
               })

               time.sleep(0.5)  # Simulate tool execution
               result = self.TOOLS[tool_name](message)

               # Log tool result
               collector({
                   'id': tool_id,
                   'name': tool_name.capitalize(),
                   'type': 'ToolResult',
                   'item': {'result': result},
                   'parent_ids': [f'Agent_{session_id[:8]}']
               })

               return result

           return f"I can help with: search, calculate, or weather queries."

   agent = ToolAgent()

   def handler(message: str, session_id: str) -> str:
       return agent.process(message, session_id)

   if __name__ == '__main__':
       app = DashInteractiveAppWithLogs(
           title="Tool Agent",
           message_handler=handler
       )

       # Callback to update log visualization
       @app.app.callback(
           Output('log-data-store', 'data'),
           Input('messages-store', 'data'),
           prevent_initial_call=True
       )
       def update_logs(_):
           return {
               'graph_data': collector.get_graph_structure(),
               'log_groups': dict(collector.log_groups)
           }

       app.run()

Streaming Responses
-------------------

Pattern for streaming agent responses:

.. code-block:: python

   from agent_foundation.ui.dash_interactive import QueueBasedDashInteractiveApp
   from queue import Queue
   from threading import Thread
   import time

   request_queue = Queue()
   response_queue = Queue()

   def streaming_agent():
       """Agent that streams responses word by word."""
       while True:
           request = request_queue.get()
           if request is None:
               break

           session_id = request['session_id']
           message = request['message']

           # Generate response
           words = f"Here is my detailed response to: {message}".split()

           # Stream word by word
           accumulated = ""
           for word in words:
               accumulated += word + " "
               time.sleep(0.1)  # Typing effect

               response_queue.put({
                   'session_id': session_id,
                   'response': accumulated.strip(),
                   'is_streaming': True
               })

           # Final response
           response_queue.put({
               'session_id': session_id,
               'response': accumulated.strip(),
               'is_streaming': False
           })

   # Start agent
   Thread(target=streaming_agent, daemon=True).start()

   def send_message(message: str, session_id: str):
       request_queue.put({'session_id': session_id, 'message': message})
       return "..."

   def check_responses():
       try:
           data = response_queue.get_nowait()
           return (data['session_id'], data['response'], None)
       except:
           return (None, None, None)

   if __name__ == '__main__':
       app = QueueBasedDashInteractiveApp(
           title="Streaming Agent",
           message_handler=send_message,
           response_checker=check_responses,
           special_waiting_message="..."
       )
       app.run()

LangChain Integration
---------------------

Example integrating with LangChain:

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveAppWithLogs

   # Note: Requires langchain installed
   try:
       from langchain.chat_models import ChatOpenAI
       from langchain.schema import HumanMessage, SystemMessage
       HAS_LANGCHAIN = True
   except ImportError:
       HAS_LANGCHAIN = False

   def create_langchain_handler():
       """Create handler using LangChain."""
       if not HAS_LANGCHAIN:
           return lambda msg, sid: "LangChain not installed"

       chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

       def handler(message: str, session_id: str) -> str:
           try:
               messages = [
                   SystemMessage(content="You are a helpful assistant."),
                   HumanMessage(content=message)
               ]
               response = chat(messages)
               return response.content
           except Exception as e:
               return f"Error: {e}"

       return handler

   if __name__ == '__main__':
       app = DashInteractiveAppWithLogs(
           title="LangChain Chat",
           message_handler=create_langchain_handler()
       )
       app.run()

See Also
--------

- :doc:`../applications/dash_interactive_app_with_logs` - DashInteractiveAppWithLogs reference
- :doc:`../applications/queue_based_app` - QueueBasedDashInteractiveApp reference
- :doc:`log_debugging` - Adding log visualization
