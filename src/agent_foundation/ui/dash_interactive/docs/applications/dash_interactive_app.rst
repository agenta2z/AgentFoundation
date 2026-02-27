DashInteractiveApp
==================

The ``DashInteractiveApp`` is the base application class providing a GPT-like chat interface
with session management and a customizable message handler.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

``DashInteractiveApp`` provides:

- Left sidebar with chat history and session management
- Right panel with chat window, message display, and input
- Session persistence using Dash stores
- Customizable message handling

This is the foundational class that :doc:`dash_interactive_app_with_logs` extends.

Class Reference
---------------

.. py:class:: DashInteractiveApp(title="Interactive Debugger", port=8050, debug=True, message_handler=None)

   Main Dash application for GPT-like chat interface.

   :param str title: Application title (shown in browser tab)
   :param int port: Port number to run the server on
   :param bool debug: Whether to run in debug mode with hot reloading
   :param callable message_handler: Optional callback function to handle user messages

   .. py:attribute:: app
      :type: dash.Dash

      The underlying Dash application instance.

   .. py:attribute:: chat_history
      :type: ChatHistoryList

      Left sidebar component for session management.

   .. py:attribute:: chat_window
      :type: ChatWindow

      Right panel chat interface component.

   .. py:attribute:: sessions
      :type: List[Dict]

      List of chat sessions with metadata.

   .. py:attribute:: current_session_id
      :type: str

      ID of the currently active session.

   .. py:attribute:: session_messages
      :type: Dict[str, List]

      Mapping of session IDs to message lists.

Quick Start
-----------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveApp

   # Create application with default echo handler
   app = DashInteractiveApp(
       title="My Chat App",
       port=8050
   )

   # Run the application
   app.run()

With Custom Handler
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveApp

   def my_message_handler(message: str) -> str:
       """Process user message and return response."""
       # Your custom logic here
       if "hello" in message.lower():
           return "Hi there! How can I help you today?"
       return f"You said: {message}"

   app = DashInteractiveApp(
       title="My Assistant",
       message_handler=my_message_handler
   )
   app.run(debug=True)

Message Handler Interface
-------------------------

The message handler is a callable that receives the user's message and returns a response.

Simple Handler
^^^^^^^^^^^^^^

.. code-block:: python

   def simple_handler(message: str) -> str:
       return f"Received: {message}"

Handler with External API
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import openai

   def llm_handler(message: str) -> str:
       response = openai.ChatCompletion.create(
           model="gpt-4",
           messages=[{"role": "user", "content": message}]
       )
       return response.choices[0].message.content

Handler with State
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class StatefulHandler:
       def __init__(self):
           self.history = []

       def __call__(self, message: str) -> str:
           self.history.append(message)
           return f"Message #{len(self.history)}: {message}"

   handler = StatefulHandler()
   app = DashInteractiveApp(message_handler=handler)

Changing Handler at Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   app = DashInteractiveApp()

   # Later, set a different handler
   app.set_message_handler(my_new_handler)

Application Layout
------------------

The application uses a flexbox layout:

.. code-block:: text

   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │  ┌─────────────────┐  ┌────────────────────────────────────────┐  │
   │  │                 │  │                                        │  │
   │  │  ChatHistoryList│  │           ChatWindow                   │  │
   │  │                 │  │                                        │  │
   │  │  [+ New Chat]   │  │  ┌────────────────────────────────┐   │  │
   │  │                 │  │  │                                │   │  │
   │  │  Session 1      │  │  │    Message Display Area        │   │  │
   │  │  Session 2      │  │  │                                │   │  │
   │  │  Session 3      │  │  │                                │   │  │
   │  │                 │  │  └────────────────────────────────┘   │  │
   │  │                 │  │                                        │  │
   │  │                 │  │  ┌────────────────────┐ ┌────────┐   │  │
   │  │                 │  │  │    Input Field     │ │  Send  │   │  │
   │  │                 │  │  └────────────────────┘ └────────┘   │  │
   │  │                 │  │                                        │  │
   │  └─────────────────┘  └────────────────────────────────────────┘  │
   │                                                                    │
   └────────────────────────────────────────────────────────────────────┘

Session Management
------------------

Session Data Structure
^^^^^^^^^^^^^^^^^^^^^^

Each session is a dictionary:

.. code-block:: python

   {
       'id': 'session_1_20240115103045',
       'title': 'Hello there...',
       'timestamp': '2024-01-15 10:30',
       'active': True
   }

Session Lifecycle
^^^^^^^^^^^^^^^^^

1. **Creation**: Click "+ New Chat" button or auto-created on first message
2. **Selection**: Click session in sidebar to switch
3. **Title Update**: Automatically set from first message (first 5 words)
4. **Persistence**: Sessions stored in ``dcc.Store`` (client-side)

Data Stores
^^^^^^^^^^^

The application uses three Dash stores:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Store ID
     - Purpose
   * - ``sessions-store``
     - List of all session metadata
   * - ``current-session-store``
     - ID of the currently active session
   * - ``messages-store``
     - Dictionary mapping session IDs to message lists

Message Format
^^^^^^^^^^^^^^

.. code-block:: python

   {
       'role': 'user',       # or 'assistant'
       'content': 'Hello!',
       'timestamp': '10:30:45'
   }

Callbacks
---------

The application registers several callbacks:

Session Callbacks
^^^^^^^^^^^^^^^^^

- ``create_new_session``: Creates a new chat session
- ``update_history_display``: Updates the sidebar session list
- ``select_session``: Handles session selection clicks

Message Callbacks
^^^^^^^^^^^^^^^^^

- ``load_session_messages``: Loads messages when switching sessions
- ``send_message``: Processes new messages and gets responses

Customization
-------------

Custom Styling
^^^^^^^^^^^^^^

Override component styles by accessing components:

.. code-block:: python

   app = DashInteractiveApp()

   # Customize chat window style
   app.chat_window.style['backgroundColor'] = '#2D2D2D'

   # Customize sidebar width (modify layout)
   app.chat_history.style['width'] = '300px'

Extended Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   app = DashInteractiveApp(
       title="Production Assistant",
       port=8080,
       debug=False  # Disable for production
   )

   # Access underlying Dash app for advanced configuration
   app.app.config.suppress_callback_exceptions = True

Running the Application
-----------------------

Development Mode
^^^^^^^^^^^^^^^^

.. code-block:: python

   app.run(debug=True)  # Hot reloading enabled

Production Mode
^^^^^^^^^^^^^^^

.. code-block:: python

   app = DashInteractiveApp(debug=False)
   app.run(host='0.0.0.0')

With Custom Host/Port
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   app = DashInteractiveApp(port=9000)
   app.run(host='127.0.0.1')

Complete Example
----------------

.. code-block:: python

   from agent_foundation.ui.dash_interactive import DashInteractiveApp
   import random

   # Example responses
   RESPONSES = [
       "That's an interesting question! Let me think about that...",
       "Great point! Here's what I think...",
       "I understand. Let me help you with that.",
       "Could you tell me more about what you mean?",
   ]

   def chatbot_handler(message: str) -> str:
       """Simple chatbot with random responses."""
       # Simulate processing
       if "help" in message.lower():
           return "I'm here to help! You can ask me anything."
       if "bye" in message.lower():
           return "Goodbye! Have a great day!"
       return random.choice(RESPONSES) + f"\n\nYou asked: {message}"

   if __name__ == '__main__':
       app = DashInteractiveApp(
           title="Simple Chatbot",
           port=8050,
           debug=True,
           message_handler=chatbot_handler
       )

       print("Starting chatbot...")
       app.run()

Methods Reference
-----------------

.. py:method:: run(host='0.0.0.0')

   Run the Dash application server.

   :param str host: Host address to bind to

.. py:method:: set_message_handler(handler)

   Set a custom message handler function.

   :param callable handler: Function that takes a message string and returns a response string

.. py:method:: _create_layout()

   Create the main application layout.

   :return: Dash Div containing the full layout

.. py:method:: _register_callbacks()

   Register all Dash callbacks for interactivity.

.. py:method:: _register_session_callbacks()

   Register callbacks for session management.

.. py:method:: _register_message_callbacks()

   Register callbacks for message handling.

.. py:method:: _default_message_handler(message)

   Default message handler that echoes back the message.

   :param str message: User input message
   :return: Echo response

See Also
--------

- :doc:`dash_interactive_app_with_logs` - Extended version with log visualization
- :doc:`queue_based_app` - Async version using queues
- :doc:`../components/chat_history` - ChatHistoryList component documentation
- :doc:`../components/chat_window` - ChatWindow component documentation
