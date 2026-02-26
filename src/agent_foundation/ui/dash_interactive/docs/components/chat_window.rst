==========
ChatWindow
==========

.. module:: science_modeling_tools.ui.dash_interactive.components.chat_window
   :synopsis: Main chat message display and input component

The ``ChatWindow`` component provides the main chat interface with message display
and user input functionality.


Overview
========

This component provides:

* **Message display area** - Scrollable area showing conversation messages
* **User/Assistant differentiation** - Visual distinction between message types
* **Text input** - Textarea for composing messages
* **Send button** - Button to submit messages
* **Typing indicator** - Visual feedback when waiting for responses


Class Definition
================

.. code-block:: python

   class ChatWindow(BaseComponent):
       """
       Component for displaying chat messages and handling user input.

       This component shows the conversation messages in a scrollable area
       with a text input box at the bottom for sending new messages.
       """


Constructor
-----------

.. code-block:: python

   def __init__(
       self,
       component_id: str = "chat-window",
       messages: Optional[List[Dict[str, Any]]] = None,
       placeholder: str = "Send a message...",
       style: Optional[Dict[str, Any]] = None
   )

**Parameters:**

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``component_id``
     - str
     - Unique identifier for this component (default: "chat-window")
   * - ``messages``
     - List[Dict]
     - Initial list of chat messages
   * - ``placeholder``
     - str
     - Placeholder text for input box (default: "Send a message...")
   * - ``style``
     - Dict
     - Optional CSS style overrides


Message Data Structure
======================

Messages are represented as dictionaries:

.. code-block:: python

   message = {
       'role': 'user',        # 'user' or 'assistant'
       'content': 'Hello!',   # Message text
       'timestamp': '12:30:45' # Optional timestamp string
   }

**Special Messages:**

.. code-block:: python

   # Waiting/typing indicator
   waiting_message = {
       'role': 'assistant',
       'content': '__WAITING_FOR_RESPONSE__',  # Special marker
       'timestamp': ''
   }


Layout Structure
================

The component generates this structure:

.. code-block:: text

   ChatWindow
   ├── Messages Area (scrollable)
   │   ├── Welcome Message (if empty)
   │   └── Message List
   │       ├── Message 1
   │       │   ├── Avatar (👤 or 🤖)
   │       │   ├── Content
   │       │   └── Timestamp
   │       ├── Message 2
   │       └── ...
   └── Input Area
       ├── Textarea
       └── Send Button


Generated Element IDs
---------------------

The component generates these element IDs (with ``component_id="chat-window"``):

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Element ID
     - Purpose
   * - ``chat-window``
     - Main container
   * - ``chat-window-messages``
     - Messages display area
   * - ``chat-window-input``
     - Text input textarea
   * - ``chat-window-send-btn``
     - Send button


Methods
=======

layout()
--------

.. code-block:: python

   def layout(self) -> html.Div:
       """Generate the chat window layout."""

Returns a Dash ``html.Div`` containing the complete chat interface.


update_messages()
-----------------

.. code-block:: python

   def update_messages(self, messages: List[Dict[str, Any]]) -> List[html.Div]:
       """
       Update the displayed messages.

       Args:
           messages: New list of message dictionaries

       Returns:
           List of rendered message Div elements
       """

Use this in callbacks to refresh the message display.


add_message()
-------------

.. code-block:: python

   def add_message(
       self,
       role: str,
       content: str,
       timestamp: str = ""
   ) -> List[html.Div]:
       """
       Add a new message to the chat.

       Args:
           role: Message role ('user' or 'assistant')
           content: Message content text
           timestamp: Optional timestamp string

       Returns:
           Updated list of rendered message Div elements
       """


get_callback_inputs()
---------------------

.. code-block:: python

   def get_callback_inputs(self) -> List[Input]:
       """Get list of callback inputs."""

Returns:

* Send button clicks


get_callback_outputs()
----------------------

.. code-block:: python

   def get_callback_outputs(self) -> List[Output]:
       """Get list of callback outputs."""

Returns:

* Messages area children
* Input field value


get_callback_states()
---------------------

.. code-block:: python

   def get_callback_states(self) -> List[State]:
       """Get list of callback states."""

Returns:

* Input field value


Styling
=======

Default Styles
--------------

.. code-block:: python

   default_style = {
       'flex': '1',
       'height': '100%',
       'backgroundColor': '#343541',
       'display': 'flex',
       'flexDirection': 'column',
       'position': 'relative',
       'overflow': 'hidden'
   }


Message Styles
--------------

.. code-block:: python

   # User message
   user_message_style = {
       'backgroundColor': '#40414F',
       'maxWidth': '80%',
       'borderRadius': '8px',
       'padding': '16px',
       'alignSelf': 'flex-end'
   }

   # Assistant message
   assistant_message_style = {
       'backgroundColor': '#444654',
       'maxWidth': '100%',
       'borderRadius': '8px',
       'padding': '16px',
       'alignSelf': 'flex-start'
   }


Input Area Styles
-----------------

.. code-block:: python

   input_style = {
       'backgroundColor': '#40414F',
       'color': '#ECECF1',
       'border': '1px solid #565869',
       'borderRadius': '8px',
       'padding': '12px 16px',
       'fontSize': '15px',
       'resize': 'none'
   }

   send_button_style = {
       'backgroundColor': '#19C37D',
       'color': 'white',
       'border': 'none',
       'borderRadius': '6px',
       'cursor': 'pointer'
   }


Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

   from science_modeling_tools.ui.dash_interactive.components.chat_window import (
       ChatWindow
   )

   # Create chat window
   chat = ChatWindow(
       component_id="main-chat",
       placeholder="Type your message here..."
   )

   # Use in layout
   app.layout = html.Div([
       chat.layout()
   ])


With Initial Messages
---------------------

.. code-block:: python

   chat = ChatWindow(
       component_id="main-chat",
       messages=[
           {
               'role': 'assistant',
               'content': 'Hello! How can I help you today?',
               'timestamp': '10:00:00'
           }
       ]
   )


Handling Message Sending
------------------------

.. code-block:: python

   @app.callback(
       [
           Output('messages-store', 'data'),
           Output('main-chat-input', 'value')
       ],
       Input('main-chat-send-btn', 'n_clicks'),
       [
           State('main-chat-input', 'value'),
           State('messages-store', 'data')
       ],
       prevent_initial_call=True
   )
   def send_message(n_clicks, message_text, messages):
       if not message_text or not message_text.strip():
           return messages, ''

       from datetime import datetime

       # Add user message
       messages.append({
           'role': 'user',
           'content': message_text.strip(),
           'timestamp': datetime.now().strftime('%H:%M:%S')
       })

       # Process and add response (in a real app, this would call your handler)
       response = process_message(message_text)
       messages.append({
           'role': 'assistant',
           'content': response,
           'timestamp': datetime.now().strftime('%H:%M:%S')
       })

       return messages, ''  # Clear input


Updating Display from Store
---------------------------

.. code-block:: python

   @app.callback(
       Output('main-chat-messages', 'children'),
       Input('messages-store', 'data')
   )
   def update_display(messages):
       return chat.update_messages(messages or [])


Showing Typing Indicator
------------------------

.. code-block:: python

   @app.callback(
       Output('messages-store', 'data', allow_duplicate=True),
       Input('main-chat-send-btn', 'n_clicks'),
       State('messages-store', 'data'),
       prevent_initial_call=True
   )
   def show_typing(n_clicks, messages):
       if not n_clicks:
           return messages

       # Add typing indicator
       messages.append({
           'role': 'assistant',
           'content': '__WAITING_FOR_RESPONSE__',
           'timestamp': ''
       })
       return messages


Typing Indicator
================

The component includes a built-in typing indicator that displays when a message
has the special content ``__WAITING_FOR_RESPONSE__``.

The indicator shows animated dots similar to ChatGPT's typing animation:

.. code-block:: python

   # The typing indicator is automatically shown for this message type
   waiting_message = {
       'role': 'assistant',
       'content': '__WAITING_FOR_RESPONSE__'
   }

The animation is defined in CSS:

.. code-block:: css

   @keyframes wave {
       0%, 60%, 100% {
           transform: translateY(0);
           opacity: 0.7;
       }
       30% {
           transform: translateY(-10px);
           opacity: 1;
       }
   }


Welcome Screen
==============

When there are no messages, the component displays a welcome screen:

.. code-block:: text

   ┌─────────────────────────────────────────┐
   │                                         │
   │    Welcome to Interactive Debugger      │
   │                                         │
   │  Start a conversation or load logs      │
   │         for debugging.                  │
   │                                         │
   └─────────────────────────────────────────┘

This is automatically replaced with messages once the conversation begins.


Integration with Application
============================

The ``ChatWindow`` is typically used as part of a ``DashInteractiveApp``:

.. code-block:: python

   class DashInteractiveApp:
       def __init__(self, ...):
           self.chat_window = ChatWindow(
               component_id="chat-window",
               messages=[]
           )

       def _create_layout(self):
           return html.Div([
               self.chat_history.layout(),   # Left sidebar
               self.chat_window.layout()     # Right panel
           ], style={'display': 'flex'})


See Also
========

* :doc:`base` - BaseComponent class
* :doc:`chat_history` - ChatHistoryList component
* :doc:`../applications/dash_interactive_app` - Main application class
