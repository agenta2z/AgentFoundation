===============
ChatHistoryList
===============

.. module:: agent_foundation.ui.dash_interactive.components.chat_history
   :synopsis: Chat session history sidebar component

The ``ChatHistoryList`` component displays a list of chat sessions in a sidebar,
similar to ChatGPT's conversation history.


Overview
========

This component provides:

* **New Chat button** - Create new conversation sessions
* **Session list** - Scrollable list of previous conversations
* **Session selection** - Click to switch between sessions
* **Settings section** - Access to settings and debug mode


Class Definition
================

.. code-block:: python

   class ChatHistoryList(BaseComponent):
       """
       Component for displaying a list of chat sessions/conversations.

       This component shows a vertical list of chat sessions in the left sidebar,
       similar to ChatGPT's conversation history.
       """


Constructor
-----------

.. code-block:: python

   def __init__(
       self,
       component_id: str = "chat-history",
       sessions: Optional[List[Dict[str, Any]]] = None,
       show_settings: bool = True,
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
     - Unique identifier for this component (default: "chat-history")
   * - ``sessions``
     - List[Dict]
     - Initial list of chat sessions
   * - ``show_settings``
     - bool
     - Whether to display settings section (default: True)
   * - ``style``
     - Dict
     - Optional CSS style overrides


Session Data Structure
======================

Sessions are represented as dictionaries:

.. code-block:: python

   session = {
       'id': 'session_1_20241216120000',  # Unique identifier
       'title': 'Chat about Python',       # Display title
       'timestamp': '2024-12-16 12:00',    # Creation timestamp
       'active': True                      # Whether currently selected
   }


Layout Structure
================

The component generates this structure:

.. code-block:: text

   ChatHistoryList
   ├── Header
   │   └── "+ New Chat" button
   ├── Session List (scrollable)
   │   ├── Session Item 1
   │   │   ├── Title
   │   │   ├── Timestamp
   │   │   └── Session ID
   │   ├── Session Item 2
   │   └── ...
   └── Settings Section (optional)
       ├── "⚙️ Settings" button
       └── "📊 Debug Mode" toggle


Generated Element IDs
---------------------

The component generates these element IDs (with ``component_id="chat-history"``):

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Element ID
     - Purpose
   * - ``chat-history``
     - Main container
   * - ``chat-history-new-chat-btn``
     - New Chat button
   * - ``chat-history-session-list``
     - Session list container
   * - ``chat-history-session-item``
     - Pattern-matching ID for session items
   * - ``chat-history-settings-btn``
     - Settings button
   * - ``chat-history-debug-toggle``
     - Debug mode toggle


Methods
=======

layout()
--------

.. code-block:: python

   def layout(self) -> html.Div:
       """Generate the chat history sidebar layout."""

Returns a Dash ``html.Div`` containing the complete sidebar structure.


update_sessions()
-----------------

.. code-block:: python

   def update_sessions(self, sessions: List[Dict[str, Any]]) -> List[html.Div]:
       """
       Update the session list.

       Args:
           sessions: New list of session dictionaries

       Returns:
           List of rendered session Div elements
       """

Use this method to refresh the session list when sessions change.


get_callback_inputs()
---------------------

.. code-block:: python

   def get_callback_inputs(self) -> List[Input]:
       """Get list of callback inputs."""

Returns inputs for:

* New chat button clicks
* Session item clicks (pattern-matching)
* Settings button clicks
* Debug toggle clicks


get_callback_outputs()
----------------------

.. code-block:: python

   def get_callback_outputs(self) -> List[Output]:
       """Get list of callback outputs."""

Returns output for the session list children.


Styling
=======

Default Styles
--------------

.. code-block:: python

   default_style = {
       'width': '300px',
       'height': '100vh',
       'backgroundColor': '#202123',
       'color': '#ECECF1',
       'overflowY': 'auto',
       'display': 'flex',
       'flexDirection': 'column',
       'borderRight': '1px solid #4D4D4F'
   }


Session Item Styles
-------------------

Sessions have different styles based on state:

.. code-block:: python

   # Normal state
   normal_style = {
       'backgroundColor': 'transparent'
   }

   # Active/Selected state
   active_style = {
       'backgroundColor': '#343541'
   }


Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

   from agent_foundation.ui.dash_interactive.components.chat_history import (
       ChatHistoryList
   )

   # Create with initial sessions
   history = ChatHistoryList(
       component_id="sidebar",
       sessions=[
           {
               'id': 'session_1',
               'title': 'First Chat',
               'timestamp': '2024-12-16 10:00',
               'active': True
           },
           {
               'id': 'session_2',
               'title': 'Second Chat',
               'timestamp': '2024-12-16 11:00',
               'active': False
           }
       ]
   )

   # Use in layout
   app.layout = html.Div([
       history.layout(),
       # ... main content
   ])


Handling Session Selection
--------------------------

.. code-block:: python

   @app.callback(
       Output('current-session-store', 'data'),
       Input({'type': 'sidebar-session-item', 'index': ALL}, 'n_clicks'),
       State('sessions-store', 'data')
   )
   def select_session(n_clicks_list, sessions):
       ctx = dash.callback_context
       if not ctx.triggered or not any(n_clicks_list):
           return dash.no_update

       # Get clicked session ID
       import json
       clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
       session_id = json.loads(clicked_id)['index']

       return session_id


Handling New Chat
-----------------

.. code-block:: python

   @app.callback(
       [
           Output('sessions-store', 'data'),
           Output('current-session-store', 'data')
       ],
       Input('sidebar-new-chat-btn', 'n_clicks'),
       State('sessions-store', 'data')
   )
   def create_new_session(n_clicks, sessions):
       if not n_clicks:
           return sessions, dash.no_update

       from datetime import datetime

       # Create new session
       session_id = f"session_{len(sessions)+1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
       new_session = {
           'id': session_id,
           'title': f'New Chat {len(sessions)+1}',
           'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
           'active': True
       }

       # Deactivate others
       for s in sessions:
           s['active'] = False

       sessions.append(new_session)
       return sessions, session_id


Custom Styling
--------------

.. code-block:: python

   # Create with custom styles
   history = ChatHistoryList(
       component_id="sidebar",
       style={
           'width': '350px',                    # Wider sidebar
           'backgroundColor': '#1a1a2e',        # Custom color
           'borderRight': '2px solid #00ff88'   # Custom border
       }
   )


Without Settings Section
------------------------

.. code-block:: python

   # Hide settings section
   history = ChatHistoryList(
       component_id="sidebar",
       show_settings=False
   )


Integration with Application
============================

The ``ChatHistoryList`` is typically used as part of a ``DashInteractiveApp``:

.. code-block:: python

   class DashInteractiveApp:
       def __init__(self, ...):
           self.chat_history = ChatHistoryList(
               component_id="chat-history",
               sessions=[]
           )

       def _create_layout(self):
           return html.Div([
               self.chat_history.layout(),  # Left sidebar
               self.chat_window.layout()    # Right panel
           ], style={'display': 'flex'})


See Also
========

* :doc:`base` - BaseComponent class
* :doc:`chat_window` - ChatWindow component
* :doc:`../applications/dash_interactive_app` - Main application class
