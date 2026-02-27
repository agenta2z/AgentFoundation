QueueBasedDashInteractiveApp
=============================

The ``QueueBasedDashInteractiveApp`` is a specialized variant that polls responses
from a shared queue service instead of per-session queues, ideal for distributed
agent architectures.

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

``QueueBasedDashInteractiveApp`` extends ``DashInteractiveAppWithLogs`` with:

- Custom response polling from shared queue service
- Configurable response checker function
- Placeholder message support while waiting for responses
- Seamless integration with external agent services

Use this application when:

- Your agents run in a separate service/process
- Multiple UI instances share a common queue
- You need decoupled architecture between UI and agent execution

Class Reference
---------------

.. py:class:: QueueBasedDashInteractiveApp(title="Interactive Debugger with Logs", port=8050, debug=True, message_handler=None, response_checker=None, special_waiting_message="__WAITING_FOR_RESPONSE__", custom_monitor_tabs=None, custom_main_tabs=None)

   Dash app that polls from a shared queue service instead of per-session queues.

   :param str title: Application title
   :param int port: Port number for the server
   :param bool debug: Enable debug mode
   :param callable message_handler: Custom message handler function
   :param callable response_checker: Function that checks for responses from queue
   :param str special_waiting_message: Placeholder message shown while waiting
   :param list custom_monitor_tabs: Optional custom monitor tabs
   :param list custom_main_tabs: Optional custom main tabs

   **Key Attributes:**

   .. py:attribute:: response_checker
      :type: Callable[[], Tuple]

      Function that polls the queue and returns ``(session_id, response, log_collector)``
      or ``(None, None, None)`` if no response is available.

   .. py:attribute:: special_waiting_message
      :type: str

      Placeholder text shown in the chat while waiting for agent response.
      Defaults to ``"__WAITING_FOR_RESPONSE__"``.

Quick Start
-----------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from agent_foundation.ui.dash_interactive import QueueBasedDashInteractiveApp

   def check_responses():
       """Check for responses from the queue service."""
       # Your queue polling logic here
       response_data = queue_service.get('response_queue', blocking=False)
       if response_data:
           return (
               response_data.get('session_id'),
               response_data.get('response'),
               response_data.get('log_collector')
           )
       return (None, None, None)

   app = QueueBasedDashInteractiveApp(
       title="Queue-Based Chat",
       response_checker=check_responses
   )
   app.run()

Response Checker Function
-------------------------

The response checker is the core mechanism for integrating with external services.

Signature
^^^^^^^^^

.. code-block:: python

   def response_checker() -> Tuple[Optional[str], Optional[str], Optional[LogCollector]]:
       """
       Check for available responses from the queue.

       Returns:
           Tuple containing:
           - session_id: Target session for the response (or None)
           - response: Response text content (or None)
           - log_collector: Optional LogCollector with execution logs (or None)
       """
       pass

Example Implementations
^^^^^^^^^^^^^^^^^^^^^^^

**Simple Queue Service:**

.. code-block:: python

   from queue import Queue, Empty

   response_queue = Queue()

   def check_responses():
       try:
           data = response_queue.get_nowait()
           return (
               data.get('session_id'),
               data.get('response'),
               data.get('log_collector')
           )
       except Empty:
           return (None, None, None)

**Redis-Based Queue:**

.. code-block:: python

   import redis
   import json

   redis_client = redis.Redis()

   def check_responses():
       result = redis_client.lpop('agent_responses')
       if result:
           data = json.loads(result)
           return (
               data['session_id'],
               data['response'],
               None  # Logs not available from Redis
           )
       return (None, None, None)

**StorageBasedQueueService:**

.. code-block:: python

   from queue_service import StorageBasedQueueService

   queue_service = StorageBasedQueueService()

   def check_responses():
       response_data = queue_service.get('agent_response', blocking=False, timeout=0)
       if response_data:
           return (
               response_data.get('session_id'),
               response_data.get('response', ''),
               response_data.get('log_collector')
           )
       return (None, None, None)

Setting Response Checker at Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   app = QueueBasedDashInteractiveApp()

   # Later, set the response checker
   app.set_response_checker(my_check_function)

Waiting Message Behavior
------------------------

The ``special_waiting_message`` parameter controls placeholder behavior:

How It Works
^^^^^^^^^^^^

1. When a user sends a message, a placeholder message can be added immediately
2. The polling callback checks if the last assistant message matches the waiting message
3. If it matches, the placeholder is replaced with the actual response
4. If it doesn't match, a new message is appended

Custom Waiting Message
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   app = QueueBasedDashInteractiveApp(
       special_waiting_message="⏳ Processing your request..."
   )

Message Handler with Waiting Indicator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def message_handler(message: str, session_id: str):
       """Handler that shows waiting message immediately."""
       # Send message to agent service
       agent_service.send({
           'session_id': session_id,
           'message': message
       })

       # Return waiting placeholder
       # This will be replaced when response_checker gets actual response
       return "__WAITING_FOR_RESPONSE__"

   app = QueueBasedDashInteractiveApp(
       message_handler=message_handler,
       special_waiting_message="__WAITING_FOR_RESPONSE__"
   )

Polling Behavior
----------------

The callback polls every second via ``response-poll-interval``:

.. code-block:: python

   @self.app.callback(
       Output('messages-store', 'data', allow_duplicate=True),
       [Input('response-poll-interval', 'n_intervals')],
       [
           State('current-session-store', 'data'),
           State('messages-store', 'data'),
           State('page-visibility-store', 'data')
       ],
       prevent_initial_call=True
   )
   def poll_and_refresh_messages(n_intervals, session_id, messages_store, visibility_data):
       # Polling logic here
       pass

Response Routing
^^^^^^^^^^^^^^^^

Responses are routed to sessions based on the ``session_id`` returned by the checker:

.. code-block:: python

   if response_session_id and response_session_id in messages_store:
       # Update messages for this specific session
       messages_store[response_session_id].append(new_message)

Architecture Integration
------------------------

Typical Distributed Setup
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   ┌─────────────────┐       ┌─────────────────┐
   │                 │       │                 │
   │  QueueBased     │──────▶│  Message Queue  │
   │  DashApp        │       │  (Redis/etc)    │
   │                 │◀──────│                 │
   └─────────────────┘       └────────┬────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │                 │
                             │  Agent Service  │
                             │  (Background)   │
                             │                 │
                             └─────────────────┘

Multi-Process Example
^^^^^^^^^^^^^^^^^^^^^

**UI Process:**

.. code-block:: python

   # ui_app.py
   import redis
   from agent_foundation.ui.dash_interactive import QueueBasedDashInteractiveApp

   redis_client = redis.Redis()

   def send_to_agent(message, session_id):
       redis_client.rpush('user_messages', json.dumps({
           'session_id': session_id,
           'message': message
       }))
       return "__WAITING__"

   def check_agent_response():
       result = redis_client.lpop('agent_responses')
       if result:
           data = json.loads(result)
           return (data['session_id'], data['response'], None)
       return (None, None, None)

   app = QueueBasedDashInteractiveApp(
       message_handler=send_to_agent,
       response_checker=check_agent_response,
       special_waiting_message="__WAITING__"
   )
   app.run()

**Agent Process:**

.. code-block:: python

   # agent_worker.py
   import redis

   redis_client = redis.Redis()

   while True:
       # Block waiting for messages
       _, data = redis_client.blpop('user_messages')
       request = json.loads(data)

       # Process with agent
       response = agent.process(request['message'])

       # Send response back
       redis_client.rpush('agent_responses', json.dumps({
           'session_id': request['session_id'],
           'response': response
       }))

Complete Example
----------------

.. code-block:: python

   from queue import Queue
   from threading import Thread
   from agent_foundation.ui.dash_interactive import QueueBasedDashInteractiveApp

   # Shared queues
   request_queue = Queue()
   response_queue = Queue()

   # Background agent worker
   def agent_worker():
       while True:
           request = request_queue.get()
           if request is None:
               break

           session_id = request['session_id']
           message = request['message']

           # Simulate agent processing
           import time
           time.sleep(2)  # Simulate work
           response = f"Agent processed: {message}"

           response_queue.put({
               'session_id': session_id,
               'response': response
           })

   # Start worker thread
   worker = Thread(target=agent_worker, daemon=True)
   worker.start()

   # Message handler - sends to worker
   def handle_message(message: str, session_id: str):
       request_queue.put({
           'session_id': session_id,
           'message': message
       })
       return "⏳ Processing your request..."

   # Response checker
   def check_responses():
       try:
           data = response_queue.get_nowait()
           return (data['session_id'], data['response'], None)
       except:
           return (None, None, None)

   # Create and run app
   if __name__ == '__main__':
       app = QueueBasedDashInteractiveApp(
           title="Queue-Based Agent Chat",
           message_handler=handle_message,
           response_checker=check_responses,
           special_waiting_message="⏳ Processing your request..."
       )
       app.run()

Methods Reference
-----------------

.. py:method:: set_response_checker(checker)

   Set the response checker function.

   :param callable checker: Function that returns ``(session_id, response, log_collector)``
                           or ``(None, None, None)``

.. py:method:: _register_polling_callback()

   Register custom polling callback that polls from shared queue.
   Override of parent method.

See Also
--------

- :doc:`dash_interactive_app_with_logs` - Parent class with log visualization
- :doc:`dash_interactive_app` - Base application class
- :doc:`../utilities/log_collector` - LogCollector for capturing execution logs
