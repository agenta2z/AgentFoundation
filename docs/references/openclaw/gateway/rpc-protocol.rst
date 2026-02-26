.. _gateway-rpc-protocol:

============
RPC Protocol
============

The gateway exposes a JSON-RPC-style protocol for all inter-component
communication. Both HTTP and WebSocket transports are supported.

Protocol Format
===============

Requests follow a JSON-RPC pattern:

.. code-block:: json

   {
     "method": "agent",
     "params": {
       "sessionKey": "main",
       "message": "Hello, what can you do?"
     }
   }

Responses include results or errors:

.. code-block:: json

   {
     "result": {
       "reply": "I can help with many tasks...",
       "usage": { "inputTokens": 150, "outputTokens": 200 }
     }
   }

Core Methods
============

Agent Methods
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - ``agent``
     - Send a message to an agent session and get a reply
   * - ``agent.wait``
     - Poll for completion of an active agent run
   * - ``agent.steer``
     - Inject a message into an actively running agent to redirect it
   * - ``agent.abort``
     - Cancel an active agent run

Session Methods
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - ``sessions.list``
     - List all session entries
   * - ``sessions.get``
     - Get a specific session by key
   * - ``sessions.delete``
     - Delete a session and its transcript
   * - ``sessions.patch``
     - Update session properties (model, thinking level, etc.)

Chat Methods
------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - ``chat.send``
     - Send a message to a channel destination
   * - ``chat.history``
     - Retrieve conversation history

System Methods
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - ``health``
     - Health check endpoint
   * - ``status``
     - Channel connection status
   * - ``config.reload``
     - Hot-reload configuration from disk

Error Codes
===========

The protocol defines structured error codes for common failure modes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Code
     - Description
   * - ``AUTH_FAILED``
     - Invalid or missing authentication token
   * - ``SESSION_NOT_FOUND``
     - Requested session does not exist
   * - ``AGENT_BUSY``
     - Agent is already processing a request for this session
   * - ``MODEL_ERROR``
     - LLM provider returned an error
   * - ``RATE_LIMITED``
     - API key rate limit hit

Schema definitions: ``src/gateway/protocol/schema/``

Source: ``src/gateway/protocol/index.ts``, ``src/gateway/protocol/schema.ts``
