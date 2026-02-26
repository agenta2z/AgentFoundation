====================================
MCP Integration
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

OpenManus has **both client and server** implementations for the
`Model Context Protocol (MCP) <https://modelcontextprotocol.io/>`_.
MCP enables standardized communication between LLM applications and
external tool providers.

- **MCP Client**: Connects to external MCP servers to discover and
  use their tools dynamically
- **MCP Server**: Exposes OpenManus tools (Bash, Browser, Editor) as
  an MCP-compatible server for other applications to use


MCP Client
==========

**File**: ``app/tool/mcp.py``

The MCP client system consists of two classes: ``MCPClients`` (connection
manager) and ``MCPClientTool`` (tool proxy).

MCPClients
----------

``MCPClients`` extends ``ToolCollection`` to manage connections to
multiple MCP servers simultaneously:

.. code-block:: python

    class MCPClients(ToolCollection):
        sessions: Dict[str, ClientSession] = {}
        exit_stacks: Dict[str, AsyncExitStack] = {}

Connection Methods
^^^^^^^^^^^^^^^^^^

**SSE (Server-Sent Events)**:

.. code-block:: python

    await mcp_clients.connect_sse(
        server_url="http://localhost:8080/sse",
        server_id="my-server"
    )

**Stdio (Standard I/O)**:

.. code-block:: python

    await mcp_clients.connect_stdio(
        command="python",
        args=["my_mcp_server.py"],
        server_id="local-server"
    )

Tool Discovery
^^^^^^^^^^^^^^

After connecting, tools are automatically discovered:

::

    connect_sse/connect_stdio
         │
         ▼
    session.initialize()
         │
         ▼
    session.list_tools()
         │
         ▼
    For each remote tool:
         │
         ├── Create MCPClientTool proxy
         │     name = "mcp_{server_id}_{original_name}"
         │     description = tool.description
         │     parameters = tool.inputSchema
         │     session = active_session
         │
         └── Register in tool_map

Tool Name Sanitization
^^^^^^^^^^^^^^^^^^^^^^

Tool names are sanitized for compatibility:

.. code-block:: python

    def _sanitize_tool_name(self, name: str) -> str:
        # Replace invalid characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Truncate to 64 characters
        if len(sanitized) > 64:
            sanitized = sanitized[:64]
        return sanitized

Disconnection
^^^^^^^^^^^^^

Clean disconnection with resource cleanup:

.. code-block:: python

    # Disconnect specific server
    await mcp_clients.disconnect("my-server")

    # Disconnect all servers
    await mcp_clients.disconnect()

MCPClientTool
-------------

Proxy tool that forwards execution to remote MCP servers:

.. code-block:: python

    class MCPClientTool(BaseTool):
        session: Optional[ClientSession] = None
        server_id: str = ""
        original_name: str = ""

        async def execute(self, **kwargs) -> ToolResult:
            if not self.session:
                return ToolResult(error="Not connected to MCP server")
            result = await self.session.call_tool(
                self.original_name, kwargs
            )
            content_str = ", ".join(
                item.text for item in result.content
                if isinstance(item, TextContent)
            )
            return ToolResult(output=content_str or "No output returned.")


MCP Server
==========

**File**: ``app/mcp/server.py``

The MCP server exposes OpenManus tools for external consumption via
the ``FastMCP`` library.

.. code-block:: python

    class MCPServer:
        def __init__(self, name: str = "openmanus"):
            self.server = FastMCP(name)
            self.tools = [
                Bash(),
                BrowserUseTool(),
                StrReplaceEditor(),
                Terminate(),
            ]
            for tool in self.tools:
                self.register_tool(tool)

        def run(self, transport: str = "stdio"):
            self.server.run(transport=transport)

Tool Registration
-----------------

The server dynamically creates async wrapper functions for each tool:

.. code-block:: python

    def register_tool(self, tool: BaseTool):
        """Register a BaseTool as an MCP tool."""
        # Build function signature from tool parameters
        params = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])

        # Create async wrapper with proper type annotations
        async def tool_wrapper(**kwargs):
            result = await tool.execute(**kwargs)
            return str(result)

        # Register with FastMCP
        self.server.tool(
            name=tool.name,
            description=tool.description,
        )(tool_wrapper)

Running the Server
------------------

.. code-block:: bash

    python run_mcp_server.py

    # Or with explicit transport
    python run_mcp_server.py --transport stdio

The server supports stdio transport (default) for local integration.


MCP Agent
=========

**File**: ``app/agent/mcp.py``

The ``MCPAgent`` is a specialized ``ToolCallAgent`` that works
exclusively with MCP server tools.

Key Features
------------

- **Dynamic Tool Loading**: Tools are loaded from MCP servers at init
- **Periodic Refresh**: Tools are refreshed every N steps to detect
  schema changes:

  .. code-block:: python

      _refresh_tools_interval: int = 5

      async def think(self):
          if self.current_step % self._refresh_tools_interval == 0:
              await self._refresh_tools()
          return await super().think()

- **Multimedia Handling**: Processes base64 image responses from MCP tools
- **Graceful Shutdown**: Detects when MCP service becomes unavailable
  and terminates cleanly

Error Handling
--------------

The MCP agent includes specialized error prompts:

.. code-block:: python

    # From app/prompt/mcp.py
    TOOL_ERROR_PROMPT = (
        "The tool execution failed. "
        "Please check the error message and try again "
        "with corrected parameters."
    )

    MULTIMEDIA_RESPONSE_PROMPT = (
        "The tool returned multimedia content. "
        "Please describe what you see."
    )


MCP in Manus Agent
==================

The ``Manus`` agent integrates MCP alongside its local tools:

.. code-block:: python

    # app/agent/manus.py
    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        instance = cls(**kwargs)

        # Load MCP server configs
        mcp_config = config.mcp
        if mcp_config and mcp_config.servers:
            for server_config in mcp_config.servers:
                if server_config.type == "sse":
                    await instance.mcp_clients.connect_sse(
                        server_url=server_config.url,
                        server_id=server_config.id,
                    )
                elif server_config.type == "stdio":
                    await instance.mcp_clients.connect_stdio(
                        command=server_config.command,
                        args=server_config.args,
                        server_id=server_config.id,
                    )
            # Add MCP tools to available tools
            for tool in instance.mcp_clients.tool_map.values():
                instance.available_tools.add_tool(tool)

        return instance

This means the Manus agent can use both:

- **Local tools**: PythonExecute, BrowserUseTool, StrReplaceEditor, etc.
- **Remote MCP tools**: Any tools provided by connected MCP servers


Configuration
=============

MCP servers are configured in ``config/mcp.json``:

.. code-block:: json

    {
        "mcpServers": {
            "filesystem": {
                "url": "http://localhost:3000/sse",
                "type": "sse"
            },
            "database": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-sqlite",
                         "path/to/db.sqlite"],
                "type": "stdio"
            }
        }
    }

The path to this file is specified in the main config:

.. code-block:: toml

    [mcp]
    config_path = "config/mcp.json"


Running with MCP
================

**As a Client** (connecting to external MCP servers):

.. code-block:: bash

    # Interactive mode
    python run_mcp.py --connection sse --interactive

    # With a specific prompt
    python run_mcp.py --connection stdio --prompt "List files"

**As a Server** (exposing OpenManus tools via MCP):

.. code-block:: bash

    python run_mcp_server.py

Other MCP clients (e.g., Claude Desktop, Continue.dev) can then
connect to the OpenManus MCP server and use its tools.
