====================================
Sandbox System
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

OpenManus provides **two independent sandboxing systems** for isolated
code execution:

1. **Docker Sandbox (Local)**: Docker container-based isolation for
   local development and testing
2. **Daytona Sandbox (Cloud)**: Cloud-based sandboxed environments
   with VNC, browser, and desktop automation support


Docker Sandbox
==============

Architecture
------------

::

    ┌──────────────────────────────────────────────┐
    │                Host System                    │
    │                                              │
    │  LocalSandboxClient (SANDBOX_CLIENT singleton)│
    │       │                                       │
    │       ▼                                       │
    │  DockerSandbox                                │
    │       │                                       │
    │       ├── Docker Container (python:3.12-slim) │
    │       │     ├── /workspace (working dir)      │
    │       │     └── Terminal session               │
    │       │                                       │
    │       ├── Volume bindings (host ↔ container)  │
    │       └── Resource limits (CPU, memory)       │
    │                                              │
    └──────────────────────────────────────────────┘

Components
----------

BaseSandboxClient
^^^^^^^^^^^^^^^^^

**File**: ``app/sandbox/client.py``

Abstract interface for sandbox operations:

.. code-block:: python

    class BaseSandboxClient(ABC):
        async def create(self, config, volume_bindings) -> None: ...
        async def run_command(self, command, timeout) -> str: ...
        async def copy_from(self, container_path, local_path) -> None: ...
        async def copy_to(self, local_path, container_path) -> None: ...
        async def read_file(self, path) -> str: ...
        async def write_file(self, path, content) -> None: ...
        async def cleanup(self) -> None: ...

LocalSandboxClient
^^^^^^^^^^^^^^^^^^^

Concrete implementation using ``DockerSandbox``:

.. code-block:: python

    class LocalSandboxClient(BaseSandboxClient):
        def __init__(self):
            self.sandbox: Optional[DockerSandbox] = None

        async def create(self, config=None, volume_bindings=None):
            self.sandbox = DockerSandbox(config, volume_bindings)
            await self.sandbox.create()

Global singleton:

.. code-block:: python

    SANDBOX_CLIENT = create_sandbox_client()  # Module-level singleton

DockerSandbox
^^^^^^^^^^^^^

**File**: ``app/sandbox/core/sandbox.py``

Manages Docker container lifecycle:

.. code-block:: python

    class DockerSandbox:
        def __init__(self, config=None, volume_bindings=None):
            self.config = config or SandboxSettings()
            self.container = None
            self.terminal = None

        async def create(self):
            """Create and start a Docker container."""
            self.container = docker_client.containers.run(
                image=self.config.image,
                working_dir=self.config.work_dir,
                mem_limit=self.config.memory_limit,
                cpu_quota=int(self.config.cpu_limit * 100000),
                network_mode="none" if not self.config.network_enabled
                    else "bridge",
                detach=True,
                tty=True,
            )

Safety Mechanisms
-----------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Mechanism
     - Description
   * - Resource Limits
     - Memory limit and CPU quota via Docker cgroups
   * - Network Isolation
     - ``network_mode="none"`` when ``network_enabled=False``
   * - Path Traversal Prevention
     - ``_safe_resolve_path()`` rejects ``..`` in paths
   * - Timeout Enforcement
     - ``SandboxTimeoutError`` for long-running commands
   * - Isolated Working Directory
     - Temp dir on host mapped to container's ``/workspace``

Path Safety
^^^^^^^^^^^

.. code-block:: python

    def _safe_resolve_path(self, path: str) -> str:
        """Prevent path traversal attacks."""
        if ".." in path.split("/"):
            raise ValueError("Path contains potentially unsafe patterns")
        resolved = (
            os.path.join(self.config.work_dir, path)
            if not os.path.isabs(path)
            else path
        )
        return resolved

File Operations
---------------

The sandbox supports full file operations:

- **read_file**: Read file content from container
- **write_file**: Write content to container file
- **copy_from**: Copy file from container to host
- **copy_to**: Copy file from host to container

All file operations use tar archives for data transfer with the
Docker API.

Integration with Tools
----------------------

The ``StrReplaceEditor`` uses the sandbox through ``SandboxFileOperator``:

.. code-block:: python

    class SandboxFileOperator:
        """File operations through Docker sandbox."""
        async def read_file(self, path: str) -> str:
            return await SANDBOX_CLIENT.read_file(path)

        async def write_file(self, path: str, content: str):
            await SANDBOX_CLIENT.write_file(path, content)

Selection is based on configuration:

.. code-block:: python

    if config.sandbox.use_sandbox:
        file_operator = SandboxFileOperator()
    else:
        file_operator = LocalFileOperator()


Daytona Sandbox
===============

Architecture
------------

::

    ┌──────────────────────────────────────────────────────┐
    │              Daytona Cloud Platform                    │
    │                                                       │
    │  ┌───────────────────────────────────────────────┐    │
    │  │           Sandbox Instance                     │    │
    │  │                                               │    │
    │  │  ┌─────────────────────────────────────────┐  │    │
    │  │  │  /workspace                             │  │    │
    │  │  │    Agent working directory               │  │    │
    │  │  └─────────────────────────────────────────┘  │    │
    │  │                                               │    │
    │  │  ┌──────────┐  ┌──────────┐  ┌────────────┐  │    │
    │  │  │ Browser  │  │  Shell   │  │ Automation │  │    │
    │  │  │ :8003    │  │  (tmux)  │  │   :8000    │  │    │
    │  │  └──────────┘  └──────────┘  └────────────┘  │    │
    │  │                                               │    │
    │  │  ┌──────────────────────────────────────────┐ │    │
    │  │  │  VNC / Desktop Environment               │ │    │
    │  │  └──────────────────────────────────────────┘ │    │
    │  └───────────────────────────────────────────────┘    │
    │                                                       │
    └──────────────────────────────────────────────────────┘

Components
----------

Sandbox Management
^^^^^^^^^^^^^^^^^^

**File**: ``app/daytona/sandbox.py``

Creates and manages Daytona cloud sandboxes:

.. code-block:: python

    # Configuration
    resources = {
        "cpu": 2,
        "memory": 4,      # GB
        "disk": 5,         # GB
    }
    auto_stop_interval = 15   # minutes
    auto_archive_interval = 24 * 60  # minutes (24 hours)

SandboxToolsBase
^^^^^^^^^^^^^^^^

**File**: ``app/daytona/tool_base.py``

Base class for all Daytona sandbox tools. Uses lazy sandbox creation
with automatic state management:

.. code-block:: python

    class SandboxToolsBase(BaseTool):
        """Base class for all sandbox tools that provides
        project-based sandbox access."""

        _sandbox: Optional[Sandbox] = None
        workspace_path: str = "/workspace"

        async def _ensure_sandbox(self) -> Sandbox:
            """Ensure we have a valid sandbox instance,
            creating one if needed."""
            if self._sandbox is None:
                self._sandbox = create_sandbox(
                    password=config.daytona.VNC_password
                )
            elif self._sandbox.state in (
                SandboxState.ARCHIVED, SandboxState.STOPPED
            ):
                daytona.start(self._sandbox)
                start_supervisord_session(self._sandbox)
            return self._sandbox

Sandbox Tools
-------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Tool
     - Capabilities
   * - **SandboxBrowserTool**
     - Browser automation via HTTP API on port 8003. 15 actions:
       ``navigate_to``, ``click_element``, ``input_text``, ``scroll``,
       ``drag_drop``, ``click_coordinates``, etc. Screenshot validation
       (base64 format check).
   * - **SandboxFilesTool**
     - File operations in ``/workspace``: ``create_file``,
       ``str_replace``, ``full_file_rewrite``, ``delete_file``.
       Auto-detects ``index.html`` and provides HTTP server URL.
   * - **SandboxShellTool**
     - Shell execution via tmux sessions: ``execute_command``,
       ``check_command_output``, ``terminate_command``,
       ``list_commands``. Supports blocking/non-blocking with timeout.
   * - **SandboxVisionTool**
     - Image reading from sandbox with compression (max 1920x1080,
       quality 85). Action: ``see_image``. Returns base64 data.

SandboxManus Agent
------------------

**File**: ``app/agent/sandbox_agent.py``

.. code-block:: python

    class SandboxManus(ToolCallAgent):
        """Cloud sandbox variant of Manus."""

        # Starts with only AskHuman + Terminate;
        # sandbox tools are added during initialization
        available_tools: ToolCollection = Field(
            default_factory=lambda: ToolCollection(
                AskHuman(),
                Terminate(),
            )
        )

        @classmethod
        async def create(cls, **kwargs):
            instance = cls(**kwargs)
            await instance.initialize_mcp_servers()
            await instance.initialize_sandbox_tools()
            return instance

        async def initialize_sandbox_tools(self, password=...):
            sandbox = create_sandbox(password=password)
            sb_tools = [
                SandboxBrowserTool(sandbox),
                SandboxFilesTool(sandbox),
                SandboxShellTool(sandbox),
                SandboxVisionTool(sandbox),
            ]
            self.available_tools.add_tools(*sb_tools)

Sandbox tools are passed the ``Sandbox`` instance at construction
time (not via a ``setup()`` method), and are added to the agent's
``ToolCollection`` dynamically after sandbox creation.

Daytona Configuration
---------------------

All Daytona settings are defined in ``config/config.toml``:

.. code-block:: toml

    [daytona]
    daytona_api_key = "your-api-key"
    daytona_server_url = "https://app.daytona.io/api"
    daytona_target = "us"              # "eu" or "us"
    sandbox_image_name = "whitezxj/sandbox:0.1.0"
    VNC_password = "123456"

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Setting
     - Description
   * - ``daytona_api_key``
     - API key for Daytona cloud platform (required)
   * - ``daytona_server_url``
     - Server URL (default: ``https://app.daytona.io/api``)
   * - ``daytona_target``
     - Target region: ``"eu"`` or ``"us"`` (default: ``"us"``)
   * - ``sandbox_image_name``
     - Docker image for sandbox (default: ``whitezxj/sandbox:0.1.0``)
   * - ``VNC_password``
     - Password for VNC access to sandbox desktop (default: ``"123456"``)


Python Execution Safety
=======================

**File**: ``app/tool/python_execute.py``

The ``PythonExecute`` tool provides a lighter-weight isolation than
full containerization:

.. code-block:: python

    class PythonExecute(BaseTool):
        async def execute(self, code: str, timeout: int = 5) -> str:
            # Run in separate process
            process = multiprocessing.Process(
                target=self._run_code,
                args=(code, result_queue)
            )
            process.start()
            process.join(timeout=timeout)

            if process.is_alive():
                process.terminate()
                return "Execution timed out"

.. warning::

   ``PythonExecute`` uses ``exec()`` with full ``__builtins__``. This
   is **not a security boundary** -- it prevents only accidental
   infinite loops (via timeout), not malicious code execution. For
   production use, the Docker or Daytona sandboxes should be used
   instead.


Comparison
==========

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Feature
     - Docker Sandbox
     - Daytona Sandbox
   * - Isolation Level
     - Container-level
     - Cloud VM-level
   * - Setup
     - Local Docker required
     - API key + cloud account
   * - Resource Control
     - Memory + CPU limits
     - CPU + Memory + Disk
   * - Network
     - Optional isolation
     - Cloud networking
   * - Browser
     - Not built-in
     - Built-in on port 8003
   * - Desktop/VNC
     - No
     - Yes
   * - File Operations
     - Via Docker API (tar)
     - Via Daytona SDK
   * - Auto-cleanup
     - Manual
     - Auto-stop (15min) + auto-archive (24h)
   * - Use Case
     - Development, testing
     - Production, complex tasks
