====================================
API Reference
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

This document provides technical API specifications for all OpenManus
modules, intended for developers extending or integrating with the
framework.


Core Data Models (``app/schema.py``)
=====================================

Enums
-----

.. code-block:: python

    class Role(str, Enum):
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"
        TOOL = "tool"

    class ToolChoice(str, Enum):
        NONE = "none"       # LLM must not call tools
        AUTO = "auto"       # LLM decides whether to call tools
        REQUIRED = "required"  # LLM must call at least one tool

    class AgentState(str, Enum):
        IDLE = "IDLE"
        RUNNING = "RUNNING"
        FINISHED = "FINISHED"
        ERROR = "ERROR"

Message
-------

.. code-block:: python

    class Message(BaseModel):
        role: Role
        content: Optional[str] = None
        tool_calls: Optional[List[ToolCall]] = None
        name: Optional[str] = None
        tool_call_id: Optional[str] = None
        base64_image: Optional[str] = None

        # Factory methods
        @classmethod
        def user_message(cls, content: str) -> "Message"
        @classmethod
        def assistant_message(cls, content: str) -> "Message"
        @classmethod
        def system_message(cls, content: str) -> "Message"
        @classmethod
        def tool_message(
            cls, content: str, name,
            tool_call_id: str,
            base64_image: Optional[str] = None
        ) -> "Message"

        # Conversion
        def to_dict(self) -> dict  # OpenAI API format

ToolCall
--------

.. code-block:: python

    class ToolCall(BaseModel):
        id: str
        type: str = "function"
        function: Function

    class Function(BaseModel):
        name: str
        arguments: str  # JSON-encoded string

Memory
------

.. code-block:: python

    class Memory(BaseModel):
        messages: List[Message] = Field(default_factory=list)
        max_messages: int = Field(default=100)

        def add_message(self, message: Message) -> None
        def add_messages(self, messages: List[Message]) -> None
        def clear(self) -> None
        def get_recent_messages(self, n: int) -> List[Message]
        def to_dict_list(self) -> List[dict]


Agent API (``app/agent/``)
===========================

BaseAgent
---------

.. code-block:: python

    class BaseAgent(BaseModel, ABC):
        # Configuration
        name: str = "base"
        description: str = ""
        system_prompt: str = ""
        next_step_prompt: str = ""
        max_steps: int = 10
        duplicate_threshold: int = 2

        # State
        memory: Memory = Field(default_factory=Memory)
        state: AgentState = AgentState.IDLE
        current_step: int = 0
        llm: LLM = Field(default_factory=LLM)

        # Methods
        async def run(self, request: Optional[str] = None) -> str
        @abstractmethod
        async def step(self) -> str
        def update_memory(self, role: str, content: str) -> None
        def is_stuck(self) -> bool
        async def cleanup(self) -> None

ReActAgent
----------

.. code-block:: python

    class ReActAgent(BaseAgent, ABC):
        @abstractmethod
        async def think(self) -> bool
        @abstractmethod
        async def act(self) -> str
        async def step(self) -> str  # think() then act()

ToolCallAgent
-------------

.. code-block:: python

    class ToolCallAgent(ReActAgent):
        available_tools: ToolCollection
        tool_calls: List[ToolCall] = []
        tool_choices: ToolChoice = ToolChoice.AUTO
        special_tool_names: List[str] = ["terminate"]
        max_steps: int = 30
        max_observe: Optional[int] = None

        async def think(self) -> bool
        async def act(self) -> str
        async def execute_tool(self, command: ToolCall) -> str
        async def cleanup(self) -> None


Tool API (``app/tool/``)
=========================

BaseTool
--------

.. code-block:: python

    class BaseTool(ABC, BaseModel):
        name: str
        description: str
        parameters: dict = {}

        @abstractmethod
        async def execute(self, **kwargs) -> Any

        def to_param(self) -> dict     # OpenAI function format
        def __call__(self, **kwargs)   # Delegates to execute()
        def success_response(self, data: Any) -> ToolResult
        def fail_response(self, msg: str) -> ToolResult

ToolResult
----------

.. code-block:: python

    class ToolResult(BaseModel):
        output: Any = None
        error: Optional[str] = None
        base64_image: Optional[str] = None
        system: Optional[str] = None

        def __bool__(self) -> bool  # True if output or error
        def __add__(self, other: "ToolResult") -> "ToolResult"
        def __str__(self) -> str
        def replace(self, **kwargs) -> "ToolResult"

ToolCollection
--------------

.. code-block:: python

    class ToolCollection:
        tools: Tuple[BaseTool, ...] = ()
        tool_map: Dict[str, BaseTool] = {}

        def __init__(self, *tools: BaseTool)
        def to_params(self) -> List[dict]
        async def execute(
            self, name: str, tool_input: dict
        ) -> ToolResult
        def add_tool(self, tool: BaseTool) -> None
        def add_tools(self, *tools: BaseTool) -> None
        def get_tool(self, name: str) -> Optional[BaseTool]
        async def execute_all(self) -> List[ToolResult]


Flow API (``app/flow/``)
=========================

BaseFlow
--------

.. code-block:: python

    class BaseFlow(BaseModel, ABC):
        agents: Dict[str, BaseAgent]
        tools: Optional[List] = None
        primary_agent_key: Optional[str] = None

        @abstractmethod
        async def execute(self, input_text: str) -> str

        def get_agent(self, key: str) -> Optional[BaseAgent]
        @property
        def primary_agent(self) -> Optional[BaseAgent]

PlanningFlow
------------

.. code-block:: python

    class PlanningFlow(BaseFlow):
        planning_tool: PlanningTool = Field(
            default_factory=PlanningTool
        )
        active_plan_id: Optional[str] = None

        async def execute(self, input_text: str) -> str
        async def _create_initial_plan(self, request: str) -> str
        async def _execute_step(
            self, executor: BaseAgent, step_info: dict
        ) -> str
        def get_executor(self, step_type: str) -> BaseAgent

FlowFactory
-----------

.. code-block:: python

    class FlowType(str, Enum):
        PLANNING = "planning"

    class FlowFactory:
        @staticmethod
        def create_flow(
            flow_type: FlowType = FlowType.PLANNING,
            agents: Optional[dict] = None,
            **kwargs
        ) -> BaseFlow


LLM API (``app/llm.py``)
==========================

.. code-block:: python

    class LLM:
        _instances: Dict[str, "LLM"] = {}

        @classmethod
        def get_instance(cls, config_name: str = "default") -> "LLM"

        async def ask(
            self,
            messages: List[dict],
            system_msgs: Optional[List[dict]] = None,
            stream: bool = True,
            temperature: Optional[float] = None,
        ) -> str

        async def ask_with_images(
            self,
            messages: List[dict],
            images: List[str],
            system_msgs: Optional[List[dict]] = None,
        ) -> str

        async def ask_tool(
            self,
            messages: List[dict],
            system_msgs: Optional[List[dict]] = None,
            tools: Optional[List[dict]] = None,
            tool_choice: str = "auto",
            temperature: Optional[float] = None,
        ) -> ChatCompletionMessage

        def format_messages(
            self,
            messages: List[Message],
            system_msgs: Optional[List[dict]] = None,
        ) -> List[dict]


Configuration API (``app/config.py``)
=======================================

.. code-block:: python

    class Config:
        """Singleton configuration manager."""

        @property
        def llm(self) -> Dict[str, LLMSettings]

        @property
        def browser_config(self) -> Optional[BrowserSettings]

        @property
        def search_config(self) -> Optional[SearchSettings]

        @property
        def sandbox(self) -> SandboxSettings

        @property
        def mcp_config(self) -> MCPSettings

        @property
        def run_flow_config(self) -> RunflowSettings

        @property
        def daytona(self) -> DaytonaSettings

        @property
        def workspace_root(self) -> Path

        @property
        def root_path(self) -> Path

    class LLMSettings(BaseModel):
        model: str
        base_url: str
        api_key: str
        max_tokens: int = 4096
        max_input_tokens: Optional[int] = None
        temperature: float = 1.0
        api_type: str
        api_version: str

    class BrowserSettings(BaseModel):
        headless: bool
        disable_security: bool
        extra_chromium_args: List[str]
        proxy: Optional[ProxySettings]

    class SearchSettings(BaseModel):
        engine: str
        fallback_engines: List[str]
        retry_delay: int
        max_retries: int
        lang: str
        country: str

    class SandboxSettings(BaseModel):
        use_sandbox: bool
        image: str
        work_dir: str
        memory_limit: str
        cpu_limit: float
        timeout: int
        network_enabled: bool

    class MCPSettings(BaseModel):
        server_reference: str
        servers: Dict[str, MCPServerConfig]

    class DaytonaSettings(BaseModel):
        daytona_api_key: str
        daytona_server_url: Optional[str]
        daytona_target: Optional[str]
        sandbox_image_name: Optional[str]
        sandbox_entrypoint: Optional[str]
        VNC_password: Optional[str]

    class RunflowSettings(BaseModel):
        use_data_analysis_agent: bool


Exception Classes (``app/exceptions.py``)
==========================================

.. code-block:: python

    class ToolError(Exception):
        """Raised when a tool encounters an error."""

    class OpenManusError(Exception):
        """Base exception for OpenManus."""

    class TokenLimitExceeded(OpenManusError):
        """Raised when token limit is exceeded."""


MCP API
=======

Client (``app/tool/mcp.py``)
-----------------------------

.. code-block:: python

    class MCPClientTool(BaseTool):
        session: Optional[ClientSession]
        server_id: str
        original_name: str

        async def execute(self, **kwargs) -> ToolResult

    class MCPClients(ToolCollection):
        sessions: Dict[str, ClientSession]
        exit_stacks: Dict[str, AsyncExitStack]

        async def connect_sse(
            self, server_url: str, server_id: str = ""
        ) -> None
        async def connect_stdio(
            self, command: str, args: List[str],
            server_id: str = ""
        ) -> None
        async def disconnect(self, server_id: str = "") -> None
        async def list_tools(self) -> ListToolsResult

Server (``app/mcp/server.py``)
-------------------------------

.. code-block:: python

    class MCPServer:
        server: FastMCP

        def __init__(self, name: str = "openmanus")
        def register_tool(self, tool: BaseTool) -> None
        def run(self, transport: str = "stdio") -> None

    def parse_args() -> argparse.Namespace
