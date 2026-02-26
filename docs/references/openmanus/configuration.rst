====================================
Configuration
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

OpenManus uses **TOML-based configuration** with a singleton ``Config``
class. Configuration is loaded from ``config/config.toml`` at startup
and validated through Pydantic models.


Configuration System
====================

**File**: ``app/config.py``

The ``Config`` class follows the singleton pattern:

.. code-block:: python

    class Config:
        _instance = None
        _config: dict = {}

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._load_config()
            return cls._instance

        def _load_config(self):
            config_path = PROJECT_ROOT / "config" / "config.toml"
            if config_path.exists():
                with open(config_path, "rb") as f:
                    self._config = tomli.load(f)


Configuration File
==================

Copy ``config/config.example.toml`` to ``config/config.toml`` and edit:

.. code-block:: bash

    cp config/config.example.toml config/config.toml


LLM Settings
-------------

.. code-block:: toml

    # Global LLM configuration (required)
    [llm]
    model = "gpt-4o"
    base_url = "https://api.openai.com/v1"
    api_key = "sk-..."
    max_tokens = 4096
    temperature = 0.0
    timeout = 60
    max_retries = 3

    # Optional: Vision model (for multimodal tasks)
    [llm.vision]
    model = "gpt-4o"
    base_url = "https://api.openai.com/v1"
    api_key = "sk-..."

Pydantic model:

.. code-block:: python

    class LLMSettings(BaseModel):
        model: str = "gpt-4o"
        base_url: str = "https://api.openai.com/v1"
        api_key: str = ""
        max_tokens: int = 4096
        max_input_tokens: Optional[int] = None  # Total input token budget
        temperature: float = 1.0
        api_type: str = ""       # "azure", "aws", or "" (default/openai)
        api_version: str = ""    # Required for Azure

Browser Settings
----------------

.. code-block:: toml

    [browser]
    headless = true
    disable_security = true
    extra_chromium_args = []

    [browser.proxy]
    server = ""
    username = ""
    password = ""

Pydantic model:

.. code-block:: python

    class BrowserSettings(BaseModel):
        headless: bool = True
        disable_security: bool = True
        extra_chromium_args: List[str] = []
        proxy: Optional[ProxySettings] = None

Search Settings
---------------

.. code-block:: toml

    [search]
    engine = "Google"            # Google, Baidu, DuckDuckGo, Bing
    fallback_engines = ["DuckDuckGo", "Baidu"]
    retry_delay = 60
    max_retries = 3
    lang = "en"
    country = ""

Pydantic model:

.. code-block:: python

    class SearchSettings(BaseModel):
        engine: str = "Google"
        fallback_engines: List[str] = Field(
            default_factory=lambda: ["DuckDuckGo", "Baidu"]
        )
        retry_delay: int = 60
        max_retries: int = 3
        lang: str = "en"
        country: str = ""

Sandbox Settings
----------------

.. code-block:: toml

    [sandbox]
    use_sandbox = false
    image = "python:3.12-slim"
    work_dir = "/workspace"
    memory_limit = "512m"
    cpu_limit = 0.5
    timeout = 300
    network_enabled = true

MCP Settings
------------

.. code-block:: toml

    [mcp]
    # MCP server connections are defined in a separate JSON file
    config_path = "config/mcp.json"

MCP server configuration (``config/mcp.json``):

.. code-block:: json

    {
        "mcpServers": {
            "my-server": {
                "url": "http://localhost:8080/sse",
                "type": "sse"
            },
            "local-server": {
                "command": "python",
                "args": ["my_mcp_server.py"],
                "type": "stdio"
            }
        }
    }

RunFlow Settings
----------------

.. code-block:: toml

    [runflow]
    use_data_analysis_agent = false

Daytona Settings
----------------

.. code-block:: toml

    [daytona]
    daytona_api_key = ""
    daytona_server_url = "https://app.daytona.io/api"
    daytona_target = "us"                         # "eu" or "us"
    sandbox_image_name = "whitezxj/sandbox:0.1.0"
    VNC_password = "123456"

Pydantic model:

.. code-block:: python

    class DaytonaSettings(BaseModel):
        daytona_api_key: str
        daytona_server_url: Optional[str] = "https://app.daytona.io/api"
        daytona_target: Optional[str] = "us"   # "eu" or "us"
        sandbox_image_name: Optional[str] = "whitezxj/sandbox:0.1.0"
        sandbox_entrypoint: Optional[str] = (
            "/usr/bin/supervisord -n -c "
            "/etc/supervisor/conf.d/supervisord.conf"
        )
        VNC_password: Optional[str] = "123456"


Provider-Specific Configurations
=================================

OpenManus provides example configs for various LLM providers:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Config File
     - Provider
   * - ``config.example.toml``
     - OpenAI (default)
   * - ``config.example-model-anthropic.toml``
     - Anthropic Claude
   * - ``config.example-model-azure.toml``
     - Azure OpenAI
   * - ``config.example-model-google.toml``
     - Google models
   * - ``config.example-model-ollama.toml``
     - Ollama (local models)
   * - ``config.example-model-jiekouai.toml``
     - Jiekou.AI
   * - ``config.example-model-ppio.toml``
     - PPIO
   * - ``config.example-daytona.toml``
     - Daytona sandbox configuration


Global Constants
================

Defined in ``app/config.py``:

.. code-block:: python

    # Project root directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Workspace directory for agent outputs
    WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


Environment Variables
=====================

Some settings can be overridden via environment variables:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Purpose
   * - ``ENV_MODE``
     - Logger mode: ``"LOCAL"`` (console) or other (JSON)
   * - API provider keys
     - Can be set via ``python-dotenv`` (.env file)

The ``python-dotenv`` library is used to load ``.env`` files
automatically.
