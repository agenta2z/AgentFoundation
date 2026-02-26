====================================
Getting Started
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Prerequisites
=============

- **Python**: 3.11, 3.12, or 3.13 (3.12 recommended)
- **Git**: For cloning the repository
- **LLM API Key**: OpenAI, Anthropic, Azure, or compatible provider
- **Docker** (optional): For sandboxed code execution
- **Node.js** (optional): For data visualization features


Installation
============

Method 1: Using conda
----------------------

.. code-block:: bash

    # Create a new conda environment
    conda create -n open_manus python=3.12
    conda activate open_manus

    # Clone the repository
    git clone https://github.com/FoundationAgents/OpenManus.git
    cd OpenManus

    # Install dependencies
    pip install -r requirements.txt

Method 2: Using uv (Recommended)
----------------------------------

.. code-block:: bash

    # Install uv (fast Python package manager)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Clone the repository
    git clone https://github.com/FoundationAgents/OpenManus.git
    cd OpenManus

    # Create and activate virtual environment
    uv venv --python 3.12
    source .venv/bin/activate  # Unix/macOS
    # Or on Windows: .venv\Scripts\activate

    # Install dependencies
    uv pip install -r requirements.txt

Browser Automation (Optional)
------------------------------

If you plan to use browser automation tools:

.. code-block:: bash

    playwright install


Configuration
=============

Step 1: Create Configuration File
-----------------------------------

.. code-block:: bash

    cp config/config.example.toml config/config.toml

Step 2: Add Your API Key
--------------------------

Edit ``config/config.toml``:

.. code-block:: toml

    [llm]
    model = "gpt-4o"
    base_url = "https://api.openai.com/v1"
    api_key = "sk-your-api-key-here"
    max_tokens = 4096
    temperature = 0.0

Step 3: Configure Optional Features
-------------------------------------

**Vision model** (for multimodal tasks):

.. code-block:: toml

    [llm.vision]
    model = "gpt-4o"
    base_url = "https://api.openai.com/v1"
    api_key = "sk-your-api-key-here"

**Browser settings**:

.. code-block:: toml

    [browser]
    headless = true
    disable_security = true

**Search engine**:

.. code-block:: toml

    [search]
    engine = "Google"


Quick Start
===========

Running the Single Agent
-------------------------

The simplest way to use OpenManus:

.. code-block:: bash

    python main.py

This will:

1. Create a ``Manus`` agent with default tools
2. Prompt you for input via the terminal
3. Execute the task using the think-act loop
4. Output results to the terminal and ``workspace/`` directory

You can also pass a prompt directly:

.. code-block:: bash

    python main.py --prompt "Create a Python script that calculates fibonacci numbers"

Running the Multi-Agent Planning Flow
--------------------------------------

For complex tasks that benefit from structured planning:

.. code-block:: bash

    python run_flow.py

This creates a ``PlanningFlow`` with a ``Manus`` agent that:

1. Decomposes your task into a structured plan
2. Executes each step sequentially
3. Tracks progress and adjusts the plan as needed

.. note::

   The multi-agent flow is marked as "unstable" in the README. For
   reliable operation, use the single-agent mode (``main.py``).

Running with MCP
-----------------

To use tools from external MCP servers:

.. code-block:: bash

    # First, configure MCP servers in config/mcp.json
    cp config/mcp.example.json config/mcp.json

    # Run the MCP agent
    python run_mcp.py --connection sse --interactive

Running in Sandbox Mode
------------------------

For isolated execution in a Daytona cloud sandbox:

.. code-block:: bash

    python sandbox_main.py

Requires Daytona configuration:

.. code-block:: toml

    [daytona]
    api_key = "your-daytona-api-key"


Usage Examples
==============

Example 1: Web Research
-----------------------

.. code-block:: text

    Enter your prompt: Research the latest developments in quantum
    computing and create a summary document.

The agent will:

1. Use ``web_search`` to find recent articles
2. Use ``browser_use`` to visit and extract content
3. Use ``str_replace_editor`` to create a summary file in ``workspace/``
4. Call ``terminate`` when done

Example 2: Code Generation
---------------------------

.. code-block:: text

    Enter your prompt: Write a Python web scraper that extracts
    product prices from an e-commerce site and saves them to CSV.

The agent will:

1. Use ``python_execute`` to prototype the scraper
2. Use ``str_replace_editor`` to create the final script
3. Test the code with ``python_execute``
4. Save the output CSV to ``workspace/``

Example 3: Data Analysis
--------------------------

First, enable the data analysis agent in ``config.toml``:

.. code-block:: toml

    [runflow]
    use_data_analysis_agent = true

Then:

.. code-block:: bash

    python run_flow.py

.. code-block:: text

    Enter your prompt: Analyze the dataset at workspace/sales.csv
    and create visualizations showing monthly trends.


Project Structure for New Users
=================================

Key files to understand first:

1. **``main.py``** -- Entry point; shows how agents are created and run
2. **``app/agent/manus.py``** -- The main agent implementation
3. **``app/agent/toolcall.py``** -- The think-act loop
4. **``app/tool/base.py``** -- How tools are defined
5. **``app/config.py``** -- Configuration system
6. **``app/schema.py``** -- Core data models (Message, Memory, etc.)


Troubleshooting
===============

Common Issues
-------------

**"Python version not supported"**

OpenManus requires Python 3.11-3.13. Check your version:

.. code-block:: bash

    python --version

**"API key not configured"**

Ensure ``config/config.toml`` exists and contains a valid API key:

.. code-block:: bash

    ls config/config.toml
    grep "api_key" config/config.toml

**"Browser not installed"**

For browser automation, install Playwright browsers:

.. code-block:: bash

    playwright install

**"Docker not available"**

For sandbox mode, ensure Docker is running:

.. code-block:: bash

    docker info

**"MCP connection failed"**

Check that MCP servers are running and the configuration in
``config/mcp.json`` is correct.


Next Steps
==========

- Read :doc:`architecture` to understand the system design
- Read :doc:`agent_system` to learn about agent types and lifecycle
- Read :doc:`tool_system` to explore available tools
- Read :doc:`tutorials` for step-by-step guides
- Read :doc:`knowledge_and_skills` to understand the system's
  capabilities and limitations
