====================================
Orchestration Flow
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

OpenManus supports two orchestration patterns:

1. **Single-Agent Mode**: A single ``Manus`` agent handles the entire
   task through its think-act loop (``main.py``)
2. **Multi-Agent Planning Mode**: A ``PlanningFlow`` decomposes tasks
   into steps and delegates each step to specialized agents
   (``run_flow.py``)

This document focuses on the multi-agent orchestration flow, which is
the more architecturally interesting pattern.


Single-Agent Execution
======================

In single-agent mode, the flow is straightforward:

::

    User Input
         │
         ▼
    Manus.run(prompt)
         │
         ▼
    ┌── Step 1 ──────────────────────┐
    │  think(): LLM decides actions  │
    │  act(): Execute tool calls     │
    └────────────┬───────────────────┘
                 │
    ┌── Step 2 ──▼───────────────────┐
    │  think(): LLM reasons on       │
    │           previous results     │
    │  act(): Execute next tools     │
    └────────────┬───────────────────┘
                 │
                ...
                 │
    ┌── Step N ──▼───────────────────┐
    │  think(): LLM calls terminate  │
    │  act(): Terminate tool →       │
    │         state = FINISHED       │
    └────────────────────────────────┘

The agent autonomously decides:

- Which tools to call and with what arguments
- When to terminate (by calling the ``terminate`` tool)
- How to handle errors and retry


Multi-Agent Planning Flow
=========================

**Entry Point**: ``run_flow.py``

**Core Class**: ``PlanningFlow`` (``app/flow/planning.py``)

The ``PlanningFlow`` implements a **Plan-and-Execute** orchestration
pattern where:

1. An LLM creates a structured plan with numbered steps
2. Each step is assigned to and executed by a specialized agent
3. Progress is tracked through a ``PlanningTool``
4. The plan can be dynamically updated based on execution results

Architecture
------------

::

    PlanningFlow
    ├── PlanningTool (plan CRUD operations)
    ├── LLM (for plan creation/updates)
    └── Agents Dictionary
        ├── "manus" → Manus (default/primary)
        └── "data_analysis" → DataAnalysis (optional)


Flow Factory
------------

**File**: ``app/flow/flow_factory.py``

The ``FlowFactory`` creates flow instances:

.. code-block:: python

    class FlowType(str, Enum):
        PLANNING = "planning"

    class FlowFactory:
        @staticmethod
        def create_flow(
            flow_type: FlowType = FlowType.PLANNING,
            agents: dict | None = None,
            **kwargs
        ) -> BaseFlow:
            flows = {FlowType.PLANNING: PlanningFlow}
            return flows[flow_type](agents=agents, **kwargs)


PlanningFlow Detailed Walkthrough
=================================

Initialization
--------------

.. code-block:: python

    # From run_flow.py
    agents = {"manus": Manus}

    # Optional: Add DataAnalysis agent if configured
    if config.runflow.use_data_analysis_agent:
        agents["data_analysis"] = DataAnalysis

    flow = FlowFactory.create_flow(
        flow_type=FlowType.PLANNING,
        agents=agents,
    )
    await flow.execute(prompt)

Step 1: Plan Creation
---------------------

When ``execute()`` is called, the flow first creates a plan:

1. The user's prompt is sent to the LLM along with the ``PlanningTool``
2. The LLM creates a plan by calling ``PlanningTool.execute(command="create", ...)``
3. The plan contains:
   - A title
   - Numbered steps (e.g., "Step 1: Research the topic")
   - Step statuses (all initially ``not_started``)

.. code-block:: python

    # PlanningTool plan structure (in-memory dict)
    plans = {
        "plan_001": {
            "title": "Research and summarize AI trends",
            "steps": [
                "Search for recent AI trends",
                "Extract key findings",
                "Write summary document"
            ],
            "step_statuses": [
                "not_started",  # Step 0
                "not_started",  # Step 1
                "not_started"   # Step 2
            ],
            "step_notes": [
                "",  # Step 0
                "",  # Step 1
                ""   # Step 2
            ]
        }
    }

Step 2: Step Execution Loop
----------------------------

The flow iterates through plan steps:

::

    ┌── For each step in plan ──────────────────────────────────┐
    │                                                           │
    │   1. Get current step text and status                     │
    │   2. Select executor agent:                               │
    │      - PlanningFlow.get_executor(step_type)               │
    │      - Falls back to primary agent (Manus)                │
    │                                                           │
    │   3. Build step prompt:                                   │
    │      - Include overall plan context                       │
    │      - Include specific step instructions                 │
    │      - Include results from previous steps                │
    │                                                           │
    │   4. Execute: agent.run(step_prompt)                      │
    │      - Agent runs its own think-act loop                  │
    │      - Agent uses its own tools autonomously              │
    │                                                           │
    │   5. Capture result                                       │
    │   6. Mark step as "completed" via PlanningTool             │
    │   7. Record step notes for context                        │
    │                                                           │
    │   Check: All steps done? → exit loop                      │
    │   Check: Max steps exceeded? → exit loop                  │
    │                                                           │
    └───────────────────────────────────────────────────────────┘

Step 3: Plan Updates
--------------------

After each step execution, the LLM may update the plan:

- Add new steps based on discovered information
- Modify remaining steps based on results
- Mark steps as ``blocked`` if prerequisites aren't met
- Add notes to steps for context


PlanningTool Details
====================

**File**: ``app/tool/planning.py``

The ``PlanningTool`` provides CRUD operations for plan management:

Commands
--------

.. list-table::
   :widths: 15 25 60
   :header-rows: 1

   * - Command
     - Parameters
     - Description
   * - ``create``
     - ``plan_id``, ``title``, ``steps``
     - Create a new plan with numbered steps
   * - ``update``
     - ``plan_id``, ``title``, ``steps``
     - Replace plan title and/or steps
   * - ``list``
     - (none)
     - List all plans with their statuses
   * - ``get``
     - ``plan_id``
     - Get full details of a specific plan
   * - ``set_active``
     - ``plan_id``
     - Set which plan is currently active
   * - ``mark_step``
     - ``plan_id``, ``step_index``, ``step_status``, ``step_note``
     - Update a step's status and notes
   * - ``delete``
     - ``plan_id``
     - Delete a plan

Step Statuses
-------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Status
     - Meaning
   * - ``not_started``
     - Step has not been attempted yet
   * - ``in_progress``
     - Step is currently being executed
   * - ``completed``
     - Step finished successfully
   * - ``blocked``
     - Step cannot proceed (dependency not met)

Storage
-------

Plans are stored in an **in-memory Python dictionary** (``self.plans``).
This means:

- Plans are **ephemeral** -- lost when the process exits
- There is **no persistence** across sessions
- Each ``PlanningFlow`` instance has its own ``PlanningTool`` with
  isolated plan storage

.. note::

   This is a significant architectural limitation. The planning system
   has no mechanism to save plans for later resumption, share plans
   across processes, or learn from past plans.


Planning System Prompt
======================

**File**: ``app/prompt/planning.py``

The planning system uses a specialized prompt that instructs the LLM to:

1. Create structured, actionable plans
2. Break complex tasks into manageable steps
3. Consider dependencies between steps
4. Track progress and adjust plans dynamically

Key prompt excerpt (``PLANNING_SYSTEM_PROMPT``):

::

    You are an expert planning agent. Your job is to create
    and manage plans to solve complex tasks.

    When creating a plan:
    - Break the task into clear, actionable steps
    - Order steps logically (consider dependencies)
    - Make each step specific enough to be executable
    - Include verification/validation steps when appropriate

The ``NEXT_STEP_PROMPT`` emphasizes efficiency:

::

    Focus on completing the current step efficiently.
    Use the most appropriate tools for the task.
    Report your progress clearly.


Agent Selection in Planning Flow
================================

The ``PlanningFlow`` maintains a dictionary of available agents:

.. code-block:: python

    agents = {
        "manus": Manus,         # General-purpose (primary)
        "data_analysis": DataAnalysis  # Optional
    }

Agent selection is done by the ``get_executor()`` method, which can
route steps to specialized agents based on step content or type. The
primary agent (``Manus``) is used as the default fallback.

Current Implementation Limitations
-----------------------------------

- The agent selection logic is relatively simple
- There is no automatic agent routing based on step content analysis
- The user must pre-register available agents at flow creation time
- Agents do not share memory or context between steps (only explicit
  step notes provide cross-step context)


Inter-Agent Communication
=========================

Agents in a ``PlanningFlow`` communicate **indirectly** through:

1. **Plan State**: The ``PlanningTool`` serves as a shared state store.
   Each agent can read the full plan, step statuses, and step notes.

2. **Step Notes**: After completing a step, the result is stored as a
   step note. Subsequent agents see these notes as context.

3. **Step Prompts**: The flow builds prompts for each step that include
   context from previous steps' results.

There is **no direct agent-to-agent communication** -- all coordination
flows through the ``PlanningFlow`` orchestrator and the ``PlanningTool``.

::

    Agent A                    PlanningFlow              Agent B
      │                            │                        │
      │  Execute Step 1            │                        │
      │ ◄──────────────────────────│                        │
      │                            │                        │
      │  Return result             │                        │
      │ ──────────────────────────►│                        │
      │                            │                        │
      │                            │  Update plan notes     │
      │                            │  Select Agent B        │
      │                            │  Build step 2 prompt   │
      │                            │  (includes step 1      │
      │                            │   results as context)  │
      │                            │                        │
      │                            │  Execute Step 2        │
      │                            │ ──────────────────────►│
      │                            │                        │
      │                            │  Return result         │
      │                            │ ◄──────────────────────│


Error Handling and Recovery
===========================

Step-Level Errors
-----------------

If an agent encounters an error during step execution:

1. The step can be marked as ``blocked``
2. The ``PlanningFlow`` can ask the LLM to revise the plan
3. Alternative steps can be created to work around the blockage
4. The primary agent can be used as a fallback executor

Agent-Level Errors
------------------

If an agent's entire ``run()`` fails:

1. The agent's state transitions to ``ERROR``
2. The ``PlanningFlow`` catches the exception
3. The flow can retry with the same or different agent
4. The overall flow terminates if recovery fails

Stuck Detection
---------------

Each agent has built-in stuck detection (``is_stuck()``). If an agent
produces identical responses repeatedly:

1. The ``next_step_prompt`` is modified to encourage different approaches
2. After ``duplicate_threshold`` (default: 2) identical responses, the
   agent is pushed to try new strategies


Comparison: Single-Agent vs. Multi-Agent
=========================================

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Aspect
     - Single-Agent (main.py)
     - Multi-Agent (run_flow.py)
   * - Entry Point
     - ``python main.py``
     - ``python run_flow.py``
   * - Planning
     - Implicit (LLM decides step by step)
     - Explicit (structured plan with PlanningTool)
   * - Agent Count
     - 1 (Manus)
     - 1+ (Manus + optional specialists)
   * - Tool Access
     - Single agent's tool set
     - Each agent has its own specialized tools
   * - Coordination
     - Self-directed
     - Orchestrated by PlanningFlow
   * - Max Steps
     - 20 (Manus default)
     - Configurable per flow + per agent
   * - Complexity Handling
     - Good for straightforward tasks
     - Better for multi-faceted tasks
   * - Stability
     - Stable (recommended)
     - Unstable (as noted in README)


A2A Protocol Integration
=========================

**Directory**: ``protocol/a2a/``

OpenManus can be exposed as an **Agent-to-Agent (A2A)** service,
allowing other agents or systems to invoke it:

Components
----------

- **A2AManus** (``protocol/a2a/app/agent.py``): Extends ``Manus`` with
  ``invoke()`` and ``stream()`` methods for A2A compatibility
- **ManusExecutor** (``protocol/a2a/app/agent_executor.py``): Implements
  the A2A ``AgentExecutor`` interface
- **Server** (``protocol/a2a/app/main.py``): Starlette-based A2A server
  on port 10000

Agent Card
----------

The A2A server advertises its capabilities via an ``AgentCard``:

.. code-block:: python

    AgentCard(
        name="Manus Agent",
        description="A versatile agent that can solve various tasks...",
        url="http://localhost:10000/",
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=True
        ),
        skills=[
            AgentSkill(id="Python Execute", ...),
            AgentSkill(id="Browser use", ...),
            AgentSkill(id="Replace String", ...),
            AgentSkill(id="Ask human", ...),
            AgentSkill(id="terminate", ...),
        ]
    )
