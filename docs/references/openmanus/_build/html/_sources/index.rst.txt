====================================
OpenManus Documentation
====================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Welcome to the OpenManus documentation. OpenManus is an open-source AI agent
framework built by the MetaGPT team, designed to enable LLM-powered agents to
solve arbitrary tasks using a variety of tools including browser automation,
code execution, file editing, web search, and Model Context Protocol (MCP)
integration.

Overview
========

OpenManus provides:

- A **hierarchical agent architecture** with a clear inheritance chain from
  abstract base agents to specialized concrete agents
- A **ReAct (Reason + Act) execution loop** where agents iteratively think
  (call LLM) and act (execute tools)
- A **multi-agent planning flow** for decomposing complex tasks into steps
  assigned to specialized agents
- A **rich tool ecosystem** including browser automation, code execution,
  file editing, web search, and desktop automation
- **MCP (Model Context Protocol)** integration for both client and server
  roles, enabling dynamic tool discovery and inter-agent communication
- **Sandboxed execution** via Docker containers and Daytona cloud sandboxes
- An **A2A (Agent-to-Agent) protocol** integration for exposing agents as
  services

Documentation Structure
=======================

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Document
     - Description
   * - :doc:`architecture`
     - High-level system architecture, design patterns, and component overview
   * - :doc:`agent_system`
     - Agent class hierarchy, lifecycle, state machine, and configuration
   * - :doc:`orchestration_flow`
     - Agentic orchestration patterns, PlanningFlow, multi-agent coordination
   * - :doc:`tool_system`
     - Tool abstraction, all available tools, tool collections, and execution
   * - :doc:`knowledge_and_skills`
     - Knowledge handling, memory system, skill management, and limitations
   * - :doc:`llm_integration`
     - LLM client, prompt management, token counting, and API abstraction
   * - :doc:`configuration`
     - Configuration system, TOML format, settings models, and examples
   * - :doc:`mcp_integration`
     - MCP client and server, dynamic tool discovery, transport protocols
   * - :doc:`sandbox_system`
     - Docker sandbox, Daytona cloud sandbox, safety mechanisms
   * - :doc:`api_reference`
     - Technical API specifications for all modules
   * - :doc:`getting_started`
     - Installation, configuration, quick start, and tutorials
   * - :doc:`tutorials`
     - Step-by-step tutorials for common use cases

Quick Links
===========

- **GitHub Repository**: https://github.com/FoundationAgents/OpenManus
- **License**: MIT
- **Python**: 3.11 - 3.13 (3.12 recommended)
- **Version**: 0.1.0

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   architecture
   agent_system
   orchestration_flow
   tool_system
   knowledge_and_skills
   llm_integration
   configuration
   mcp_integration
   sandbox_system
   api_reference
   tutorials
