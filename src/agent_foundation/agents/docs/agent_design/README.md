# Agent Architecture Documentation

Comprehensive documentation for the Agent framework — an agentic loop system with DAG-based task execution, branched recursion, and multi-pattern memory propagation.

## Reading Order

| # | Document | Description |
|---|----------|-------------|
| 1 | [Architecture](01_architecture.md) | Class hierarchy, core data structures (`AgentResponse`, `AgentAction`, `AgentStateItem`) |
| 2 | [Execution Flow](02_execution_flow.md) | Complete `Agent.__call__()` flow diagram and the reasoner-action feedback loop |
| 3 | [WorkGraph](03_workgraph.md) | How WorkGraph executes actions — argument passing, two creation methods (action-based vs path-based), polymorphic dispatch |
| 4 | [Branching](04_branching.md) | Parallel execution via branched recursion — agent copies, wiring, termination |
| 5 | [State & Memory](05_state_and_memory.md) | **How memory flows between nodes** — six propagation patterns, `action_results` lifecycle, parallel isolation, cross-agent context passing |
| 6 | [Data Flow](06_data_flow.md) | Visual diagrams for input resolution, response parsing, and action execution |
| 7 | [API Reference](07_api_reference.md) | Key methods reference for all agent classes |
| 8 | [Examples](08_examples.md) | End-to-end trace of a user request through the system |
| 9 | [Workflow Control](09_workflow_control.md) | Stop/pause/resume mechanics and framework summary |
| 10 | [Knowledge Integration](10_knowledge_integration.md) | End-to-end flow: knowledge ingestion, three-layer storage, retrieval, and injection into agent prompts via `knowledge_provider` |
| 11 | [Prompt Template Integration](11_prompt_template_integration.md) | How `PromptBasedAgent` integrates with `TemplateManager` — state-driven template selection, hierarchical fallback, versioning, and runtime switching |

## Quick Links

- **New to the framework?** Start with [Architecture](01_architecture.md)
- **How does the agent execute tasks?** See [Execution Flow](02_execution_flow.md) and [WorkGraph](03_workgraph.md)
- **How does memory flow between nodes?** See [State & Memory](05_state_and_memory.md)
- **How do parallel branches work?** See [Branching](04_branching.md)
- **How does knowledge get into agents?** See [Knowledge Integration](10_knowledge_integration.md)
- **How do prompt templates work?** See [Prompt Template Integration](11_prompt_template_integration.md)
