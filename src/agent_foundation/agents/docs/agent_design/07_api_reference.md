# Key Methods Reference

Quick reference for the most important methods in each agent class.

**Related documents:** [Architecture](01_architecture.md) | [Execution Flow](02_execution_flow.md) | [WorkGraph](03_workgraph.md)

---

## Agent (Base Class)

| Method | Purpose | Lines |
|--------|---------|-------|
| `__call__` | Main entry point, orchestrates the execution loop | 1048-1675 |
| `_construct_reasoner_input` | Build input for the reasoner | 488-523 |
| `_parse_raw_response` | Convert raw response to (AgentResponse, AgentStateItem) | 525-546 |
| `_run_single_action` | Execute one action via actor | 723-845 |
| `_resolve_actor_for_next_action` | Find the right actor for an action type | 613-654 |
| `_get_agent_results` | Transform action_results to presentation format | 847-901 |
| `_finalize_and_send_agent_results` | Send final response with metadata | 943-1002 |
| `copy` | Create independent agent copy for branching | 678-709 |
| `stop/pause/resume/step_by_step` | Workflow control methods | 1735-1784 |
| `close` | Cleanup resources | 1786-1796 |

## PromptBasedAgent

| Method | Purpose |
|--------|---------|
| `_construct_prompt_feed` | Build dict of template placeholders → values |
| `_construct_reasoner_input` | Apply template with feed to create prompt |
| `_parse_raw_response` | Extract structured response from delimiters, parse XML/JSON |
| `_get_user_input_string` | Format user input (optionally as conversation) |
| `_get_state_strings` | Build current/previous state strings for context |

## PromptBasedActionAgent

| Method | Purpose |
|--------|---------|
| `_create_action_item` | Parse raw action dict → AgentAction |
| `_create_next_actions` | Process all action items |
| `_create_agent_response` | Build AgentResponse with parsed actions |
| `_create_agent_state` | Build AgentStateItem from response |
| `_get_agent_results` | Extract AgentActionResult from states |

## PromptBasedActionPlanningAgent

| Method | Purpose |
|--------|---------|
| `_create_action_item` | Create partial(actor, **args) for each problem |
| `_create_next_actions` | Build WorkGraph from problem dependency graph |
| `_create_agent_state` | Returns None (planning agents don't track state traditionally) |

---

**Previous:** [Data Flow](06_data_flow.md) | **Next:** [Examples](08_examples.md)
