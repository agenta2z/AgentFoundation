# Data Flow Diagrams

Visual diagrams showing how data flows through the agent framework at key stages.

**Related documents:** [Execution Flow](02_execution_flow.md) | [WorkGraph](03_workgraph.md) | [State & Memory](05_state_and_memory.md)

---

## Input Resolution Flow

```mermaid
flowchart TB
    Start["Agent.__call__(*args, **kwargs)"]
    Resolve["_resolve_task_input_from_call_args(args, kwargs)"]

    subgraph TaskInputTypes["task_input could be:"]
        T1["None → read from interactive.get_input()"]
        T2["str → direct user_input"]
        T3["dict with 'user_input' key"]
    end

    Extract["_extract_user_input_and_metadata(raw_input)"]

    subgraph Returns["Returns"]
        R1["user_input: Clean input for prompts"]
        R2["task_input_metadata: {session_id, ...}"]
        R3["agent_fields: {user_profile, context, ...}"]
    end

    Start --> Resolve
    Resolve --> TaskInputTypes
    TaskInputTypes --> Extract
    Extract --> Returns
```

## Response Parsing Flow (PromptBasedActionAgent)

```mermaid
flowchart TB
    Raw["raw_response (LLM output)"]

    subgraph Parse["_parse_raw_response()"]
        P1["Handle InferencerResponse wrapper"]
        P2["Extract between delimiters"]
        P3{"Structured or Plain?"}
    end

    subgraph Structured["Structured Response"]
        S1["xml_to_dict() or json.loads()"]
        S2["Parsed structure with InstantResponse, TaskStatus, ImmediateNextActions"]
    end

    subgraph Extract["_extract_from_raw_response_parse()"]
        E1["_create_agent_response() → AgentResponse"]
        E2["_create_agent_state() → AgentStateItem"]
    end

    Output["(agent_response, agent_state)"]

    Raw --> Parse
    P1 --> P2 --> P3
    P3 -->|"Structured"| Structured
    P3 -->|"Plain"| PlainText["Plain text response"]
    Structured --> Extract
    S1 --> S2
    E1 --> Output
    E2 --> Output
```

## Action Execution Flow

```mermaid
flowchart TB
    NextActions["next_actions = [[A], [B, C], [D]]"]

    subgraph BuildGraph["Build WorkGraph"]
        B1{"len(group) == 1?"}
        B2["Sequential: WorkGraphNode(_run_single_action)"]
        B3["Parallel: Create branched agents"]
        B4["Wire: action → branched_agent → summary"]
    end

    subgraph Execute["work_graph.run()"]
        E1["1. A executes"]
        E2["2. B, C execute in parallel"]
        E3["3. Results merged at summary_node"]
        E4["4. D executes with merged results"]
    end

    Result["work_graph_final_result = {user_input, action_results}"]
    FeedBack["action_results feeds back to next iteration"]

    NextActions --> BuildGraph
    B1 -->|"Yes"| B2
    B1 -->|"No"| B3
    B3 --> B4
    BuildGraph --> Execute
    E1 --> E2 --> E3 --> E4
    Execute --> Result
    Result --> FeedBack
```

---

**Previous:** [State & Memory](05_state_and_memory.md) | **Next:** [API Reference](07_api_reference.md)
