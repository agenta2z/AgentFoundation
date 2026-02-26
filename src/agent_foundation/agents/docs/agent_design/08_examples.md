# Example: Complete Request Processing

End-to-end trace of a user request through the agent system.

**Related documents:** [Execution Flow](02_execution_flow.md) | [WorkGraph](03_workgraph.md) | [Branching](04_branching.md)

---

## User Request
```
"Order a coffee from Starbucks and a burger from McDonald's"
```

```mermaid
sequenceDiagram
    participant User
    participant Agent as Agent.__call__()
    participant Reasoner as Reasoner (LLM)
    participant WorkGraph
    participant StarbucksAgent as Branched Agent (Starbucks)
    participant McDonaldsAgent as Branched Agent (McDonalds)
    participant Actor

    User->>Agent: "Order coffee from Starbucks and burger from McDonald's"

    Note over Agent: STEP 1: Extract user_input

    Agent->>Reasoner: _construct_reasoner_input()
    Reasoner-->>Agent: Structured XML Response

    Note over Agent: Parse: next_actions = [[(Starbucks, McDonalds)]]<br/>Both in one group = PARALLEL

    Agent->>WorkGraph: Build parallel branches

    par Parallel Execution
        WorkGraph->>StarbucksAgent: action_node → branched_agent.__call__()
        StarbucksAgent->>Actor: Execute Starbucks order
        Actor-->>StarbucksAgent: Order result
        StarbucksAgent->>Reasoner: Continue reasoning...
        Reasoner-->>StarbucksAgent: Completed
    and
        WorkGraph->>McDonaldsAgent: action_node → branched_agent.__call__()
        McDonaldsAgent->>Actor: Execute McDonalds order
        Actor-->>McDonaldsAgent: Order result
        McDonaldsAgent->>Reasoner: Continue reasoning...
        Reasoner-->>McDonaldsAgent: Completed
    end

    WorkGraph->>WorkGraph: Summarize results
    WorkGraph-->>Agent: Merged action_results

    Agent->>Reasoner: Next iteration with merged results
    Reasoner-->>Agent: agent_state == Completed

    Agent->>Agent: _get_agent_results()
    Agent->>Agent: _finalize_and_send_agent_results()
    Agent-->>User: Final response with summaries
```

## Step-by-Step Breakdown

**Step 1: Agent.__call__() receives request**
```python
# task_input = "Order a coffee from Starbucks and a burger from McDonald's"
user_input, metadata, agent_fields = _extract_user_input_and_metadata(task_input)
```

**Step 2: First reasoning iteration**
```python
reasoner_input = _construct_reasoner_input(
    user_input="Order a coffee from Starbucks and a burger from McDonald's",
    user_profile=None,
    context=None,
    action_results=None  # First iteration, no previous results
)

raw_response = self.reasoner(reasoner_input, config)
```

**Step 3: Parse response**
```xml
<StructuredResponse>
  <InstantResponse>I'll help you place both orders.</InstantResponse>
  <TaskStatus>Ongoing</TaskStatus>
  <ImmediateNextActions>
    <AlternativeActions>
      <Action>
        <Type>Order.Starbucks</Type>
        <Target>coffee</Target>
        <Reasoning>Need to order from Starbucks</Reasoning>
      </Action>
      <Action>
        <Type>Order.McDonalds</Type>
        <Target>burger</Target>
        <Reasoning>Need to order from McDonald's</Reasoning>
      </Action>
    </AlternativeActions>
  </ImmediateNextActions>
</StructuredResponse>
```

**Step 4: Build and execute WorkGraph**
- Since `len(action_group) > 1`, create parallel branches
- Each branch: `action_node → branched_agent_node → summary_node`

**Step 5: Branched agents execute recursively**
- Each executes its immediate action
- Calls `branched_agent.__call__()` for follow-up reasoning
- Returns when reasoner says `Completed`

**Step 6: Results merge and return**
```python
agent_results = _get_agent_results(...)  # Structured summary
_finalize_and_send_agent_results(agent_response, agent_results, ...)
return agent_results
```

---

**Previous:** [API Reference](07_api_reference.md) | **Next:** [Workflow Control](09_workflow_control.md)
