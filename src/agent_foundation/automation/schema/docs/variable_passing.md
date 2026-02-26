# Variable Passing in ActionGraph and ActionFlow

This document explains how template variables are passed down through the action execution hierarchy, from ActionGraph invocation to individual ActionNode execution.

## ActionGraph: Single-Root-Node DAG

ActionGraph is a **single-root-node DAG** (Directed Acyclic Graph):
- All workflows start from a single root node created at initialization (`_nodes[0]`)
- Subsequent nodes are added via `condition()` and linked through `add_next()`
- Execution always begins from the root node

## Architecture Overview

```
ActionGraph.__call__(base_url='...', query='...')
    │
    ▼
ActionGraph.execute(initial_variables={...})
    │
    ▼
WorkGraph.run(ExecutionResult with context.variables)
    │
    ▼
ActionSequenceNode._execute_sequence(ExecutionResult)
    │
    ▼
ActionFlow.execute(initial_variables=extracted from ExecutionResult)
    │
    ▼
ActionFlow._run() iterates ActionNodes  [inherits from Workflow]
    │
    ▼
ActionNode._execute_action(context: ExecutionRuntime)
    │
    ▼
ActionNode.substitute_variables(context.variables)
```

## Component Responsibilities

### 1. ActionGraph (Entry Point)

**File:** `action_graph.py`

ActionGraph is the top-level workflow builder and entry point. It aggregates template variables from all nodes and validates them at invocation time.

#### `__call__(**variables)` (lines 634-670)

Makes ActionGraph callable for convenient workflow invocation:

```python
def __call__(self, **variables) -> ExecutionResult:
    # Validate all required variables are provided
    required = self.required_variables  # Aggregated from all nodes
    missing = required - set(variables.keys())
    if missing:
        raise ValueError(f"Missing required template variables: {missing}")

    return self.execute(initial_variables=variables)
```

#### `execute(initial_variables)` (lines 672-691)

Wraps variables in an `ExecutionResult` with `ExecutionRuntime.variables`:

```python
def execute(self, initial_variables: Optional[Dict[str, Any]] = None) -> ExecutionResult:
    self._set_start_node()

    # Create initial ExecutionResult with variables for first node
    initial_result = ExecutionResult(
        success=True,
        context=ExecutionRuntime(variables=initial_variables or {}),
    )
    return self.run(initial_result)  # WorkGraph.run()
```

#### `required_variables` Property (lines 295-309)

Aggregates variables from all ActionSequenceNodes:

```python
@property
def required_variables(self) -> Set[str]:
    all_vars: Set[str] = set()
    for node in self._nodes:
        all_vars.update(node.required_variables)
    return all_vars
```

### 2. ActionSequenceNode (Graph Node)

**File:** `action_graph.py` (lines 1516-1600)

Each node in the graph holds a sequence of actions. It passes variables down to ActionFlow.

#### `_execute_sequence(*args)` (lines 1582-1600)

Extracts variables from the incoming `ExecutionResult` and passes to ActionFlow:

```python
def _execute_sequence(self, *args, **kwargs) -> ExecutionResult:
    sequence = ActionSequence(
        id=f"sequence_{self.name}",
        actions=self._actions
    )

    # Extract variables from incoming ExecutionResult
    variables = {}
    if args and isinstance(args[0], ExecutionResult):
        variables = args[0].context.variables if args[0].context else {}

    executor = ActionFlow(
        action_executor=self.action_executor,
        action_metadata=self.action_metadata,
        template_engine=self.template_engine,
    )
    return executor.execute(sequence=sequence, initial_variables=variables)
```

#### `required_variables` Property (lines 1540-1570)

Aggregates from temporary ActionNodes to detect what variables this node needs:

```python
@property
def required_variables(self) -> Set[str]:
    if self._cached_required_variables is not None:
        return self._cached_required_variables

    all_vars: Set[str] = set()
    for action in self._actions:
        node = ActionNode(
            action=action,
            action_executor=self.action_executor,
            action_metadata=self.action_metadata,
            template_engine=self.template_engine,
        )
        all_vars.update(node.required_variables)

    self._cached_required_variables = all_vars
    return all_vars
```

### 3. ActionFlow (Sequential Executor)

**File:** `action_flow.py`

ActionFlow inherits from Workflow and orchestrates sequential execution of actions.

#### `execute(sequence, initial_variables)`

Creates `ExecutionRuntime` with variables and builds ActionNodes:

```python
def execute(
    self,
    sequence: Optional[Union[ActionSequence, str, Path]] = None,
    initial_variables: Optional[Dict[str, Any]] = None,
) -> ExecutionResult:
    # Set up runtime state
    self.context = ExecutionRuntime(variables=initial_variables or {})

    # Build ActionNodes from the sequence with template settings
    self.action_nodes = [
        ActionNode(
            action=action,
            action_executor=self.action_executor,
            action_metadata=self.action_metadata,
            template_engine=self.template_engine,  # Passed down
        )
        for action in sequence.actions
    ]

    # Execute using inherited Workflow.run() -> _run()
    self.run()
    return ExecutionResult(success=True, context=self.context)
```

#### `_run()` Method

ActionFlow's `_run()` iterates through ActionNodes with shared context (O(1) stack depth):

```python
def _run(self, *args, **kwargs):
    for node in self.action_nodes:
        # Execute with shared context
        result = node.run(self.context)  # Context passed to ActionNode

        # Store result in context for subsequent actions
        self.context.set_result(node.action.id, result)

    return self.context
```

### 4. ActionNode (Single Action Executor)

**File:** `action_node.py`

ActionNode is where template substitution actually happens. Each node autonomously handles its own variable detection and substitution.

#### Template Detection (lines 226-257)

At construction time, ActionNode scans target and args for template variables:

```python
def _detect_template_variables(self) -> None:
    compile_fn, _ = _get_template_utils(self.template_engine)

    # Get action metadata for arg_types (for type coercion)
    action_meta = self.action_metadata.get_metadata(self.action.type)
    parsed_arg_types = action_meta.parsed_arg_types if action_meta else {}

    # Scan target
    if isinstance(self.action.target, str):
        self._scan_template_value(value=self.action.target, ...)

    # Scan args
    if self.action.args:
        for arg_name, arg_value in self.action.args.items():
            if isinstance(arg_value, str):
                self._scan_template_value(value=arg_value, ...)
```

#### `_execute_action(context)` (lines 450-514)

Called with ExecutionRuntime context, substitutes variables before executing:

```python
def _execute_action(self, context: ExecutionRuntime) -> ActionResult:
    # Substitute template variables using context.variables
    resolved_target, resolved_args = self.substitute_variables(context.variables)

    # Execute action with resolved values
    result = self.action_executor(
        action_type=self.action.type,
        action_target=target_value,
        action_args=resolved_args,
    )

    # Store result in context if output variable specified
    if self.output_variable:
        context.set_variable(self.output_variable, action_result.value)

    # Always store last result as '_' for implicit reference
    context.set_variable('_', action_result.value)

    return action_result
```

#### `substitute_variables(variables)` (lines 303-355)

Performs the actual template substitution with type coercion:

```python
def substitute_variables(
    self,
    variables: Dict[str, Any],
) -> Tuple[Optional[...], Optional[Dict[str, Any]]]:
    # No template variables for this action - return originals
    if not self._required_variables:
        return self.action.target, self.action.args

    _, format_fn = _get_template_utils(self.template_engine)

    # Substitute target
    resolved_target = self._substitute_value(
        value=self.action.target,
        arg_name='_target',
        variables=variables,
        format_fn=format_fn,
    )

    # Substitute args
    resolved_args = {}
    for arg_name, arg_value in self.action.args.items():
        resolved_args[arg_name] = self._substitute_value(...)

    return resolved_target, resolved_args
```

## Variable Flow Diagram

```
User provides: {base_url: 'https://x.com', query: 'test', delay: 2.5}
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ActionGraph.__call__(base_url='https://x.com', query='test', ...)   │
│   - Validates: required_variables ⊆ provided variables              │
│   - Calls: execute(initial_variables=variables)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ActionGraph.execute(initial_variables={...})                        │
│   - Creates: ExecutionResult(                                       │
│       context=ExecutionRuntime(variables={...})                     │
│     )                                                               │
│   - Calls: _set_start_node() then WorkGraph.run(initial_result)     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ WorkGraph.run(initial_result: ExecutionResult)                      │
│   - Passes ExecutionResult to each ActionSequenceNode               │
│   - Uses ResultPassDownMode.ResultAsFirstArg                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ActionSequenceNode._execute_sequence(result: ExecutionResult)       │
│   - Extracts: variables = result.context.variables                  │
│   - Creates: ActionFlow(template_engine=self.template_engine)       │
│   - Calls: flow.execute(sequence, initial_variables=variables)      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ActionFlow.execute(initial_variables={...})                         │
│   - Sets: self.context = ExecutionRuntime(variables=...)            │
│   - Sets: self.action_nodes = [ActionNode(...) for action in ...]   │
│   - Calls: self.run()  # Inherited from Workflow                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ActionFlow._run()  [inherits from Workflow]                         │
│   - Iterates: for node in self.action_nodes                         │
│   - Calls: node.run(self.context)  # Shared context passed          │
│   - Stores: context.set_result(node.action.id, result)              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ActionNode._execute_action(context: ExecutionRuntime)               │
│   - Substitutes: resolved_target, resolved_args =                   │
│                  self.substitute_variables(context.variables)       │
│   - Applies type coercion for single-var templates                  │
│   - Executes: action_executor(type, target, args)                   │
│   - Stores output: context.set_variable(self.output_variable, val)  │
│   - Stores implicit: context.set_variable('_', val)                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Type Coercion Rules

ActionNode applies type coercion based on the template pattern:

### Single-Variable Templates

Templates that are exactly one variable (e.g., `'{delay}'`) can preserve or coerce types.

```python
# wait action has arg_types={'seconds': 'float'}
graph.wait(seconds='{delay}')

# At runtime with delay='2.5' (string):
# → coerce_to_type('2.5', (float,)) → 2.5 (float)

# At runtime with delay=2.5 (float):
# → coerce_to_type(2.5, (float,)) → 2.5 (float, preserved)
```

### Multi-Variable Templates

Templates with multiple variables or prefix/suffix always return strings.

```python
graph.visit_url('{base_url}/api/{version}')

# At runtime with base_url='https://x.com', version='v2':
# → format_template(...) → 'https://x.com/api/v2' (string concatenation)
```

## Output Variables and Result Chaining

Actions can store their results for use by subsequent actions:

```python
graph.get_element('#search', output='search_box')  # Store as 'search_box'
graph.input_text('{search_box}', text='{query}')   # Reference stored variable
graph.click('{_}')                                  # Reference last result
```

The `_` variable always holds the last action's result:

```python
# In ActionNode._execute_action:
context.set_variable('_', action_result.value)  # Always set
```

## Template Engine Support

The system supports multiple template engines via the `template_engine` attribute:

| Engine | Syntax | Example |
|--------|--------|---------|
| `python` (default) | `{var}` | `{base_url}` |
| `jinja2` | `{{ var }}` | `{{ base_url }}` |
| `handlebars` | `{{var}}` | `{{base_url}}` |
| `string_template` | `$var` or `${var}` | `$base_url` |

Each ActionNode uses `_get_template_utils(engine)` to get the appropriate `compile_template` and `format_template` functions.

## Summary

1. **ActionGraph** validates and wraps variables in `ExecutionResult.context.variables`
2. **ActionSequenceNode** extracts variables and passes to ActionFlow
3. **ActionFlow** creates `ExecutionRuntime` with variables, builds ActionNodes, and iterates with O(1) stack depth
4. **ActionNode** autonomously substitutes variables using `context.variables`

Each level passes the same variable dictionary down; only ActionNode performs actual substitution. This "autonomous node" design means:
- Each ActionNode knows what variables it needs (`required_variables`)
- Parent nodes aggregate to know total requirements
- Substitution happens at execution time, close to where values are used
- Type coercion is applied based on action metadata (`arg_types`)
