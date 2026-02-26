# DevMate Inferencer Example

This example demonstrates how to use the `DevmateInferencer` to interact with the DevMate CLI programmatically from Python.

## Overview

The `DevmateInferencer` is a terminal-based inferencer that wraps the `devmate` CLI tool, allowing you to:
- Send prompts to DevMate and receive responses
- Configure model selection and token limits
- Use it programmatically in your Python scripts

## Usage

### Running with Buck2

```bash
# Interactive mode (enter queries in a loop)
buck2 run fbcode//_tony_dev/ScienceModelingTools/examples/agent_foundation/common/inferencers/terminal_inferencers/devmate:devmate_example

# Single query
buck2 run fbcode//_tony_dev/ScienceModelingTools/examples/agent_foundation/common/inferencers/terminal_inferencers/devmate:devmate_example -- "What is Python?"

# Dry run (show command without executing)
buck2 run fbcode//_tony_dev/ScienceModelingTools/examples/agent_foundation/common/inferencers/terminal_inferencers/devmate:devmate_example -- --dry-run "Explain decorators"

# Custom model
buck2 run fbcode//_tony_dev/ScienceModelingTools/examples/agent_foundation/common/inferencers/terminal_inferencers/devmate:devmate_example -- --model claude-3-opus "Write a haiku"

# Quiet mode (response only)
buck2 run fbcode//_tony_dev/ScienceModelingTools/examples/agent_foundation/common/inferencers/terminal_inferencers/devmate:devmate_example -- --quiet "What is 2+2?"
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `prompt` | The prompt to send (optional, enters interactive mode if not provided) |
| `--model`, `-m` | Model to use (default: `claude-sonnet-4.5`) |
| `--max-tokens`, `-t` | Maximum response tokens (default: `32768`) |
| `--timeout` | Timeout in seconds (default: `300`) |
| `--dry-run`, `-d` | Show command without executing |
| `--quiet`, `-q` | Minimal output (response only) |

### Interactive Mode Commands

When running in interactive mode:
- **Type your question** and press Enter to query DevMate
- **`quit`**, **`exit`**, or **`q`** - Exit the program
- **`help`** - Show help message
- **`config`** - Show current configuration

## Programmatic Usage

You can also use the `DevmateInferencer` directly in your Python code:

```python
from science_modeling_tools.common.inferencers.terminal_inferencers.devmate.devmate_inferencer import (
    DevmateInferencer,
)

# Create an inferencer
inferencer = DevmateInferencer(
    model_name="claude-sonnet-4.5",
    max_tokens=32768,
    timeout=300,
    no_create_commit=True,
)

# Run a query
result = inferencer.infer("What is Python?")

# Check if successful
if result["success"]:
    print(result["output"])
else:
    print(f"Error: {result.get('error')}")

# Or use the helper method
response_text = inferencer.get_response_text(result)
print(response_text)
```

## Example Output

```
============================================================
  DevMate Inferencer - Interactive Example
============================================================

This example demonstrates using DevmateInferencer to interact
with the DevMate CLI programmatically.

🎯 Interactive Mode
============================================================
Enter your queries below. Commands:
  - Type your question and press Enter to query DevMate
  - Type 'quit', 'exit', or 'q' to exit
  - Type 'help' for this message
  - Type 'config' to show current configuration
============================================================

🤖 You: What is 2+2?

📝 Prompt: What is 2+2?
------------------------------------------------------------
⏳ Sending to DevMate...

✅ Response:
------------------------------------------------------------
2 + 2 = 4
------------------------------------------------------------

⏱️  Execution time: 2.34s
📊 Return code: 0
```

## Related Files

- **Source**: `src/science_modeling_tools/common/inferencers/terminal_inferencers/devmate/devmate_inferencer.py`
- **Tests**: `test/science_modeling_tools/common/inferencers/terminal_inferencers/test_devmate_inferencer.py`
- **Live Tests**: `test/science_modeling_tools/common/inferencers/terminal_inferencers/test_devmate_inferencer_live.py`
