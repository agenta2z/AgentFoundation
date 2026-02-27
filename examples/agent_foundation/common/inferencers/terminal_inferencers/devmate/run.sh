#!/bin/bash
# Run the DevMate Inferencer Interactive Example
#
# Usage:
#   ./run.sh                                    # Interactive mode (default model)
#   ./run.sh "What is Python?"                  # Single query
#   ./run.sh -s "What is Python?"               # Streaming mode - real-time output
#   ./run.sh --streaming "Explain decorators"   # Streaming single query
#   ./run.sh -s -o response.txt "Hello"         # Stream to terminal and file
#   ./run.sh -m claude-3-opus "Write a haiku"   # Use specific model
#   ./run.sh --model claude-3-opus              # Interactive with specific model
#   ./run.sh --dry-run "Test prompt"            # Show command without executing
#   ./run.sh --follow-up "What was my question" # Multi-turn demo with defaults
#   ./run.sh "Hello" -f "Follow up"             # Custom multi-turn demo
#   ./run.sh --help                             # Show all options
#
# Available models (examples):
#   - claude-sonnet-4.5 (default)
#   - claude-3-opus
#   - claude-3-sonnet
#   - gpt-4
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
MODEL="claude-sonnet-4.5"
MAX_TOKENS="32768"
TIMEOUT="300"
DRY_RUN=""
QUIET=""
PROMPT=""
FOLLOWUP=""
STREAMING=""
OUTPUT_FILE=""

# Parse arguments
show_help() {
    echo "DevMate Inferencer Example with Session Support and Streaming"
    echo ""
    echo "Usage: ./run.sh [OPTIONS] [PROMPT]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL      Model to use (default: claude-sonnet-4.5)"
    echo "  -t, --max-tokens N     Maximum tokens for response (default: 32768)"
    echo "  --timeout SECONDS      Timeout in seconds (default: 300)"
    echo "  -s, --streaming        Enable streaming mode (real-time output)"
    echo "  -o, --output-file FILE Write streaming output to file"
    echo "  -f, --follow-up TEXT   Follow-up prompt for multi-turn demo"
    echo "  -d, --dry-run          Show command without executing"
    echo "  -q, --quiet            Minimal output (response only)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                                    # Interactive mode"
    echo "  ./run.sh \"What is Python?\"                  # Single query"
    echo "  ./run.sh -s \"What is Python?\"               # Streaming mode"
    echo "  ./run.sh --streaming \"Explain decorators\"   # Streaming single query"
    echo "  ./run.sh -s -o response.txt \"Hello\"         # Stream to file"
    echo "  ./run.sh -m claude-3-opus \"Write a haiku\"   # Use specific model"
    echo "  ./run.sh --model claude-3-opus              # Interactive with model"
    echo "  ./run.sh --dry-run \"Test prompt\"            # Dry run mode"
    echo ""
    echo "Streaming examples:"
    echo "  ./run.sh -s                                 # Interactive with streaming ON"
    echo "  ./run.sh --streaming \"Explain Python\"       # Single streaming query"
    echo "  ./run.sh -s -o out.txt \"Explain Python\"     # Stream to terminal and file"
    echo ""
    echo "Multi-turn examples (session continuation):"
    echo "  ./run.sh \"\" --follow-up \"\"                  # Use default prompts"
    echo "  ./run.sh \"What is Python?\" -f \"What was my last question?\""
    echo ""
    echo "Default prompts for multi-turn demo:"
    echo "  Initial: \"What is Python?\""
    echo "  Follow-up: \"What was my last question?\""
    echo ""
    echo "Available models (examples):"
    echo "  - claude-sonnet-4.5 (default)"
    echo "  - claude-3-opus"
    echo "  - claude-3-sonnet"
    echo ""
    echo "Note: The command will display the actual devmate command that will be executed,"
    echo "      including the working directory change and shell command format."
    echo ""
}

# Process arguments
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -s|--streaming)
            STREAMING="--streaming"
            shift
            ;;
        -o|--output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -f|--follow-up)
            FOLLOWUP="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        -q|--quiet)
            QUIET="--quiet"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            # Collect remaining arguments (prompt)
            ARGS+=("$1")
            shift
            ;;
    esac
done

echo "============================================================"
echo "  DevMate Inferencer Example"
echo "============================================================"
echo ""
echo "Model: $MODEL"
echo "Max tokens: $MAX_TOKENS"
echo "Timeout: ${TIMEOUT}s"
if [[ -n "$FOLLOWUP" || ${#ARGS[@]} -eq 0 && -n "$FOLLOWUP" ]]; then
    echo "Mode: Multi-turn session demo"
fi
echo ""

# Show expected command format for reference
if [[ -z "$QUIET" ]]; then
    echo "Expected command format (shell execution):"
    echo "  cd \"\$HOME/fbsource\" && devmate run freeform \"prompt=...\" \"model_name=$MODEL\" \"max_tokens=$MAX_TOKENS\" --no-create-commit"
    echo ""
    echo "For session resume:"
    echo "  cd \"\$HOME/fbsource\" && devmate run --resume --session-id 'UUID' freeform \"prompt=...\" ..."
    echo ""
fi

echo "Running with buck2..."
echo ""

# Build the command arguments
CMD_ARGS=()
CMD_ARGS+=("--model" "$MODEL")
CMD_ARGS+=("--max-tokens" "$MAX_TOKENS")
CMD_ARGS+=("--timeout" "$TIMEOUT")

if [[ -n "$DRY_RUN" ]]; then
    CMD_ARGS+=("$DRY_RUN")
fi

if [[ -n "$QUIET" ]]; then
    CMD_ARGS+=("$QUIET")
fi

if [[ -n "$STREAMING" ]]; then
    CMD_ARGS+=("$STREAMING")
fi

if [[ -n "$OUTPUT_FILE" ]]; then
    CMD_ARGS+=("--output-file" "$OUTPUT_FILE")
fi

# Add the prompt if provided
if [[ ${#ARGS[@]} -gt 0 ]]; then
    CMD_ARGS+=("${ARGS[@]}")
fi

# Add follow-up if provided
if [[ -n "$FOLLOWUP" || "$FOLLOWUP" == "" && ${#ARGS[@]} -gt 0 ]]; then
    # Only add --follow-up if it was explicitly set (even if empty)
    if [[ "${!#}" == "-f" || "${!#}" == "--follow-up" || -n "$FOLLOWUP" ]]; then
        CMD_ARGS+=("--follow-up" "$FOLLOWUP")
    fi
fi

# Check if follow-up was explicitly requested
for arg in "$@"; do
    if [[ "$arg" == "-f" || "$arg" == "--follow-up" ]]; then
        # Ensure --follow-up is in CMD_ARGS
        if [[ ! " ${CMD_ARGS[*]} " =~ " --follow-up " ]]; then
            CMD_ARGS+=("--follow-up" "$FOLLOWUP")
        fi
        break
    fi
done

# Run the example with buck2
buck2 run fbcode//_tony_dev/ScienceModelingTools/examples/agent_foundation/common/inferencers/terminal_inferencers/devmate:devmate_example -- "${CMD_ARGS[@]}"
