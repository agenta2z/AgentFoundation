#!/bin/bash
# Run the DevMate Inferencer Interactive Example
#
# Usage:
#   ./run.sh                                    # Interactive mode (default model)
#   ./run.sh "What is Python?"                  # Single query
#   ./run.sh -m claude-3-opus "Write a haiku"   # Use specific model
#   ./run.sh --model claude-3-opus              # Interactive with specific model
#   ./run.sh --dry-run "Test prompt"            # Show command without executing
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

# Parse arguments
show_help() {
    echo "DevMate Inferencer Example"
    echo ""
    echo "Usage: ./run.sh [OPTIONS] [PROMPT]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL      Model to use (default: claude-sonnet-4.5)"
    echo "  -t, --max-tokens N     Maximum tokens for response (default: 32768)"
    echo "  --timeout SECONDS      Timeout in seconds (default: 300)"
    echo "  -d, --dry-run          Show command without executing"
    echo "  -q, --quiet            Minimal output (response only)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                                    # Interactive mode"
    echo "  ./run.sh \"What is Python?\"                  # Single query"
    echo "  ./run.sh -m claude-3-opus \"Write a haiku\"   # Use specific model"
    echo "  ./run.sh --model claude-3-opus              # Interactive with model"
    echo "  ./run.sh --dry-run \"Test prompt\"            # Dry run mode"
    echo ""
    echo "Available models (examples):"
    echo "  - claude-sonnet-4.5 (default)"
    echo "  - claude-3-opus"
    echo "  - claude-3-sonnet"
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
echo ""
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

# Add the prompt if provided
if [[ ${#ARGS[@]} -gt 0 ]]; then
    CMD_ARGS+=("${ARGS[@]}")
fi

# Run the example with buck2
buck2 run fbcode//_tony_dev/ScienceModelingTools/examples/agent_foundation/common/inferencers/terminal_inferencers/devmate:devmate_example -- "${CMD_ARGS[@]}"
