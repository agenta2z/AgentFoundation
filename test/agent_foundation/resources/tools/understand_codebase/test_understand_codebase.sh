#!/usr/bin/env bash
# understand_codebase tool test — delegates to task with understand_codebase preamble.
#
# Usage:
#   bash test/agent_foundation/resources/tools/understand_codebase/test_understand_codebase.sh
#   bash test/agent_foundation/resources/tools/understand_codebase/test_understand_codebase.sh /path/to/code
#   UC_MODEL=sonnet bash test/agent_foundation/resources/tools/understand_codebase/test_understand_codebase.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AF_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
RPU_ROOT="$AF_ROOT/../RichPythonUtils"
OS_ROOT="$AF_ROOT/../OpenStartup"

export PYTHONPATH="$AF_ROOT/src:$RPU_ROOT/src:$OS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Auto-detect Python >= 3.11
_find_python() {
    for candidate in "${OPENSTARTUP_PYTHON:-}" \
                     /opt/homebrew/anaconda3/bin/python \
                     python3.13 python3.12 python3.11 python3; do
        [ -z "$candidate" ] && continue
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    echo "python3"
    return 1
}
PYTHON="$(_find_python)"

TARGET="${1:-$AF_ROOT/src/agent_foundation}"
MODEL="${UC_MODEL:-opus[1m]}"
BACKEND="${UC_BACKEND:-}"

RUNTIME_DIR="$AF_ROOT/_runtime/understand_codebase_tests"
mkdir -p "$RUNTIME_DIR"
LOG_FILE="$RUNTIME_DIR/test_uc_$(date +%Y%m%d_%H%M%S).log"

echo "╔══════════════════════════════════════════════╗"
echo "║  Understand Codebase Test                    ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Target: $TARGET"
echo "║  Model:  $MODEL"
echo "║  Python: $PYTHON"
echo "║  Log:    $LOG_FILE"
echo "╚══════════════════════════════════════════════╝"

"$PYTHON" -c "from agent_foundation.resources.tools.understand_codebase.executor import execute; print('Import OK')" || {
    echo "ERROR: Cannot import understand_codebase executor"; exit 1
}

BACKEND_FLAG=""
[ -n "${BACKEND:-}" ] && BACKEND_FLAG="--backend $BACKEND"

"$PYTHON" -m agent_foundation.resources.tools.understand_codebase \
    "$TARGET" \
    --model "$MODEL" \
    $BACKEND_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
