#!/usr/bin/env bash
# SOP CLI test launcher — runs role_creation SOP in yolo mode.
#
# Usage:
#   bash test/agent_foundation/resources/tools/sop/test_sop.sh
#   bash test/agent_foundation/resources/tools/sop/test_sop.sh code_optimization "optimize src/lib"
#   SOP_MODEL=sonnet bash test/agent_foundation/resources/tools/sop/test_sop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AF_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
RPU_ROOT="$AF_ROOT/../RichPythonUtils"
OS_ROOT="$AF_ROOT/../OpenStartup"

export PYTHONPATH="$AF_ROOT/src:$RPU_ROOT/src:$OS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${OPENSTARTUP_PYTHON:-python3}"

SOP_NAME="${1:-role_creation}"
REQUEST="${2:-hire a machine learning engineer}"
MODEL="${SOP_MODEL:-opus[1m]}"
EXTRA_SOP_DIRS="$OS_ROOT/src/openteam/server/resources/sops"
EXTRA_TOOL_DIRS="$OS_ROOT/src/openteam/server/resources/tools"

RUNTIME_DIR="$AF_ROOT/_runtime/sop_tests"
mkdir -p "$RUNTIME_DIR"
LOG_FILE="$RUNTIME_DIR/test_sop_$(date +%Y%m%d_%H%M%S).log"

echo "╔══════════════════════════════════════════════╗"
echo "║  SOP CLI Test                                ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  SOP:     $SOP_NAME"
echo "║  Request: $REQUEST"
echo "║  Model:   $MODEL"
echo "║  Python:  $PYTHON"
echo "║  Log:     $LOG_FILE"
echo "╚══════════════════════════════════════════════╝"

"$PYTHON" -c "from agent_foundation.resources.tools.sop.cli import main; print('Import OK')" || {
    echo "ERROR: Cannot import SOP CLI"; exit 1
}

"$PYTHON" -m agent_foundation.resources.tools.sop \
    "$SOP_NAME" \
    --yolo \
    --model "$MODEL" \
    --extra-sop-dirs "$EXTRA_SOP_DIRS" \
    --extra-tool-dirs "$EXTRA_TOOL_DIRS" \
    "$REQUEST" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
