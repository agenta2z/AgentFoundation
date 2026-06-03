#!/usr/bin/env bash
# Research-propose tool test launcher.
#
# Usage:
#   bash test/agent_foundation/resources/tools/task/test_research_propose.sh
#   bash test/agent_foundation/resources/tools/task/test_research_propose.sh "custom request"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AF_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
RPU_ROOT="$AF_ROOT/../RichPythonUtils"
OS_ROOT="$AF_ROOT/../OpenStartup"

export PYTHONPATH="$AF_ROOT/src:$RPU_ROOT/src:$OS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

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

REQUEST="${1:-Investigate opportunities for improvement}"

RUNTIME_DIR="$AF_ROOT/_runtime/research_propose_tests"
mkdir -p "$RUNTIME_DIR"
LOG_FILE="$RUNTIME_DIR/test_rp_$(date +%Y%m%d_%H%M%S).log"

echo "╔══════════════════════════════════════════════════╗"
echo "║  Research-Propose Tool Test                      ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Request: ${REQUEST:0:45}...                     "
echo "║  Log:     $LOG_FILE                              "
echo "║  Python:  $PYTHON                                "
echo "╚══════════════════════════════════════════════════╝"

exec "$PYTHON" -u -c "
import asyncio, json, sys
from pathlib import Path

async def main():
    from agent_foundation.resources.tools.task.executor import execute

    # Load tool.json to get defaults (same as derived_tool_execute would)
    tool_json = Path('$AF_ROOT/src/agent_foundation/resources/tools/research_propose/tool.json')
    tool = json.loads(tool_json.read_text())
    defaults = tool['derived_from']['defaults']

    # Build arguments with all defaults applied
    arguments = {
        'request': '''$REQUEST''',
    }
    for k, v in defaults.items():
        arguments.setdefault(k, v)

    print(f'Config: {arguments.get(\"config\")}')
    print(f'Template master: {arguments.get(\"template_master_version\")}')
    print(f'Tool name: {tool[\"name\"]}')
    print('---')
    sys.stdout.flush()

    # Call execute() with tool_name in session_context for correct workspace naming
    result = await execute(arguments, {'tool_name': tool['name']})

    output = result.result if hasattr(result, 'result') else str(result)
    print('=== RESULT ===')
    print(output[:3000] if len(output) > 3000 else output)

asyncio.run(main())
" 2>&1 | tee "$LOG_FILE"
