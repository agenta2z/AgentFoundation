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

REQUEST="${1:-Investigate opportunities for agentic AI in Jira admin workflows}"

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
import asyncio, json, sys, os
from pathlib import Path

async def main():
    from agent_foundation.resources.tools.task.executor import execute

    tool_json = Path('$AF_ROOT/src/agent_foundation/resources/tools/research_propose/tool.json')
    tool = json.loads(tool_json.read_text())
    defaults = tool['derived_from']['defaults']

    arguments = {
        'request': '''$REQUEST''',
        'plan': True,
        'config': defaults.get('config', 'breakdown-multiflow-plan'),
        'template_master_version': defaults.get('template_master_version', 'research_propose'),
        'config_overrides': defaults.get('config_overrides', {}),
    }

    print(f'Config: {arguments[\"config\"]}')
    print(f'Config overrides: {json.dumps(arguments[\"config_overrides\"], indent=2)}')
    print(f'Template master: {arguments[\"template_master_version\"]}')
    print('---')

    result = await execute(arguments, {})
    print(result.result if hasattr(result, 'result') else str(result))

asyncio.run(main())
" 2>&1 | tee "$LOG_FILE"
