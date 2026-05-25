#!/usr/bin/env python3
"""CI guardrail: detect duplicate widget/component files in webui/react/ that
should be re-exports from react-shared/.

Fails the build if any file in webui/react/src/components/{widgets,common}/
has the same name as a file in react-shared/src/ and is NOT a one-line re-export.

Usage:
    python scripts/check_no_duplicate_widgets.py
"""
from pathlib import Path
import re
import sys

UI_ROOT = Path(__file__).resolve().parent.parent / "src" / "agent_foundation" / "ui"
SHARED = UI_ROOT / "react-shared" / "src"
WEBUI = UI_ROOT / "webui" / "react" / "src"

RE_EXPORT_PATTERN = re.compile(
    r"^export\s+\{[^}]+\}\s+from\s+['\"]@agent-foundation/shared-ui",
    re.MULTILINE,
)

SCAN_DIRS = ["components/widgets", "components/common", "components/chat",
             "components/layout", "components/progress"]


def collect_shared_names() -> set[str]:
    names = set()
    for d in ["common", "inputs", "chat", "layout", "progress", "protocol"]:
        p = SHARED / d
        if p.is_dir():
            for f in p.glob("*.js"):
                if f.name != "index.js":
                    names.add(f.name)
    return names


def main() -> int:
    shared_names = collect_shared_names()
    if not shared_names:
        print("WARNING: No shared components found — is react-shared/ populated?")
        return 0

    violations = []
    for scan_dir in SCAN_DIRS:
        webui_dir = WEBUI / scan_dir
        if not webui_dir.is_dir():
            continue
        for f in webui_dir.glob("*.js"):
            if f.name in shared_names and f.name != "index.js":
                content = f.read_text()
                if not RE_EXPORT_PATTERN.search(content):
                    violations.append(f"  {f.relative_to(UI_ROOT)}")

    if violations:
        print(f"ERROR: {len(violations)} file(s) duplicate react-shared/ components")
        print("       without re-exporting from @agent-foundation/shared-ui:")
        for v in violations:
            print(v)
        print("\nFix: replace with a one-line re-export or delete the file.")
        return 1

    print(f"OK: {len(shared_names)} shared components checked, no duplicates found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
