#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# RankEvolve WebUI — Install frontend dependencies.
#
# Run this once after cloning or when sharing the webui without node_modules.
# It installs all React dependencies defined in react/package.json using yarn.
#
# Usage:
#   ./install_deps.sh            # Install dependencies
#   ./install_deps.sh --clean    # Remove existing node_modules first, then install
#
# Prerequisites:
#   - yarn available on PATH
#   - Network access (auto-detects Meta proxy; works on public machines too)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REACT_DIR="$SCRIPT_DIR/react"

# Parse arguments
CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=true ;;
        --help|-h)
            echo "Usage: $0 [--clean]"
            echo "  --clean  Remove existing node_modules before installing"
            exit 0
            ;;
    esac
done

# Verify react/package.json exists
if [[ ! -f "$REACT_DIR/package.json" ]]; then
    echo "ERROR: react/package.json not found at $REACT_DIR"
    echo "Are you running this from the webui directory?"
    exit 1
fi

# Warn if yarn.lock is missing (dependencies won't be pinned)
if [[ ! -f "$REACT_DIR/yarn.lock" ]]; then
    echo "WARNING: react/yarn.lock not found. Dependencies will not be version-pinned."
fi

# Require yarn (npm install ignores yarn.lock, producing a different dependency tree)
if ! command -v yarn &>/dev/null; then
    echo "ERROR: yarn is required but not found on PATH."
    echo "This project uses yarn.lock for dependency pinning. npm is not supported."
    exit 1
fi

# Proxy setup: Meta devservers need fwdproxy to reach external registries.
# Auto-detect by checking if fwdproxy resolves; skip on public machines.
if [[ -z "$HTTP_PROXY" ]] && getent hosts fwdproxy &>/dev/null; then
    echo "Detected Meta network, setting fwdproxy..."
    export HTTP_PROXY="http://fwdproxy:8080"
    export HTTPS_PROXY="http://fwdproxy:8080"
    export NO_PROXY="localhost,127.0.0.1,.facebook.com,.fbcdn.net,.fb.com,.intern.facebook.com"
fi

cd "$REACT_DIR"

if $CLEAN; then
    echo "Removing existing node_modules..."
    rm -rf node_modules
fi

echo "============================================"
echo "  Installing React frontend dependencies"
echo "============================================"
echo ""
echo "Directory: $REACT_DIR"
echo ""

yarn install

echo ""
echo "Done! Installed $(ls node_modules | wc -l) packages."
