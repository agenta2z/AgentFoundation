#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# RankEvolve Server — Launch the backend agent service.
#
# Starts the RankEvolveAgentService which manages sessions, processes
# messages via file-based queues, and runs DualInferencerBridge tasks.
#
# The server prints QUEUE_ROOT=<path> to stdout when ready.
# Pass the printed path to run_webui.sh --queue-root <path>.
#
# Usage:
#   ./run_server.sh --target-path /path/to/codebase
#   ./run_server.sh --model claude-opus-4.6 --target-path /path
#   ./run_server.sh --service-root /tmp/my_runtime --debug

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to fbcode root (required for buck2)
FBCODE_ROOT="${SCRIPT_DIR%/fbcode/*}/fbcode"
cd "$FBCODE_ROOT"

# Defaults
MODEL="claude-opus-4.6"
PROVIDER=""
TARGET_PATH=""
PIPELINE=""
ENABLE_KNOWLEDGE=false
DEBUG_MODE=false
SERVICE_ROOT=""
SERVER_DIR=""

# Parse arguments
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case $arg in
        --model)
            i=$((i + 1))
            MODEL="${!i}"
            ;;
        --provider)
            i=$((i + 1))
            PROVIDER="${!i}"
            ;;
        --target-path)
            i=$((i + 1))
            TARGET_PATH="${!i}"
            ;;
        --pipeline)
            i=$((i + 1))
            PIPELINE="${!i}"
            ;;
        --enable-knowledge)
            ENABLE_KNOWLEDGE=true
            ;;
        --debug)
            DEBUG_MODE=true
            ;;
        --service-root)
            i=$((i + 1))
            SERVICE_ROOT="${!i}"
            ;;
        --server-dir)
            i=$((i + 1))
            SERVER_DIR="${!i}"
            ;;
    esac
    i=$((i + 1))
done

echo "============================================"
echo "  RankEvolve Server"
echo "============================================"
echo ""
echo "Model: $MODEL"
if [[ -n "$TARGET_PATH" ]]; then echo "Target path: $TARGET_PATH"; fi
if [[ -n "$SERVICE_ROOT" ]]; then echo "Service root: $SERVICE_ROOT"; fi
if [[ -n "$SERVER_DIR" ]]; then echo "Resuming from: $SERVER_DIR"; fi
if $ENABLE_KNOWLEDGE; then echo "Knowledge: Enabled"; fi
if $DEBUG_MODE; then echo "Debug: Enabled"; fi
echo ""

# Build server arguments
SERVER_ARGS="--model $MODEL"

if [[ -n "$PROVIDER" ]]; then
    SERVER_ARGS="$SERVER_ARGS --provider $PROVIDER"
fi
if [[ -n "$TARGET_PATH" ]]; then
    SERVER_ARGS="$SERVER_ARGS --target-path $TARGET_PATH"
fi
if [[ -n "$PIPELINE" ]]; then
    SERVER_ARGS="$SERVER_ARGS --pipeline $PIPELINE"
fi
if $ENABLE_KNOWLEDGE; then
    SERVER_ARGS="$SERVER_ARGS --enable-knowledge"
fi
if $DEBUG_MODE; then
    SERVER_ARGS="$SERVER_ARGS --debug"
fi
if [[ -n "$SERVICE_ROOT" ]]; then
    SERVER_ARGS="$SERVER_ARGS --service-root $SERVICE_ROOT"
fi
if [[ -n "$SERVER_DIR" ]]; then
    SERVER_ARGS="$SERVER_ARGS --server-dir $SERVER_DIR"
fi

echo "Starting server..."
echo ""

buck2 run ${BUCK2_LOCAL_FLAGS:-} //rankevolve/src/server:rankevolve_server -- $SERVER_ARGS
