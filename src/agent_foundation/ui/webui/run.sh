#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# RankEvolve — Launch both the backend server and the frontend WebUI.
#
# This is the main entry point. It starts the RankEvolve server as a
# background process, waits for it to print its queue root path, then
# launches the WebUI thin client pointing at that queue.
#
# If --queue-root is provided, skips server launch and connects to an
# already-running server.
#
# Usage:
#   ./run.sh --target-path /path/to/codebase                  # Launch everything
#   ./run.sh --target-path /path --model claude-sonnet-4.5     # Custom model
#   ./run.sh --queue-root <path>                               # Connect to existing server
#   ./run.sh --target-path /path --dev                         # Skip React build
#   PORT=8088 ./run.sh --target-path /path                     # Custom port

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────
PORT="${PORT:-8087}"
MODEL="claude-opus-4.6"
PROVIDER=""
TARGET_PATH=""
PIPELINE=""
ENABLE_KNOWLEDGE=false
DEBUG_MODE=false
DEV_MODE=false
REBUILD=false
QUEUE_ROOT=""
SERVICE_ROOT=""
SERVER_DIR=""
LOCAL_BUILD=false

# ── Parse arguments ───────────────────────────────────────────────────
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case $arg in
        --model)
            i=$((i + 1)); MODEL="${!i}" ;;
        --provider)
            i=$((i + 1)); PROVIDER="${!i}" ;;
        --target-path)
            i=$((i + 1)); TARGET_PATH="${!i}" ;;
        --pipeline)
            i=$((i + 1)); PIPELINE="${!i}" ;;
        --enable-knowledge)
            ENABLE_KNOWLEDGE=true ;;
        --debug)
            DEBUG_MODE=true ;;
        --dev)
            DEV_MODE=true ;;
        --rebuild)
            REBUILD=true ;;
        --queue-root)
            i=$((i + 1)); QUEUE_ROOT="${!i}" ;;
        --service-root)
            i=$((i + 1)); SERVICE_ROOT="${!i}" ;;
        --server-dir)
            i=$((i + 1)); SERVER_DIR="${!i}" ;;
        --port)
            i=$((i + 1)); PORT="${!i}" ;;
        --local-build)
            LOCAL_BUILD=true ;;
    esac
    i=$((i + 1))
done

echo "============================================"
echo "  RankEvolve Agent Mode"
echo "============================================"
echo ""

# ── Local build preference ──────────────────────────────────────────
BUCK2_LOCAL_FLAGS=""
if $LOCAL_BUILD; then
    BUCK2_LOCAL_FLAGS="--prefer-local"
    export BUCK2_LOCAL_FLAGS
    echo "Build mode: prefer-local (--local-build)"
fi

# ── Force rebuild if requested ────────────────────────────────────────
if $REBUILD; then
    FBCODE_ROOT="${SCRIPT_DIR%/fbcode/*}/fbcode"
    echo "=== Rebuilding server and WebUI backend (--rebuild) ==="
    cd "$FBCODE_ROOT"
    echo "Cleaning buck2 cache..."
    buck2 clean 2>/dev/null || true
    echo "Building targets..."
    buck2 build $BUCK2_LOCAL_FLAGS //rankevolve/src/server:rankevolve_server //rankevolve/src/webui:run_webui_real
    echo "✓ Rebuild complete!"
    echo ""
    cd "$SCRIPT_DIR"
fi

# ── Cleanup handler ───────────────────────────────────────────────────
SERVER_PID=""

cleanup() {
    echo ""
    echo "Shutting down..."
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Done."
}

trap cleanup EXIT INT TERM

# ── Launch or connect to server ───────────────────────────────────────
if [[ -n "$QUEUE_ROOT" ]]; then
    # Connect to an already-running server
    echo "Connecting to existing server at: $QUEUE_ROOT"
    echo ""
else
    # Build server arguments
    BACKEND_SH_ARGS=""
    if [[ -n "$MODEL" ]]; then BACKEND_SH_ARGS="$BACKEND_SH_ARGS --model $MODEL"; fi
    if [[ -n "$PROVIDER" ]]; then BACKEND_SH_ARGS="$BACKEND_SH_ARGS --provider $PROVIDER"; fi
    if [[ -n "$TARGET_PATH" ]]; then BACKEND_SH_ARGS="$BACKEND_SH_ARGS --target-path $TARGET_PATH"; fi
    if [[ -n "$PIPELINE" ]]; then BACKEND_SH_ARGS="$BACKEND_SH_ARGS --pipeline $PIPELINE"; fi
    if $ENABLE_KNOWLEDGE; then BACKEND_SH_ARGS="$BACKEND_SH_ARGS --enable-knowledge"; fi
    if $DEBUG_MODE; then BACKEND_SH_ARGS="$BACKEND_SH_ARGS --debug"; fi
    if [[ -n "$SERVICE_ROOT" ]]; then BACKEND_SH_ARGS="$BACKEND_SH_ARGS --service-root $SERVICE_ROOT"; fi
    if [[ -n "$SERVER_DIR" ]]; then BACKEND_SH_ARGS="$BACKEND_SH_ARGS --server-dir $SERVER_DIR"; fi

    echo "=== Starting RankEvolve server in background ==="

    # Start server, tee stdout so we can parse QUEUE_ROOT= sentinel
    QUEUE_ROOT_FILE=$(mktemp)
    "$SCRIPT_DIR/run_server.sh" $BACKEND_SH_ARGS 2>&1 | while IFS= read -r line; do
        echo "[server] $line"
        if [[ "$line" == QUEUE_ROOT=* ]]; then
            echo "${line#QUEUE_ROOT=}" > "$QUEUE_ROOT_FILE"
        fi
    done &
    SERVER_PID=$!

    # Wait for the server to print QUEUE_ROOT=<path>
    echo "Waiting for server to start..."
    TIMEOUT=1800
    ELAPSED=0
    while [[ $ELAPSED -lt $TIMEOUT ]]; do
        if [[ -s "$QUEUE_ROOT_FILE" ]]; then
            QUEUE_ROOT=$(cat "$QUEUE_ROOT_FILE")
            break
        fi
        # Also check if server died
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: Server process exited unexpectedly."
            rm -f "$QUEUE_ROOT_FILE"
            exit 1
        fi
        sleep 1
        ELAPSED=$((ELAPSED + 1))
    done

    rm -f "$QUEUE_ROOT_FILE"

    if [[ -z "$QUEUE_ROOT" ]]; then
        echo "ERROR: Timed out waiting for server to start (${TIMEOUT}s)."
        echo "Check server logs above for errors."
        exit 1
    fi

    echo ""
    echo "✓ Server started. Queue root: $QUEUE_ROOT"
    echo ""
fi

# ── Launch WebUI ──────────────────────────────────────────────────────
WEBUI_ARGS="--queue-root $QUEUE_ROOT --port $PORT"
if [[ -n "$MODEL" ]]; then WEBUI_ARGS="$WEBUI_ARGS --model $MODEL"; fi
if [[ -n "$TARGET_PATH" ]]; then WEBUI_ARGS="$WEBUI_ARGS --target-path $TARGET_PATH"; fi
if [[ -n "$PROVIDER" ]]; then WEBUI_ARGS="$WEBUI_ARGS --provider $PROVIDER"; fi
if $DEV_MODE; then WEBUI_ARGS="$WEBUI_ARGS --dev"; fi
if $DEBUG_MODE; then WEBUI_ARGS="$WEBUI_ARGS --debug"; fi

exec "$SCRIPT_DIR/run_webui.sh" $WEBUI_ARGS
