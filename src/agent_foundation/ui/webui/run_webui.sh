#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# RankEvolve WebUI — Launch the WebUI thin client.
#
# Starts the FastAPI server that serves the React frontend and connects
# to the RankEvolve server via file-based queues (AgentServiceBridge).
#
# Requires --queue-root pointing to a running server's queue directory.
# Start the server first with run_server.sh, then pass its queue root here.
#
# Usage:
#   ./run_webui.sh --queue-root _runtime/queues/session_20260317_104036
#   ./run_webui.sh --queue-root <path> --dev      # Skip React build
#   ./run_webui.sh --queue-root <path> --port 9000
#   PORT=8088 ./run_webui.sh --queue-root <path>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to fbcode root (required for buck2)
FBCODE_ROOT="${SCRIPT_DIR%/fbcode/*}/fbcode"
cd "$FBCODE_ROOT"

# Defaults
PORT="${PORT:-8087}"
QUEUE_ROOT=""
DEV_MODE=false
DEBUG_MODE=false
MODEL=""
TARGET_PATH=""
PROVIDER=""

# Get hostname for SSL certificates
HOSTNAME="${HOSTNAME:-$(hostname -f)}"
CERT_FILE="/etc/pki/tls/certs/${HOSTNAME}.crt"
KEY_FILE="/etc/pki/tls/certs/${HOSTNAME}.key"

# Meta proxy settings for yarn install
export HTTP_PROXY="http://fwdproxy:8080"
export HTTPS_PROXY="http://fwdproxy:8080"
export NO_PROXY="localhost,127.0.0.1,.facebook.com,.fbcdn.net,.fb.com,.intern.facebook.com"

# Parse arguments
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case $arg in
        --queue-root)
            i=$((i + 1))
            QUEUE_ROOT="${!i}"
            ;;
        --port)
            i=$((i + 1))
            PORT="${!i}"
            ;;
        --dev)
            DEV_MODE=true
            ;;
        --debug)
            DEBUG_MODE=true
            ;;
        --model)
            i=$((i + 1))
            MODEL="${!i}"
            ;;
        --target-path)
            i=$((i + 1))
            TARGET_PATH="${!i}"
            ;;
        --provider)
            i=$((i + 1))
            PROVIDER="${!i}"
            ;;
    esac
    i=$((i + 1))
done

# Validate required argument
if [[ -z "$QUEUE_ROOT" ]]; then
    echo "ERROR: --queue-root is required."
    echo ""
    echo "Start the server first, then pass its queue root:"
    echo "  ./run_server.sh --target-path /path/to/codebase"
    echo "  # Server prints: QUEUE_ROOT=<path>"
    echo "  ./run_webui.sh --queue-root <path>"
    exit 1
fi

echo "============================================"
echo "  RankEvolve WebUI (Frontend)"
echo "============================================"
echo ""
echo "Queue root: $QUEUE_ROOT"
echo "Port: $PORT"
if $DEV_MODE; then echo "Mode: DEVELOPMENT (skip React build)"; fi
if $DEBUG_MODE; then echo "Debug: Enabled"; fi
echo ""

# Build React frontend (unless in dev mode)
if ! $DEV_MODE; then
    echo "=== Building React frontend ==="
    cd "$SCRIPT_DIR/react"

    if [[ ! -d "node_modules" ]]; then
        echo "Installing dependencies (using Meta proxy)..."
        yarn install
    fi

    echo "Cleaning old builds..."
    rm -rf build
    rm -rf "$SCRIPT_DIR/frontend"/*

    echo "Building production bundle..."
    yarn build

    echo "Copying build to frontend directory..."
    mkdir -p "$SCRIPT_DIR/frontend"
    cp -r build/* "$SCRIPT_DIR/frontend/"

    cd "$FBCODE_ROOT"
    echo "✓ React build complete!"
    echo ""
else
    echo "=== Skipping React build (dev mode) ==="
    echo ""
fi

# Build backend arguments
BACKEND_ARGS="--mode real --port $PORT --host :: --queue-root $QUEUE_ROOT"

if $DEBUG_MODE; then
    BACKEND_ARGS="$BACKEND_ARGS --debug"
fi
if [[ -n "$MODEL" ]]; then
    BACKEND_ARGS="$BACKEND_ARGS --model $MODEL"
fi
if [[ -n "$TARGET_PATH" ]]; then
    BACKEND_ARGS="$BACKEND_ARGS --target-path $TARGET_PATH"
fi
if [[ -n "$PROVIDER" ]]; then
    BACKEND_ARGS="$BACKEND_ARGS --provider $PROVIDER"
fi

# Add SSL if certificates exist
if [[ -f "$CERT_FILE" ]] && [[ -f "$KEY_FILE" ]]; then
    echo "SSL: Enabled"
    BACKEND_ARGS="$BACKEND_ARGS --ssl-keyfile $KEY_FILE --ssl-certfile $CERT_FILE"
    PROTOCOL="https"
else
    echo "SSL: Disabled (certificates not found)"
    PROTOCOL="http"
fi

echo ""
echo "============================================"
echo "  URL: ${PROTOCOL}://${HOSTNAME}:${PORT}"
echo "============================================"
echo ""

buck2 run ${BUCK2_LOCAL_FLAGS:-} //rankevolve/src/webui:run_webui_real -- $BACKEND_ARGS
