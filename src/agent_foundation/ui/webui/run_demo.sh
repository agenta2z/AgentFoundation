#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# CoScience Chatbot Demo - Unified Launch Script
#
# This script handles everything needed to run the demo:
# 1. Sets Meta proxy for yarn install
# 2. Pre-builds React frontend (production mode)
# 3. Launches FastAPI backend via buck2 with SSL support
#
# Usage:
#   ./run_demo.sh [--dev] [--debug] [--iteration N] [--step N] [--speedup N]
#
# Options:
#   --dev          Development mode: skip React build, enable hot reload
#                  Run React dev server separately: cd react && yarn start
#   --debug        Enable debug logging (shows timing details for animations)
#   --iteration N  Start from specific iteration (1-based, e.g., --iteration 1)
#   --step N       Start from specific step within iteration (0-based, e.g., --step 3)
#   --speedup N    Speed up all animation delays by factor N (e.g., --speedup 2 = 2x faster)
#
# Examples:
#   ./run_demo.sh                          # Production: build React + serve
#   ./run_demo.sh --dev                    # Development: skip build, hot reload
#   ./run_demo.sh --debug                  # Production with debug logging
#   ./run_demo.sh --dev --debug            # Development with debug logging
#   ./run_demo.sh --iteration 1 --step 3   # Start at iteration 1, step 3
#   ./run_demo.sh --speedup 5              # 5x faster animations (for debugging)
#   ./run_demo.sh --debug --speedup 3 --iteration 1 --step 2  # Combined options
#   PORT=8088 ./run_demo.sh                # Custom port

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to fbcode root (required for buck2)
FBCODE_ROOT="${SCRIPT_DIR%/fbcode/*}/fbcode"
cd "$FBCODE_ROOT"

# Default port
PORT="${PORT:-8087}"

# Get hostname for SSL certificates
HOSTNAME="${HOSTNAME:-$(hostname -f)}"

# SSL certificate paths (devserver standard locations)
CERT_FILE="/etc/pki/tls/certs/${HOSTNAME}.crt"
KEY_FILE="/etc/pki/tls/certs/${HOSTNAME}.key"

# Meta proxy settings for yarn install
export HTTP_PROXY="http://fwdproxy:8080"
export HTTPS_PROXY="http://fwdproxy:8080"
export NO_PROXY="localhost,127.0.0.1,.facebook.com,.fbcdn.net,.fb.com,.intern.facebook.com"

# Parse command line arguments
DEV_MODE=false
DEBUG_MODE=false
ITERATION=""
STEP=""
SPEEDUP=""
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case $arg in
        --dev)
            DEV_MODE=true
            ;;
        --debug)
            DEBUG_MODE=true
            ;;
        --iteration)
            i=$((i + 1))
            ITERATION="${!i}"
            ;;
        --step)
            i=$((i + 1))
            STEP="${!i}"
            ;;
        --speedup)
            i=$((i + 1))
            SPEEDUP="${!i}"
            ;;
    esac
    i=$((i + 1))
done

echo "============================================"
echo "  CoScience Chatbot Demo"
echo "============================================"
echo ""

# Step 1: Build React frontend (unless in dev mode)
if ! $DEV_MODE; then
    echo "=== Step 1: Building React frontend ==="
    cd "$SCRIPT_DIR/react"

    if [[ ! -d "node_modules" ]]; then
        echo "Installing dependencies (using Meta proxy)..."
        yarn install
    fi

    # Clean old builds to avoid serving stale cached files
    echo "Cleaning old builds..."
    rm -rf build
    rm -rf "$SCRIPT_DIR/frontend"/*

    echo "Building production bundle..."
    yarn build

    # Copy build to frontend directory
    echo "Copying build to frontend directory..."
    mkdir -p "$SCRIPT_DIR/frontend"
    cp -r build/* "$SCRIPT_DIR/frontend/"

    cd "$FBCODE_ROOT"
    echo "✓ React build complete!"
    echo ""
else
    echo "=== Step 1: Skipping React build (dev mode) ==="
    echo ""
fi

# Step 2: Launch backend via buck2
echo "=== Step 2: Starting FastAPI backend via buck2 ==="

if $DEV_MODE; then
    echo ""
    echo "Mode: DEVELOPMENT (skip React build)"
    echo "Host: ::"
    echo "Port: $PORT"
    echo ""
    echo "To run React dev server separately:"
    echo "  cd $SCRIPT_DIR/react && yarn start"
    echo ""
fi

# Check for SSL certificates
if [[ -f "$CERT_FILE" ]] && [[ -f "$KEY_FILE" ]]; then
    echo "SSL: Enabled"
    if $DEBUG_MODE; then
        echo "Logging: DEBUG (verbose)"
    fi
    if [[ -n "$ITERATION" ]]; then
        echo "Starting iteration: $ITERATION"
    fi
    if [[ -n "$STEP" ]]; then
        echo "Starting step: $STEP"
    fi
    if [[ -n "$SPEEDUP" ]]; then
        echo "Speedup: ${SPEEDUP}x faster"
    fi
    echo ""
    echo "============================================"
    echo "  Demo URL: https://${HOSTNAME}:${PORT}"
    echo "============================================"
    echo ""

    # Build command arguments
    BACKEND_ARGS="--port $PORT --host :: --ssl-keyfile $KEY_FILE --ssl-certfile $CERT_FILE"
    if $DEBUG_MODE; then
        BACKEND_ARGS="$BACKEND_ARGS --debug"
    fi
    if [[ -n "$ITERATION" ]]; then
        BACKEND_ARGS="$BACKEND_ARGS --iteration $ITERATION"
    fi
    if [[ -n "$STEP" ]]; then
        BACKEND_ARGS="$BACKEND_ARGS --step $STEP"
    fi
    if [[ -n "$SPEEDUP" ]]; then
        BACKEND_ARGS="$BACKEND_ARGS --speedup $SPEEDUP"
    fi

    buck2 run //rankevolve/src/webui:run_webui -- --mode demo $BACKEND_ARGS
else
    echo "SSL: Disabled (certificates not found)"
    echo "  Expected: $CERT_FILE"
    echo "  Expected: $KEY_FILE"
    if $DEBUG_MODE; then
        echo "Logging: DEBUG (verbose)"
    fi
    if [[ -n "$ITERATION" ]]; then
        echo "Starting iteration: $ITERATION"
    fi
    if [[ -n "$STEP" ]]; then
        echo "Starting step: $STEP"
    fi
    if [[ -n "$SPEEDUP" ]]; then
        echo "Speedup: ${SPEEDUP}x faster"
    fi
    echo ""
    echo "============================================"
    echo "  Demo URL: http://${HOSTNAME}:${PORT}"
    echo "============================================"
    echo ""

    # Build command arguments
    BACKEND_ARGS="--port $PORT --host ::"
    if $DEBUG_MODE; then
        BACKEND_ARGS="$BACKEND_ARGS --debug"
    fi
    if [[ -n "$ITERATION" ]]; then
        BACKEND_ARGS="$BACKEND_ARGS --iteration $ITERATION"
    fi
    if [[ -n "$STEP" ]]; then
        BACKEND_ARGS="$BACKEND_ARGS --step $STEP"
    fi
    if [[ -n "$SPEEDUP" ]]; then
        BACKEND_ARGS="$BACKEND_ARGS --speedup $SPEEDUP"
    fi

    buck2 run //rankevolve/src/webui:run_webui -- --mode demo $BACKEND_ARGS
fi
