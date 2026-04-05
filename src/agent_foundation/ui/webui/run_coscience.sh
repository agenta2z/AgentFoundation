#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Run script for CoScience Chatbot Demo (React + FastAPI)
#
# Usage:
#   ./run_coscience.sh [--dev]
#
# Options:
#   --dev    Run in development mode (with hot reload)
#
# In production mode:
#   - Uses SSL certificates from devserver
#   - Binds to IPv6 address (::)
#   - Serves React frontend from FastAPI
#
# In development mode:
#   - Runs uvicorn with hot reload
#   - No SSL (use React dev server separately)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default port
PORT="${PORT:-8087}"

# Get hostname for SSL certificates
HOSTNAME="${HOSTNAME:-$(hostname -f)}"

# SSL certificate paths (devserver standard locations)
CERT_FILE="/etc/pki/tls/certs/${HOSTNAME}.crt"
KEY_FILE="/etc/pki/tls/certs/${HOSTNAME}.key"

# Check for development mode
DEV_MODE=false
if [[ "$1" == "--dev" ]]; then
    DEV_MODE=true
fi

# Add chatbot_demo to Python path (for importing experiment_engine)
export PYTHONPATH="${SCRIPT_DIR}/../chatbot_demo:${PYTHONPATH}"

if $DEV_MODE; then
    echo "Starting FastAPI backend in development mode..."
    echo "Port: $PORT"
    echo ""
    echo "Run React dev server separately with:"
    echo "  cd react && yarn start"
    echo ""

    # Development mode: no SSL, with hot reload
    python -m uvicorn backend.main:app \
        --host "127.0.0.1" \
        --port "$PORT" \
        --reload \
        --log-level info
else
    echo "Starting CoScience Chatbot Demo (React + FastAPI)..."
    echo ""
    echo "Port: $PORT"
    echo "Host: ::"
    echo "SSL Cert: $CERT_FILE"
    echo "SSL Key: $KEY_FILE"
    echo ""

    # Check if SSL certificates exist
    if [[ ! -f "$CERT_FILE" ]] || [[ ! -f "$KEY_FILE" ]]; then
        echo "Warning: SSL certificates not found. Running without SSL..."
        echo ""
        python -m uvicorn backend.main:app \
            --host "::" \
            --port "$PORT" \
            --log-level info
    else
        # Production mode: SSL + IPv6
        python -m uvicorn backend.main:app \
            --host "::" \
            --port "$PORT" \
            --ssl-keyfile "$KEY_FILE" \
            --ssl-certfile "$CERT_FILE" \
            --log-level info
    fi
fi
