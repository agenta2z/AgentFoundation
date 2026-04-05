#!/usr/bin/env bash
# Example command lines for running test_retry_idle_timeout (retry + idle timeout doubling).
#
# Usage: Copy/paste individual commands — NOT intended to be run as a single script.

set -euo pipefail

ROOT_FOLDER="/data/users/zgchen/fbsource"
REQUEST_FILE="fbcode/rankevolve/test/agentic_foundation/common/inferencers/test_dual_inferencer/request_simple_transformer.txt"

# ==============================================================================
# DevmateSDK — 3s initial timeout, expect retries (SDK has long pauses)
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_retry_idle_timeout -- \
    --inferencer-type devmate_sdk \
    --request-file "$REQUEST_FILE" \
    --initial-idle-timeout 3 --max-retry 3 \
    --root-folder "$ROOT_FOLDER" --total-timeout 600

# ==============================================================================
# ClaudeCode SDK — 3s initial timeout, expect retries (SDK connect >3s)
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_retry_idle_timeout -- \
    --inferencer-type claude_code \
    --request-file "$REQUEST_FILE" \
    --initial-idle-timeout 3 --max-retry 3 \
    --root-folder "$ROOT_FOLDER" --total-timeout 600

# ==============================================================================
# ClaudeCode CLI — 3s timeout, likely succeeds (CLI streams fast)
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_retry_idle_timeout -- \
    --inferencer-type claude_code_cli \
    --request-file "$REQUEST_FILE" \
    --initial-idle-timeout 3 --max-retry 3 \
    --root-folder "$ROOT_FOLDER" --total-timeout 600

# ==============================================================================
# DevmateCLI — 1s timeout, likely succeeds (CLI streams fast)
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_retry_idle_timeout -- \
    --inferencer-type devmate_cli \
    --request-file "$REQUEST_FILE" \
    --initial-idle-timeout 1 --max-retry 3 \
    --root-folder "$ROOT_FOLDER" --total-timeout 600

# ==============================================================================
# Quick smoke test (short prompt, any inferencer)
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_retry_idle_timeout -- \
    --inferencer-type devmate_sdk \
    --request "What is 2+2? Just the number." \
    --initial-idle-timeout 3 --max-retry 3 \
    --root-folder "$ROOT_FOLDER"
