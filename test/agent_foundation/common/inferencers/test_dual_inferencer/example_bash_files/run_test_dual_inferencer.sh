#!/usr/bin/env bash
# Example command lines for running test_dual_inferencer (DualInferencer consensus loop).
#
# Usage: Copy/paste individual commands — NOT intended to be run as a single script.

set -euo pipefail

ROOT_FOLDER="/data/users/zgchen/fbsource"
REQUEST_FILE="fbcode/rankevolve/test/agentic_foundation/common/inferencers/test_dual_inferencer/request_simple_transformer.txt"

# ==============================================================================
# ClaudeCode base + DevmateCLI review (plan template)
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_dual_inferencer -- \
    --request-file "$REQUEST_FILE" \
    --template-version plan \
    --base-inferencer claude_code \
    --review-inferencer devmate_cli \
    --root-folder "$ROOT_FOLDER"

# ==============================================================================
# DevmateSDK base + ClaudeCode review
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_dual_inferencer -- \
    --request-file "$REQUEST_FILE" \
    --template-version plan \
    --base-inferencer devmate_sdk \
    --review-inferencer claude_code \
    --root-folder "$ROOT_FOLDER"

# ==============================================================================
# DevmateSDK base + DevmateCLI review
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_dual_inferencer -- \
    --request-file "$REQUEST_FILE" \
    --template-version plan \
    --base-inferencer devmate_sdk \
    --review-inferencer devmate_cli \
    --root-folder "$ROOT_FOLDER"
