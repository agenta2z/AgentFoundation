#!/usr/bin/env bash
# Example command lines for running test_plan_then_implement (E2E plan→implement).
#
# Usage: Copy/paste individual commands — NOT intended to be run as a single script.

set -euo pipefail

ROOT_FOLDER="/data/users/zgchen/fbsource"
REQUEST_FILE="fbcode/rankevolve/test/agentic_foundation/common/inferencers/test_dual_inferencer/request_simple_transformer.txt"

# ==============================================================================
# All ClaudeCode (fastest to test)
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_plan_then_implement -- \
    --request "$REQUEST_FILE" \
    --plan-base-inferencer claude_code \
    --plan-review-inferencer claude_code \
    --impl-base-inferencer claude_code \
    --impl-review-inferencer claude_code \
    --root-folder "$ROOT_FOLDER"

# ==============================================================================
# Mixed: ClaudeCode base + DevmateCLI review (both phases)
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_plan_then_implement -- \
    --request "$REQUEST_FILE" \
    --plan-base-inferencer claude_code \
    --plan-review-inferencer devmate_cli \
    --impl-base-inferencer claude_code \
    --impl-review-inferencer devmate_cli \
    --root-folder "$ROOT_FOLDER"

# ==============================================================================
# Mixed: DevmateCLI base + ClaudeCodeCLI review (both phases) with Opus 4.6
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_plan_then_implement -- \
    --request "$REQUEST_FILE" \
    --plan-base-inferencer devmate_cli \
    --plan-review-inferencer claude_code_cli \
    --impl-base-inferencer devmate_cli \
    --impl-review-inferencer claude_code_cli \
    --model claude-opus-4-6 \
    --root-folder "$ROOT_FOLDER"

# ==============================================================================
# With human approval between plan and implementation
# ==============================================================================
cd "$ROOT_FOLDER" && buck2 run fbcode//rankevolve/test/agentic_foundation:test_plan_then_implement -- \
    --request "$REQUEST_FILE" \
    --plan-base-inferencer claude_code \
    --plan-review-inferencer claude_code \
    --impl-base-inferencer claude_code \
    --impl-review-inferencer claude_code \
    --require-approval \
    --root-folder "$ROOT_FOLDER"
