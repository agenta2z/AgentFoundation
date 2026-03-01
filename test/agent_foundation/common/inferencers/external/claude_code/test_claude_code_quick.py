#!/usr/bin/env python3
"""Quick test script for ClaudeCodeInferencer.

This script performs a simple import and basic functionality test
without requiring the full SDK - just verifies the implementation loads correctly.

Can be run with: python3 this_file.py (with PYTHONPATH set appropriately)
Or within a buck test target.
"""

import sys


def test_import_claude_code_inferencer():
    """Test that ClaudeCodeInferencer can be imported."""
    print("TEST: Import ClaudeCodeInferencer")
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )

        print(f"  ✓ Import successful: {ClaudeCodeInferencer}")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_claude_code_inferencer_initialization():
    """Test that ClaudeCodeInferencer can be initialized."""
    print("TEST: Initialize ClaudeCodeInferencer")
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )

        inferencer = ClaudeCodeInferencer(
            root_folder="/tmp",
            system_prompt="Test prompt",
            allowed_tools=["Read"],
            idle_timeout_seconds=60,
        )
        print(f"  ✓ Initialization successful")
        print(f"    - root_folder: {inferencer.root_folder}")
        print(f"    - system_prompt: {inferencer.system_prompt}")
        print(f"    - allowed_tools: {inferencer.allowed_tools}")
        return True
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_extract_prompt():
    """Test _extract_prompt method."""
    print("TEST: _extract_prompt method")
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )

        inferencer = ClaudeCodeInferencer()

        # Test with string
        result1 = inferencer._extract_prompt("Hello world")
        assert result1 == "Hello world", f"Expected 'Hello world', got '{result1}'"

        # Test with dict
        result2 = inferencer._extract_prompt({"prompt": "From dict"})
        assert result2 == "From dict", f"Expected 'From dict', got '{result2}'"

        print("  ✓ _extract_prompt works correctly")
        return True
    except Exception as e:
        print(f"  ✗ _extract_prompt failed: {e}")
        return False


def test_sdk_response_type():
    """Test SDKInferencerResponse type."""
    print("TEST: SDKInferencerResponse type")
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
            SDKInferencerResponse,
        )

        response = SDKInferencerResponse(
            content="Test content",
            session_id="test_session",
            tool_uses=3,
        )

        assert response.content == "Test content"
        assert response.session_id == "test_session"
        assert response.tool_uses == 3
        assert str(response) == "Test content"

        print("  ✓ SDKInferencerResponse works correctly")
        return True
    except Exception as e:
        print(f"  ✗ SDKInferencerResponse failed: {e}")
        return False


def test_lazy_import_from_package():
    """Test lazy import from agentic_inferencers package."""
    print("TEST: Lazy import from package")
    try:
        from agent_foundation.common.inferencers.agentic_inferencers import (
            ClaudeCodeInferencer,
            SDKInferencerResponse,
        )

        print(f"  ✓ Lazy import successful")
        print(f"    - ClaudeCodeInferencer: {ClaudeCodeInferencer}")
        print(f"    - SDKInferencerResponse: {SDKInferencerResponse}")
        return True
    except Exception as e:
        print(f"  ✗ Lazy import failed: {e}")
        return False


def main():
    """Run all quick tests."""
    print("=" * 50)
    print("CLAUDE CODE INFERENCER - QUICK TESTS")
    print("=" * 50)
    print()

    results = []
    results.append(test_import_claude_code_inferencer())
    results.append(test_claude_code_inferencer_initialization())
    results.append(test_extract_prompt())
    results.append(test_sdk_response_type())
    results.append(test_lazy_import_from_package())

    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"  Passed: {passed}/{total}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
