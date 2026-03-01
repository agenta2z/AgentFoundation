#!/usr/bin/env python3
"""Minimal test script for ClaudeCodeInferencer and DevmateSDKInferencer.

This script tests:
1. Module imports work correctly
2. Classes can be instantiated
3. Helper methods work
4. Actual SDK calls (if SDK is available)

Run with: buck2 run or within bento console.
"""

import asyncio
import sys
import traceback


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ============================================================
# Test 1: Import Tests
# ============================================================


def test_imports():
    """Test all imports work correctly."""
    print_section("TEST 1: Import Tests")

    errors = []

    # Test 1.1: Import async helper
    try:
        from rich_python_utils.common_utils.async_function_helper import (
            _run_async,
            async_execute_with_retry,
        )

        print("✓ async_function_helper imports work")
    except Exception as e:
        print(f"✗ async_function_helper import failed: {e}")
        errors.append(str(e))

    # Test 1.2: Import SDK types
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
            SDKInferencerResponse,
        )

        print("✓ sdk_types imports work")
    except Exception as e:
        print(f"✗ sdk_types import failed: {e}")
        errors.append(str(e))

    # Test 1.3: Import ClaudeCodeInferencer
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )

        print("✓ ClaudeCodeInferencer imports work")
    except Exception as e:
        print(f"✗ ClaudeCodeInferencer import failed: {e}")
        errors.append(str(e))

    # Test 1.4: Import DevmateSDKInferencer
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateSDKInferencer,
        )

        print("✓ DevmateSDKInferencer imports work")
    except Exception as e:
        print(f"✗ DevmateSDKInferencer import failed: {e}")
        errors.append(str(e))

    # Test 1.5: Lazy import from package
    try:
        from agent_foundation.common.inferencers.agentic_inferencers import (
            ClaudeCodeInferencer,
            DevmateSDKInferencer,
            SDKInferencerResponse,
        )

        print("✓ Lazy package imports work")
    except Exception as e:
        print(f"✗ Lazy package import failed: {e}")
        errors.append(str(e))

    return len(errors) == 0


# ============================================================
# Test 2: Initialization Tests
# ============================================================


def test_initialization():
    """Test class initialization."""
    print_section("TEST 2: Initialization Tests")

    errors = []

    # Test 2.1: ClaudeCodeInferencer initialization
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )

        inferencer = ClaudeCodeInferencer(
            root_folder="/tmp",
            system_prompt="Test",
            allowed_tools=["Read"],
            idle_timeout_seconds=60,
        )

        assert inferencer.root_folder == "/tmp"
        assert inferencer.system_prompt == "Test"
        assert inferencer.allowed_tools == ["Read"]
        assert inferencer.idle_timeout_seconds == 60

        print("✓ ClaudeCodeInferencer initialization works")
        print(f"  - root_folder: {inferencer.root_folder}")
        print(f"  - allowed_tools: {inferencer.allowed_tools}")
    except Exception as e:
        print(f"✗ ClaudeCodeInferencer init failed: {e}")
        errors.append(str(e))

    # Test 2.2: DevmateSDKInferencer initialization
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateSDKInferencer,
        )

        inferencer = DevmateSDKInferencer(
            root_folder="/tmp",
            usecase="test_usecase",
            config_vars={"key": "value"},
            total_timeout_seconds=120,
        )

        assert inferencer.root_folder == "/tmp"
        assert inferencer.usecase == "test_usecase"
        assert inferencer.config_vars == {"key": "value"}
        assert inferencer.total_timeout_seconds == 120

        print("✓ DevmateSDKInferencer initialization works")
        print(f"  - root_folder: {inferencer.root_folder}")
        print(f"  - usecase: {inferencer.usecase}")
    except Exception as e:
        print(f"✗ DevmateSDKInferencer init failed: {e}")
        errors.append(str(e))

    # Test 2.3: SDKInferencerResponse initialization
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
            SDKInferencerResponse,
        )

        response = SDKInferencerResponse(
            content="Hello",
            session_id="sess_123",
            tool_uses=3,
            tokens_received=42,
        )

        assert response.content == "Hello"
        assert response.session_id == "sess_123"
        assert response.tool_uses == 3
        assert response.tokens_received == 42
        assert str(response) == "Hello"

        print("✓ SDKInferencerResponse initialization works")
    except Exception as e:
        print(f"✗ SDKInferencerResponse init failed: {e}")
        errors.append(str(e))

    return len(errors) == 0


# ============================================================
# Test 3: Method Tests
# ============================================================


def test_methods():
    """Test helper methods."""
    print_section("TEST 3: Method Tests")

    errors = []

    # Test 3.1: _extract_prompt for ClaudeCodeInferencer
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )

        inferencer = ClaudeCodeInferencer()

        # String input
        assert inferencer._extract_prompt("hello") == "hello"
        # Dict with prompt key
        assert inferencer._extract_prompt({"prompt": "world"}) == "world"
        # Dict without prompt key
        result = inferencer._extract_prompt({"other": "value"})
        assert "other" in result

        print("✓ ClaudeCodeInferencer._extract_prompt works")
    except Exception as e:
        print(f"✗ ClaudeCodeInferencer._extract_prompt failed: {e}")
        errors.append(str(e))

    # Test 3.2: _extract_prompt for DevmateSDKInferencer
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateSDKInferencer,
        )

        inferencer = DevmateSDKInferencer()

        # String input
        assert inferencer._extract_prompt("hello") == "hello"
        # Dict with prompt key
        assert inferencer._extract_prompt({"prompt": "world"}) == "world"

        print("✓ DevmateSDKInferencer._extract_prompt works")
    except Exception as e:
        print(f"✗ DevmateSDKInferencer._extract_prompt failed: {e}")
        errors.append(str(e))

    return len(errors) == 0


# ============================================================
# Test 4: Async Infrastructure Tests
# ============================================================


def test_async_infrastructure():
    """Test async methods exist and are callable."""
    print_section("TEST 4: Async Infrastructure Tests")

    errors = []

    # Test 4.1: Check async methods exist on ClaudeCodeInferencer
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )

        inferencer = ClaudeCodeInferencer()

        assert hasattr(inferencer, "ainfer")
        assert hasattr(inferencer, "aconnect")
        assert hasattr(inferencer, "adisconnect")
        assert hasattr(inferencer, "_ainfer")
        assert hasattr(inferencer, "__aenter__")
        assert hasattr(inferencer, "__aexit__")

        print("✓ ClaudeCodeInferencer has all async methods")
    except Exception as e:
        print(f"✗ ClaudeCodeInferencer async methods check failed: {e}")
        errors.append(str(e))

    # Test 4.2: Check async methods exist on DevmateSDKInferencer
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateSDKInferencer,
        )

        inferencer = DevmateSDKInferencer()

        assert hasattr(inferencer, "ainfer")
        assert hasattr(inferencer, "_ainfer")

        print("✓ DevmateSDKInferencer has all async methods")
    except Exception as e:
        print(f"✗ DevmateSDKInferencer async methods check failed: {e}")
        errors.append(str(e))

    # Test 4.3: Test async_execute_with_retry
    try:
        from rich_python_utils.common_utils.async_function_helper import (
            async_execute_with_retry,
        )

        async def simple_func():
            return "success"

        result = asyncio.run(async_execute_with_retry(simple_func, max_retry=1))
        assert result == "success"

        print("✓ async_execute_with_retry works")
    except Exception as e:
        print(f"✗ async_execute_with_retry failed: {e}")
        errors.append(str(e))

    return len(errors) == 0


# ============================================================
# Test 5: Real SDK Call (Optional)
# ============================================================


def test_real_sdk_call_claude():
    """Test real Claude Code SDK call (optional - will skip if SDK unavailable)."""
    print_section("TEST 5: Real Claude Code SDK Call (Optional)")

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )

        # Check if SDK is available
        try:
            from claude_agent_sdk import ClaudeSDKClient

            sdk_available = True
        except ImportError:
            sdk_available = False
            print("⚠ Claude Agent SDK not available - skipping real SDK test")
            return True  # Not a failure, just skipped

        if sdk_available:
            inferencer = ClaudeCodeInferencer(
                root_folder="/tmp",
                allowed_tools=[],
                idle_timeout_seconds=60,
            )

            print("Sending query: 'What is 2+2? Just the number.'")

            async def run_query():
                return await inferencer.ainfer(
                    "What is 2+2? Just give the number, nothing else."
                )

            response = asyncio.run(run_query())

            print(f"✓ Got response: {str(response)[:100]}...")
            return True

    except Exception as e:
        print(f"✗ Real SDK call failed: {e}")
        traceback.print_exc()
        return False


def test_real_sdk_call_devmate():
    """Test real Devmate SDK call (optional - will skip if SDK unavailable)."""
    print_section("TEST 6: Real Devmate SDK Call (Optional)")

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateSDKInferencer,
        )

        # Check if SDK is available
        try:
            from devai.devmate_sdk.python.devmate_client import DevmateSDKClient

            sdk_available = True
        except ImportError:
            sdk_available = False
            print("⚠ Devmate SDK not available - skipping real SDK test")
            return True  # Not a failure, just skipped

        if sdk_available:
            inferencer = DevmateSDKInferencer(
                root_folder="/tmp",
                total_timeout_seconds=60,
            )

            print("Sending query: 'What is 2+2? Just the number.'")

            async def run_query():
                return await inferencer.ainfer(
                    "What is 2+2? Just give the number, nothing else."
                )

            response = asyncio.run(run_query())

            print(f"✓ Got response: {str(response)[:100]}...")
            return True

    except Exception as e:
        print(f"✗ Real SDK call failed: {e}")
        traceback.print_exc()
        return False


# ============================================================
# Main
# ============================================================


def main():
    """Run all tests."""
    print("=" * 60)
    print("  INFERENCER IMPLEMENTATION TESTS")
    print("=" * 60)

    results = {
        "Import Tests": test_imports(),
        "Initialization Tests": test_initialization(),
        "Method Tests": test_methods(),
        "Async Infrastructure Tests": test_async_infrastructure(),
        "Real Claude SDK Call": test_real_sdk_call_claude(),
        "Real Devmate SDK Call": test_real_sdk_call_devmate(),
    }

    print_section("FINAL SUMMARY")

    passed = 0
    failed = 0
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n  Total: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
