#!/usr/bin/env python3
"""
Live integration test for DevmateInferencer.

This script actually executes DevMate commands and displays the results.
It can be run manually to verify DevMate integration is working correctly.

Usage:
    # Run with default "Hello, how are you?" prompt
    python test_devmate_inferencer_live.py

    # Run with custom prompt
    python test_devmate_inferencer_live.py "What is 2+2?"

    # Run with custom prompt and model
    python test_devmate_inferencer_live.py "Explain Python decorators" --model claude-sonnet-4.5

Note: This requires DevMate to be installed and properly configured.
"""

import argparse
import sys
import time

import resolve_path  # noqa: F401

from agent_foundation.common.inferencers.terminal_inferencers.devmate.devmate_inferencer import (
    DevmateInferencer,
)


def run_devmate_test(
    prompt: str,
    model_name: str = "claude-sonnet-4.5",
    max_tokens: int = 32768,
    timeout: int = 300,
    verbose: bool = True,
) -> dict:
    """
    Run a single DevMate test with the given prompt.

    Args:
        prompt: The prompt to send to DevMate.
        model_name: The model to use for inference.
        max_tokens: Maximum tokens for the response.
        timeout: Timeout in seconds for the command.
        verbose: Whether to print detailed output.

    Returns:
        The result dictionary from DevMate inference.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("=== DevMate Live Integration Test ===")
        print("=" * 60)
        print(f'Prompt: "{prompt}"')
        print(f"Model: {model_name}")
        print(f"Max tokens: {max_tokens}")
        print(f"Timeout: {timeout}s")
        print()

    inferencer = DevmateInferencer(
        model_name=model_name,
        max_tokens=max_tokens,
        timeout=timeout,
        no_create_commit=True,
    )

    command = inferencer.construct_command(prompt)
    command_str = " ".join(command)

    if verbose:
        print("Executing DevMate...")
        print(f"Command: {command_str}")
        print()

    start_time = time.time()
    result = inferencer.infer(prompt)
    execution_time = time.time() - start_time

    if verbose:
        print("Response:")
        print("-" * 60)
        response_text = inferencer.get_response_text(result)
        print(response_text)
        print("-" * 60)
        print()
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Return code: {result['return_code']}")
        print(f"Success: {result['success']}")

        if result.get("stderr"):
            print(f"Stderr: {result['stderr']}")

        if not result["success"] and result.get("error"):
            print(f"Error: {result['error']}")

    result["execution_time"] = execution_time
    return result


def run_simple_greeting_test() -> bool:
    """
    Run a simple greeting test with DevMate.

    Returns:
        True if the test passed, False otherwise.
    """
    print("\n" + "=" * 60)
    print("Test 1: Simple Greeting")
    print("=" * 60)

    result = run_devmate_test(
        prompt="Hello, how are you?",
        verbose=True,
    )

    return result["success"]


def run_math_test() -> bool:
    """
    Run a simple math test with DevMate.

    Returns:
        True if the test passed, False otherwise.
    """
    print("\n" + "=" * 60)
    print("Test 2: Simple Math Question")
    print("=" * 60)

    result = run_devmate_test(
        prompt="What is 2+2? Answer with just the number.",
        verbose=True,
    )

    if result["success"]:
        response = result.get("output", "")
        if "4" in response:
            print("✓ Math test passed - response contains '4'")
            return True
        else:
            print("✗ Math test failed - response does not contain '4'")
            return False

    return False


def run_code_question_test() -> bool:
    """
    Run a code-related question test with DevMate.

    Returns:
        True if the test passed, False otherwise.
    """
    print("\n" + "=" * 60)
    print("Test 3: Code Question")
    print("=" * 60)

    result = run_devmate_test(
        prompt="Write a Python one-liner to print 'Hello World'",
        max_tokens=256,
        verbose=True,
    )

    if result["success"]:
        response = result.get("output", "").lower()
        if "print" in response and "hello" in response:
            print("✓ Code test passed - response contains print and hello")
            return True
        else:
            print("✗ Code test failed - response does not contain expected code")
            return False

    return False


def print_command_only(
    prompt: str,
    model_name: str = "claude-sonnet-4.5",
    max_tokens: int = 32768,
) -> None:
    """
    Print the command that would be executed without actually running it.

    Args:
        prompt: The prompt to send to DevMate.
        model_name: The model to use for inference.
        max_tokens: Maximum tokens for the response.
    """
    inferencer = DevmateInferencer(
        model_name=model_name,
        max_tokens=max_tokens,
        no_create_commit=True,
    )

    command = inferencer.construct_command(prompt)
    command_str = " ".join(command)

    print("\n" + "=" * 60)
    print("Command Preview (not executed)")
    print("=" * 60)
    print(f'Prompt: "{prompt}"')
    print(f"Model: {model_name}")
    print(f"Max tokens: {max_tokens}")
    print()
    print("Generated command:")
    print(command_str)
    print()


def interactive_mode():
    """Run in interactive mode, allowing user to enter prompts."""
    print("\n" + "=" * 60)
    print("DevMate Interactive Mode")
    print("=" * 60)
    print("Enter prompts to send to DevMate. Type 'quit' or 'exit' to stop.")
    print()

    while True:
        try:
            prompt = input("Enter prompt: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                print("Exiting interactive mode.")
                break
            if not prompt:
                continue

            run_devmate_test(prompt)
            print()

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except EOFError:
            print("\nExiting interactive mode.")
            break


def main():
    """Main entry point for the live integration test."""
    parser = argparse.ArgumentParser(
        description="Live integration test for DevmateInferencer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default greeting test
    python test_devmate_inferencer_live.py

    # Run with custom prompt
    python test_devmate_inferencer_live.py "What is Python?"

    # Preview command without executing
    python test_devmate_inferencer_live.py "Test prompt" --dry-run

    # Run interactive mode
    python test_devmate_inferencer_live.py --interactive

    # Run all automated tests
    python test_devmate_inferencer_live.py --run-all-tests
        """,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        default="Hello, how are you?",
        help="The prompt to send to DevMate (default: 'Hello, how are you?')",
    )

    parser.add_argument(
        "--model",
        "-m",
        default="claude-sonnet-4.5",
        help="Model to use for inference (default: claude-sonnet-4.5)",
    )

    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=32768,
        help="Maximum tokens for response (default: 32768)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)",
    )

    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Print the command without executing it",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode",
    )

    parser.add_argument(
        "--run-all-tests",
        "-a",
        action="store_true",
        help="Run all automated tests",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output (just success/failure)",
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if args.dry_run:
        print_command_only(args.prompt, args.model, args.max_tokens)
        return

    if args.run_all_tests:
        print("\n" + "=" * 60)
        print("Running All Automated Tests")
        print("=" * 60)

        results = []

        try:
            results.append(("Simple Greeting", run_simple_greeting_test()))
        except Exception as e:
            print(f"Error in greeting test: {e}")
            results.append(("Simple Greeting", False))

        try:
            results.append(("Math Question", run_math_test()))
        except Exception as e:
            print(f"Error in math test: {e}")
            results.append(("Math Question", False))

        try:
            results.append(("Code Question", run_code_question_test()))
        except Exception as e:
            print(f"Error in code test: {e}")
            results.append(("Code Question", False))

        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        passed = sum(1 for _, success in results if success)
        total = len(results)

        for name, success in results:
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"{status}: {name}")

        print()
        print(f"Total: {passed}/{total} tests passed")

        if passed == total:
            print("All tests passed!")
            sys.exit(0)
        else:
            print("Some tests failed!")
            sys.exit(1)

        return

    result = run_devmate_test(
        prompt=args.prompt,
        model_name=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        verbose=not args.quiet,
    )

    if args.quiet:
        if result["success"]:
            print("SUCCESS")
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
