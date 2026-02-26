#!/usr/bin/env python3
"""
Interactive DevMate Inferencer Example.

This example demonstrates how to use the DevmateInferencer to interact with
DevMate CLI from Python. It provides an interactive terminal interface where
you can input queries and receive responses from DevMate.

Usage:
    # Run with buck2
    buck2 run fbcode//_tony_dev/ScienceModelingTools/examples/agent_foundation/common/inferencers/terminal_inferencers/devmate:devmate_example

    # Or run directly (if dependencies are available)
    python devmate_example.py

    # With a specific prompt (non-interactive)
    buck2 run ... -- "What is Python?"

    # Show command only (dry-run)
    buck2 run ... -- --dry-run "What is Python?"

Features:
    - Interactive mode: Enter queries and get responses in a loop
    - Single query mode: Pass a prompt as argument
    - Dry-run mode: Show the command without executing
    - Configurable model and max tokens
"""

import argparse
import sys
import time

import resolve_path  # noqa: F401

from science_modeling_tools.common.inferencers.terminal_inferencers.devmate.devmate_inferencer import (
    DevmateInferencer,
)


def print_banner():
    """Print a welcome banner."""
    print()
    print("=" * 60)
    print("  DevMate Inferencer - Interactive Example")
    print("=" * 60)
    print()
    print("This example demonstrates using DevmateInferencer to interact")
    print("with the DevMate CLI programmatically.")
    print()


def print_separator():
    """Print a separator line."""
    print("-" * 60)


def run_query(
    inferencer: DevmateInferencer,
    prompt: str,
    verbose: bool = True,
) -> dict:
    """
    Run a single query through DevMate.

    Args:
        inferencer: The DevmateInferencer instance to use.
        prompt: The prompt to send to DevMate.
        verbose: Whether to print detailed output.

    Returns:
        The result dictionary from the inference.
    """
    if verbose:
        print(f"\n📝 Prompt: {prompt}")
        print_separator()
        print("⏳ Sending to DevMate...")

    start_time = time.time()
    result = inferencer.infer(prompt)
    execution_time = time.time() - start_time

    if verbose:
        print()
        if result["success"]:
            print("✅ Response:")
            print_separator()
            print(inferencer.get_response_text(result))
            print_separator()
        else:
            print("❌ Error:")
            print_separator()
            print(result.get("error", "Unknown error"))
            if result.get("stderr"):
                print(f"\nStderr: {result['stderr']}")
            print_separator()

        print(f"\n⏱️  Execution time: {execution_time:.2f}s")
        print(f"📊 Return code: {result['return_code']}")

    result["execution_time"] = execution_time
    return result


def interactive_mode(inferencer: DevmateInferencer):
    """
    Run in interactive mode, allowing user to enter prompts.

    Args:
        inferencer: The DevmateInferencer instance to use.
    """
    print("\n🎯 Interactive Mode")
    print("=" * 60)
    print("Enter your queries below. Commands:")
    print("  - Type your question and press Enter to query DevMate")
    print("  - Type 'quit', 'exit', or 'q' to exit")
    print("  - Type 'help' for this message")
    print("  - Type 'config' to show current configuration")
    print("=" * 60)
    print()

    while True:
        try:
            prompt = input("🤖 You: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye!")
                break

            if prompt.lower() == "help":
                print("\nCommands:")
                print("  - Type your question to query DevMate")
                print("  - 'quit'/'exit'/'q' - Exit the program")
                print("  - 'help' - Show this message")
                print("  - 'config' - Show current configuration")
                print()
                continue

            if prompt.lower() == "config":
                print("\n📋 Current Configuration:")
                print(f"  Model: {inferencer.model_name}")
                print(f"  Max tokens: {inferencer.max_tokens}")
                print(f"  Timeout: {inferencer.timeout}s")
                print(f"  Working dir: {inferencer.working_dir}")
                print()
                continue

            run_query(inferencer, prompt)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


def dry_run(inferencer: DevmateInferencer, prompt: str):
    """
    Show the command that would be executed without running it.

    Args:
        inferencer: The DevmateInferencer instance to use.
        prompt: The prompt to show the command for.
    """
    command = inferencer.construct_command(prompt)
    command_str = " ".join(command)

    print("\n🔍 Dry Run - Command Preview")
    print("=" * 60)
    print(f"📝 Prompt: {prompt}")
    print(f"🤖 Model: {inferencer.model_name}")
    print(f"📊 Max tokens: {inferencer.max_tokens}")
    print()
    print("📜 Generated command:")
    print_separator()
    print(command_str)
    print_separator()
    print()


def main():
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(
        description="Interactive DevMate Inferencer Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode (default)
    python devmate_example.py

    # Single query
    python devmate_example.py "What is machine learning?"

    # Dry run (show command without executing)
    python devmate_example.py --dry-run "Explain Python decorators"

    # Custom model
    python devmate_example.py --model claude-3-opus "Write a haiku"

    # Custom max tokens
    python devmate_example.py --max-tokens 1024 "Short answer please"
        """,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Optional prompt to send (if not provided, enters interactive mode)",
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
        help="Show the command without executing it",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output (response only)",
    )

    args = parser.parse_args()

    # Create the inferencer
    inferencer = DevmateInferencer(
        model_name=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        no_create_commit=True,
    )

    # Print banner unless quiet mode
    if not args.quiet:
        print_banner()

    # Handle dry-run mode
    if args.dry_run:
        if args.prompt:
            dry_run(inferencer, args.prompt)
        else:
            print("❌ Error: --dry-run requires a prompt")
            sys.exit(1)
        return

    # Handle single query or interactive mode
    if args.prompt:
        # Single query mode
        result = run_query(inferencer, args.prompt, verbose=not args.quiet)

        if args.quiet:
            if result["success"]:
                print(inferencer.get_response_text(result))
            else:
                print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)

        sys.exit(0 if result["success"] else 1)
    else:
        # Interactive mode
        interactive_mode(inferencer)


if __name__ == "__main__":
    main()
