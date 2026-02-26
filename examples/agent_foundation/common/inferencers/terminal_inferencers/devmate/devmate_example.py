#!/usr/bin/env python3
"""
Interactive DevMate Inferencer Example with Session Support and Streaming.

This example demonstrates how to use the DevmateInferencer to interact with
DevMate CLI from Python, including session continuation for multi-turn
conversations and real-time streaming output.

Usage:
    # Run with buck2
    buck2 run fbcode//_tony_dev/ScienceModelingTools/examples/agent_foundation/common/inferencers/terminal_inferencers/devmate:devmate_example

    # Or run directly (if dependencies are available)
    python devmate_example.py

    # With a specific prompt (non-interactive)
    buck2 run ... -- "What is Python?"

    # Streaming mode - see output in real-time
    buck2 run ... -- --streaming "Explain Python decorators"

    # Interactive mode with streaming enabled
    buck2 run ... -- --streaming

    # Stream output to file and terminal
    buck2 run ... -- --streaming --output-file response.txt "What is Python?"

    # With initial and follow-up prompts (demonstrates session continuation)
    buck2 run ... -- "What is Python?" --follow-up "What was my last question?"

    # Show command only (dry-run)
    buck2 run ... -- --dry-run "What is Python?"

Features:
    - Interactive mode: Enter queries and get responses in a loop
    - Streaming mode: See DevMate responses in real-time as they're generated
    - Single query mode: Pass a prompt as argument
    - Multi-turn mode: Pass initial and follow-up prompts to test session continuation
    - Dry-run mode: Show the command without executing
    - Output to file: Write streaming output to a file
    - Configurable model and max tokens
"""

import argparse
import sys
import time

import resolve_path  # noqa: F401

from science_modeling_tools.common.inferencers.terminal_inferencers.devmate.devmate_inferencer import (
    DevmateInferencer,
)


# Default prompts for multi-turn demo
DEFAULT_INITIAL_PROMPT = "What is Python?"
DEFAULT_FOLLOWUP_PROMPT = "What was my last question?"


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
    show_command: bool = True,
) -> dict:
    """
    Run a single query through DevMate.

    Args:
        inferencer: The DevmateInferencer instance to use.
        prompt: The prompt to send to DevMate.
        verbose: Whether to print detailed output.
        show_command: Whether to show the command being executed.

    Returns:
        The result dictionary from the inference.
    """
    if verbose:
        print(f"\n📝 Prompt: {prompt}")
        print_separator()

        if show_command:
            # Show the actual command that will be executed
            command = inferencer.construct_command(prompt)
            print("📜 Command to execute:")
            print(f"   {command}")
            print()

            # Show pre-execution scripts if any
            if inferencer.pre_exec_scripts:
                print("📂 Pre-execution scripts:")
                for script in inferencer.pre_exec_scripts:
                    print(f"   {script}")
                print()

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


def run_streaming_query(
    inferencer: DevmateInferencer,
    prompt: str,
    verbose: bool = True,
    output_file: str = None,
    use_callback: bool = False,
) -> dict:
    """
    Run a streaming query through DevMate with real-time output.

    Args:
        inferencer: The DevmateInferencer instance to use.
        prompt: The prompt to send to DevMate.
        verbose: Whether to print detailed output.
        output_file: Optional file path to write streaming output to.
        use_callback: If True, use callback-based streaming; else use iterator.

    Returns:
        The result dictionary from the inference.
    """
    if verbose:
        print(f"\n📝 Prompt: {prompt}")
        print_separator()
        print("⏳ Streaming response from DevMate...")
        print_separator()
        print()

    start_time = time.time()

    # Open output file if specified
    file_handle = None
    if output_file:
        file_handle = open(output_file, "w")
        if verbose:
            print(f"📄 Writing output to: {output_file}")
            print()

    try:
        if use_callback:
            # Callback-based streaming
            def stream_callback(line: str):
                print(line, end="", flush=True)
                if file_handle:
                    file_handle.write(line)
                    file_handle.flush()

            # Consume the iterator with callback
            for _ in inferencer.infer_streaming(
                prompt, stream_callback=stream_callback
            ):
                pass
        else:
            # Iterator-based streaming (print each line as it arrives)
            for line in inferencer.infer_streaming(
                prompt, output_stream=file_handle if file_handle else None
            ):
                print(line, end="", flush=True)

    finally:
        if file_handle:
            file_handle.close()

    execution_time = time.time() - start_time

    # Get the final parsed result
    result = inferencer.get_streaming_result()

    if verbose:
        print()
        print_separator()
        print(f"\n⏱️  Execution time: {execution_time:.2f}s")
        print(f"📊 Return code: {result['return_code']}")

        if result.get("session_id"):
            print(f"🔑 Session ID: {result['session_id']}")

        if result.get("trajectory_url"):
            print(f"🔗 Trajectory: {result['trajectory_url']}")

        if output_file:
            print(f"📄 Output saved to: {output_file}")

    result["execution_time"] = execution_time
    return result


def interactive_mode(inferencer: DevmateInferencer, streaming: bool = False):
    """
    Run in interactive mode, allowing user to enter prompts.

    Args:
        inferencer: The DevmateInferencer instance to use.
        streaming: Whether to start in streaming mode.
    """
    stream_mode = streaming

    print("\n🎯 Interactive Mode")
    print("=" * 60)
    print("Enter your queries below. Commands:")
    print("  - Type your question and press Enter to query DevMate")
    print("  - Type 'quit', 'exit', or 'q' to exit")
    print("  - Type 'stream' to toggle streaming mode")
    print("  - Type 'help' for this message")
    print("  - Type 'config' to show current configuration")
    print(f"  - Streaming: {'ON 🟢' if stream_mode else 'OFF 🔴'}")
    print("=" * 60)
    print()

    while True:
        try:
            mode_indicator = "🌊" if stream_mode else "🤖"
            prompt = input(f"{mode_indicator} You: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye!")
                break

            if prompt.lower() == "stream":
                stream_mode = not stream_mode
                print(f"\n🔄 Streaming mode: {'ON 🟢' if stream_mode else 'OFF 🔴'}")
                print()
                continue

            if prompt.lower() == "help":
                print("\nCommands:")
                print("  - Type your question to query DevMate")
                print("  - 'quit'/'exit'/'q' - Exit the program")
                print("  - 'stream' - Toggle streaming mode")
                print("  - 'help' - Show this message")
                print("  - 'config' - Show current configuration")
                print(
                    f"  - Current streaming mode: {'ON 🟢' if stream_mode else 'OFF 🔴'}"
                )
                print()
                continue

            if prompt.lower() == "config":
                print("\n📋 Current Configuration:")
                print(f"  Model: {inferencer.model_name}")
                print(f"  Max tokens: {inferencer.max_tokens}")
                print(f"  Timeout: {inferencer.timeout}s")
                print(f"  Working dir: {inferencer.working_dir}")
                print(f"  Streaming: {'ON 🟢' if stream_mode else 'OFF 🔴'}")
                print()
                continue

            # Execute query based on streaming mode
            if stream_mode:
                run_streaming_query(inferencer, prompt)
            else:
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
    # construct_command now returns a shell command string (not a list)
    command = inferencer.construct_command(prompt)

    print("\n🔍 Dry Run - Command Preview")
    print("=" * 60)
    print(f"📝 Prompt: {prompt}")
    print(f"🤖 Model: {inferencer.model_name}")
    print(f"📊 Max tokens: {inferencer.max_tokens}")
    print(f"📂 Working dir: {inferencer.working_dir}")
    print(f"🔀 Repo path: {inferencer.repo_path}")
    print()

    # Show pre-execution scripts
    if inferencer.pre_exec_scripts:
        print("📂 Pre-execution scripts (run before main command):")
        print_separator()
        for i, script in enumerate(inferencer.pre_exec_scripts, 1):
            print(f"  {i}. {script}")
        print_separator()
        print()

    print("📜 Main command (shell execution):")
    print_separator()
    print(command)
    print_separator()
    print()

    # Show the full execution flow
    print("🔄 Full execution flow:")
    print_separator()
    if inferencer.pre_exec_scripts:
        for script in inferencer.pre_exec_scripts:
            print(f"$ {script}")
    print(f"$ {command}")
    print_separator()
    print()


def run_multi_turn_demo(
    inferencer: DevmateInferencer,
    initial_prompt: str,
    followup_prompt: str,
    verbose: bool = True,
):
    """
    Run a multi-turn demo showing session continuation.

    Args:
        inferencer: The DevmateInferencer instance to use.
        initial_prompt: The first prompt to send.
        followup_prompt: The follow-up prompt to send.
        verbose: Whether to print detailed output.
    """
    print("\n🔄 Multi-Turn Session Demo")
    print("=" * 60)
    print("This demonstrates session continuation - the follow-up prompt")
    print("will be sent to the same DevMate session as the initial prompt.")
    print("=" * 60)

    # First query - creates a new session
    print("\n📍 TURN 1: Initial Query")
    print("-" * 60)
    result1 = run_query(inferencer, initial_prompt, verbose=verbose)

    # Show session info
    session_id = result1.get("session_id")
    if session_id:
        print(f"\n🔑 Session ID: {session_id}")
        print(f"📊 Active session: {inferencer.active_session_id}")
    else:
        print("\n⚠️  No session ID returned - follow-up may not work correctly")

    # Second query - should automatically resume the session
    print("\n📍 TURN 2: Follow-up Query (resuming session)")
    print("-" * 60)

    # Show the resume command that will be generated
    resume_command = inferencer.construct_command(
        followup_prompt,
        session_id=inferencer.active_session_id,
        resume=True,
    )
    print("📜 Resume command:")
    print(f"   {resume_command}")
    print()

    result2 = run_query(
        inferencer, followup_prompt, verbose=verbose, show_command=False
    )

    # Show session history
    print("\n📚 Session History:")
    print("-" * 60)
    history = inferencer.get_session_history()
    for i, turn in enumerate(history, 1):
        role = "👤 User" if turn["from"] == "user" else "🤖 System"
        content = (
            turn["content"][:100] + "..."
            if len(turn["content"]) > 100
            else turn["content"]
        )
        print(f"  {i}. {role}: {content}")
    print("-" * 60)

    return result1, result2


def main():
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(
        description="Interactive DevMate Inferencer Example with Session Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode (default)
    python devmate_example.py

    # Single query
    python devmate_example.py "What is machine learning?"

    # Multi-turn demo (session continuation)
    python devmate_example.py "What is Python?" --follow-up "What was my last question?"

    # Multi-turn with defaults (empty prompt triggers defaults)
    python devmate_example.py "" --follow-up ""

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
        help="Optional prompt to send (if not provided, enters interactive mode). Use empty string with --follow-up to use defaults.",
    )

    parser.add_argument(
        "--follow-up",
        "-f",
        default=None,
        help=f"Follow-up prompt for multi-turn demo. If empty string, defaults to: '{DEFAULT_FOLLOWUP_PROMPT}'",
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

    parser.add_argument(
        "--streaming",
        "-s",
        action="store_true",
        help="Enable streaming mode - output appears in real-time as it's generated",
    )

    parser.add_argument(
        "--output-file",
        "-o",
        default=None,
        help="Write streaming output to file (in addition to terminal)",
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

    # Handle multi-turn demo mode (--follow-up provided)
    if args.follow_up is not None:
        # Use defaults if prompts are empty
        initial = args.prompt if args.prompt else DEFAULT_INITIAL_PROMPT
        followup = args.follow_up if args.follow_up else DEFAULT_FOLLOWUP_PROMPT

        print(f"📝 Initial prompt: {initial}")
        print(f"📝 Follow-up prompt: {followup}")

        result1, result2 = run_multi_turn_demo(
            inferencer,
            initial,
            followup,
            verbose=not args.quiet,
        )

        # Exit with success if both queries succeeded
        success = result1["success"] and result2["success"]
        sys.exit(0 if success else 1)

    # Handle single query or interactive mode
    if args.prompt:
        # Single query mode
        if args.streaming:
            # Streaming single query
            result = run_streaming_query(
                inferencer,
                args.prompt,
                verbose=not args.quiet,
                output_file=args.output_file,
            )
        else:
            # Non-streaming single query
            result = run_query(inferencer, args.prompt, verbose=not args.quiet)

        if args.quiet:
            if result["success"]:
                print(inferencer.get_response_text(result))
            else:
                print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)

        sys.exit(0 if result["success"] else 1)
    else:
        # Interactive mode (with streaming option)
        interactive_mode(inferencer, streaming=args.streaming)


if __name__ == "__main__":
    main()
