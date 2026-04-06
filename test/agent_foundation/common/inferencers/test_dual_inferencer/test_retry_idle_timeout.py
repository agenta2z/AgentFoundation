

"""Manual test script for retry with incremental idle timeout.

Tests the execute_with_retry mechanism with configurable inferencers:
- Starts with a low idle_timeout_seconds (e.g., 600s)
- Doubles the timeout on each retry via on_retry_callback
- Verifies retry is working via:
  1. Cache files — one per retry attempt with STREAM FAILED/COMPLETED markers
  2. StreamingConfig log entries — shows idle_timeout per call
  3. Retry log entries — shows attempt number and exception type
  4. session.jsonl — structured log of all events

Usage:
    # DevmateSDK (fresh client per call — cleanest retry):
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_retry_idle_timeout -- \
        --inferencer-type devmate_sdk \
        --request-file request_simple_transformer.txt \
        --initial-idle-timeout 10 --max-retry 3

    # ClaudeCode (persistent client — tests stale connection recovery):
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_retry_idle_timeout -- \
        --inferencer-type claude_code \
        --request-file request_simple_transformer.txt \
        --initial-idle-timeout 10 --max-retry 3

    # DevmateCLI (subprocess — tests process cleanup on retry):
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_retry_idle_timeout -- \
        --inferencer-type devmate_cli \
        --request-file request_simple_transformer.txt \
        --initial-idle-timeout 10 --max-retry 3

    # Quick smoke test:
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_retry_idle_timeout -- \
        --request "What is 2+2? Just the number." \
        --initial-idle-timeout 3 --max-retry 3

    # Two-phase session awareness test:
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_retry_idle_timeout -- \
        --inferencer-type devmate_sdk \
        --two-phase-test \
        --phase1-prompt "My secret number is 42. Please acknowledge this briefly." \
        --phase2-prompt "What was my secret number? Just tell me the number." \
        --initial-idle-timeout 3 --max-retry 3
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)

# Directory containing this script and shared assets (templates, request files)
_SCRIPT_DIR = Path(__file__).resolve().parent

INFERENCER_CHOICES = ["devmate_sdk", "claude_code", "claude_code_cli", "devmate_cli"]


# ---------------------------------------------------------------------------
# Retry callback
# ---------------------------------------------------------------------------


def make_retry_callback(inference_args: dict, inferencer_name: str):
    """Build a callback that doubles idle_timeout_seconds on each retry.

    The wrapper in ``_ainfer_single()`` passes the **local** ``inference_args``
    dict as the third argument, so mutations directly affect the dict the
    retry lambda closes over.

    Args:
        inference_args: Fallback dict (used if local_args not provided).
        inferencer_name: Name for logging.

    Returns:
        Callback ``(attempt, exception, local_args=None) -> None``.
    """

    def on_retry(attempt: int, exception: Exception, local_args: dict = None) -> None:
        target = local_args if local_args is not None else inference_args
        current = target.get("idle_timeout_seconds", 600)
        new_timeout = current * 2
        target["idle_timeout_seconds"] = new_timeout
        logger.info(
            "[%s] on_retry_callback: attempt=%d, %s: %s — "
            "idle_timeout_seconds %d → %d",
            inferencer_name,
            attempt + 1,
            type(exception).__name__,
            str(exception)[:200],
            current,
            new_timeout,
        )

    return on_retry


# ---------------------------------------------------------------------------
# Inferencer factory
# ---------------------------------------------------------------------------


def create_inferencer(
    inferencer_type: str,
    root_folder: str,
    model: str,
    max_retry: int,
    initial_idle_timeout: int,
    total_timeout: int,
    cache_folder: str,
    inferencer_logger=None,
):
    """Create an inferencer configured for retry testing.

    Args:
        inferencer_type: One of ``INFERENCER_CHOICES``.
        root_folder: Working directory for the agent.
        model: Model name/id.
        max_retry: Maximum retry attempts.
        initial_idle_timeout: Initial idle timeout in seconds.
        total_timeout: Total timeout in seconds (caps all retries).
        cache_folder: Cache directory for streaming output.
        inferencer_logger: Logger configuration.

    Returns:
        An inferencer instance with retry enabled.
    """
    if inferencer_type == "devmate_sdk":
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (  # noqa: E501
            DevmateSDKInferencer,
        )

        return DevmateSDKInferencer(
            target_path=root_folder,
            model_name=model,
            max_retry=max_retry,
            min_retry_wait=5,
            max_retry_wait=15,
            idle_timeout_seconds=initial_idle_timeout,
            total_timeout_seconds=total_timeout,
            cache_folder=cache_folder,
            logger=inferencer_logger,
            id="devmate_sdk_retry_test",
        )

    elif inferencer_type == "claude_code":
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_inferencer import (  # noqa: E501
            ClaudeCodeInferencer,
        )

        return ClaudeCodeInferencer(
            target_path=root_folder,
            model_id=model or "",
            system_prompt="",
            max_retry=max_retry,
            min_retry_wait=5,
            max_retry_wait=15,
            idle_timeout_seconds=initial_idle_timeout,
            total_timeout_seconds=total_timeout,
            allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
            cache_folder=cache_folder,
            logger=inferencer_logger,
            id="claude_code_retry_test",
        )

    elif inferencer_type == "claude_code_cli":
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (  # noqa: E501
            ClaudeCodeCliInferencer,
        )

        return ClaudeCodeCliInferencer(
            target_path=root_folder,
            model_name=model or "sonnet",
            max_retry=max_retry,
            min_retry_wait=5,
            max_retry_wait=15,
            idle_timeout_seconds=initial_idle_timeout,
            total_timeout_seconds=total_timeout,
            cache_folder=cache_folder,
            logger=inferencer_logger,
            id="claude_code_cli_retry_test",
        )

    elif inferencer_type == "devmate_cli":
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (  # noqa: E501
            DevmateCliInferencer,
        )

        return DevmateCliInferencer(
            target_path=root_folder,
            model_name=model,
            max_retry=max_retry,
            min_retry_wait=5,
            max_retry_wait=15,
            idle_timeout_seconds=initial_idle_timeout,
            total_timeout_seconds=total_timeout,
            cache_folder=cache_folder,
            logger=inferencer_logger,
            id="devmate_cli_retry_test",
        )

    else:
        raise ValueError(
            f"Unknown inferencer type: {inferencer_type}. "
            f"Choose from: {INFERENCER_CHOICES}"
        )


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def setup_workspace(workspace: Path) -> dict:
    """Create workspace directory structure, return paths dict."""
    logs_dir = workspace / "logs"
    cache_dir = workspace / "_runtime" / "inferencer_cache"
    results_dir = workspace / "results"

    for d in [logs_dir, cache_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "logs_dir": logs_dir,
        "cache_dir": cache_dir,
        "results_dir": results_dir,
    }


def print_summary(workspace: Path, result: dict) -> None:
    """Print a human-readable summary of the retry test results."""
    print("\n" + "=" * 80)
    print("RETRY TEST SUMMARY")
    print("=" * 80)
    print(f"Workspace:      {workspace}")
    print(f"Inferencer:     {result.get('inferencer_type', 'unknown')}")
    print(f"Success:        {result.get('success', False)}")
    print(f"Retries:        {result.get('retries_observed', 'unknown')}")
    print(f"Retry Mode:     {result.get('retry_prompt_mode', 'original')}")
    print(f"Response (200): {str(result.get('response', ''))[:200]}...")

    # List cache files
    cache_dir = workspace / "_runtime" / "inferencer_cache"
    if cache_dir.exists():
        print("\nCache files (one per attempt):")
        for root, dirs, files in os.walk(cache_dir):
            for f in sorted(files):
                fpath = Path(root) / f
                size = fpath.stat().st_size
                with open(fpath) as fh:
                    lines = fh.readlines()
                    status = lines[-1].strip() if lines else "empty"
                print(f"  {fpath.relative_to(cache_dir)} ({size} bytes) — {status}")

    print("\nLog files:")
    logs_dir = workspace / "logs"
    if logs_dir.exists():
        for f in sorted(logs_dir.iterdir()):
            print(f"  {f.name} ({f.stat().st_size} bytes)")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Two-phase session awareness test
# ---------------------------------------------------------------------------


async def run_two_phase_test(
    inferencer,
    phase1_prompt: str,
    phase1_timeout: int,
    phase2_prompt: str,
    phase2_timeout: int,
    retry_prompt_mode: str,
    max_retry: int,
    total_timeout: int,
) -> dict:
    """Run two-phase session awareness test.

    Phase 1: Complete a simple task to establish session context (e.g., "My secret number is 42").
    Phase 2: Run with short timeout to trigger retries, testing if session memory is preserved.

    Args:
        inferencer: The inferencer instance to test.
        phase1_prompt: Prompt for phase 1 (establishing session context).
        phase1_timeout: Generous timeout for phase 1 to ensure completion.
        phase2_prompt: Prompt for phase 2 (testing session memory).
        phase2_timeout: Short timeout for phase 2 to trigger retries.
        retry_prompt_mode: How to transform prompt on retry.
        max_retry: Maximum retry attempts.
        total_timeout: Total timeout cap.

    Returns:
        Dict with test results including session awareness status.
    """
    result = {
        "phase1_success": False,
        "phase1_session_id": None,
        "phase1_response": None,
        "phase1_error": None,
        "phase2_success": False,
        "phase2_retries": 0,
        "phase2_response": None,
        "phase2_error": None,
        "session_aware": False,
        "secret_number_found": False,
    }

    # ========== PHASE 1: Establish session context ==========
    logger.info("=" * 60)
    logger.info("PHASE 1: Establishing session context")
    logger.info("=" * 60)
    logger.info(f"Prompt: {phase1_prompt}")
    logger.info(f"Timeout: {phase1_timeout}s (generous to ensure completion)")

    try:
        phase1_response = await inferencer.ainfer(
            phase1_prompt,
            idle_timeout_seconds=phase1_timeout,
            total_timeout_seconds=total_timeout,
        )
        result["phase1_success"] = True
        result["phase1_response"] = str(phase1_response)
        result["phase1_session_id"] = getattr(inferencer, "active_session_id", None) or getattr(inferencer, "_session_id", None)
        logger.info(f"Phase 1 completed successfully!")
        logger.info(f"Session ID: {result['phase1_session_id']}")
        logger.info(f"Response (200 chars): {str(phase1_response)[:200]}...")
    except Exception as e:
        result["phase1_error"] = f"{type(e).__name__}: {str(e)[:500]}"
        logger.error(f"Phase 1 failed: {result['phase1_error']}")
        return result

    # ========== PHASE 2: Test session memory with retries ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2: Testing session memory with retry triggers")
    logger.info("=" * 60)
    logger.info(f"Prompt: {phase2_prompt}")
    logger.info(f"Timeout: {phase2_timeout}s (short to trigger retries)")
    logger.info(f"Retry prompt mode: {retry_prompt_mode}")
    logger.info(f"Max retries: {max_retry}")

    # Build inference args with retry callback
    retry_count = [0]

    def on_retry(attempt: int, exception: Exception, local_args: dict = None) -> None:
        retry_count[0] = attempt + 1
        target = local_args if local_args is not None else {}
        current = target.get("idle_timeout_seconds", phase2_timeout)
        new_timeout = current * 2
        if local_args is not None:
            local_args["idle_timeout_seconds"] = new_timeout
        logger.info(
            f"[Phase 2] Retry {attempt + 1}/{max_retry}: {type(exception).__name__} — "
            f"idle_timeout {current}s → {new_timeout}s"
        )

    inference_args = {
        "idle_timeout_seconds": phase2_timeout,
        "retry_prompt_mode": retry_prompt_mode,
        "on_retry_callback": on_retry,
    }

    try:
        phase2_response = await inferencer.ainfer(
            phase2_prompt,
            total_timeout_seconds=total_timeout,
            **inference_args,
        )
        result["phase2_success"] = True
        result["phase2_response"] = str(phase2_response)
        result["phase2_retries"] = retry_count[0]
        logger.info(f"Phase 2 completed!")
        logger.info(f"Retries observed: {retry_count[0]}")
        logger.info(f"Response (300 chars): {str(phase2_response)[:300]}...")

        # Check if agent remembered the secret number "42"
        response_text = str(phase2_response).lower()
        if "42" in response_text:
            result["secret_number_found"] = True
            result["session_aware"] = True
            logger.info("✅ SUCCESS: Agent remembered the secret number (42)!")
        else:
            logger.warning("❌ Agent did NOT mention the secret number (42)")
            # Check if agent at least acknowledged there was a previous conversation
            if any(phrase in response_text for phrase in [
                "previous", "earlier", "before", "remember", "mentioned",
                "secret", "number", "told", "said"
            ]):
                result["session_aware"] = True
                logger.info("⚠️ Agent seems session-aware but didn't recall exact number")
            else:
                logger.warning("❌ Agent shows no session awareness")

    except Exception as e:
        result["phase2_error"] = f"{type(e).__name__}: {str(e)[:500]}"
        result["phase2_retries"] = retry_count[0]
        logger.error(f"Phase 2 failed after {retry_count[0]} retries: {result['phase2_error']}")

    return result


def print_two_phase_summary(workspace: Path, result: dict) -> None:
    """Print a human-readable summary of the two-phase test results."""
    print("\n" + "=" * 80)
    print("TWO-PHASE SESSION AWARENESS TEST RESULTS")
    print("=" * 80)

    print("\n📋 PHASE 1: Establish Session Context")
    print("-" * 40)
    print(f"  Success:    {result.get('phase1_success', False)}")
    print(f"  Session ID: {result.get('phase1_session_id', 'N/A')}")
    if result.get("phase1_error"):
        print(f"  Error:      {result['phase1_error']}")
    else:
        print(f"  Response:   {str(result.get('phase1_response', ''))[:150]}...")

    print("\n📋 PHASE 2: Test Memory with Retries")
    print("-" * 40)
    print(f"  Success:    {result.get('phase2_success', False)}")
    print(f"  Retries:    {result.get('phase2_retries', 0)}")
    if result.get("phase2_error"):
        print(f"  Error:      {result['phase2_error']}")
    else:
        print(f"  Response:   {str(result.get('phase2_response', ''))[:200]}...")

    print("\n" + "=" * 80)
    print("🎯 SESSION AWARENESS VERDICT")
    print("=" * 80)

    if result.get("secret_number_found"):
        print("✅ PASS: Session-aware retry WORKS!")
        print("   Agent correctly remembered '42' from Phase 1 during Phase 2.")
        print("   This proves conversation history is preserved across:")
        print("   • Multiple calls to the same inferencer instance")
        print("   • Retry attempts within a single call")
    elif result.get("session_aware"):
        print("⚠️ PARTIAL: Agent shows session awareness but didn't recall exact value.")
        print("   Agent acknowledged previous conversation but couldn't retrieve '42'.")
    else:
        print("❌ FAIL: Session-aware retry NOT working.")
        print("   Agent did NOT remember context from Phase 1.")
        print("   Possible causes:")
        print("   • Session history not persisted before Phase 1 completed")
        print("   • auto_resume not working correctly")
        print("   • Server not replaying conversation history on resume")

    # List cache files
    cache_dir = workspace / "_runtime" / "inferencer_cache"
    if cache_dir.exists():
        print("\n📁 Cache files:")
        for root, dirs, files in os.walk(cache_dir):
            for f in sorted(files):
                fpath = Path(root) / f
                size = fpath.stat().st_size
                print(f"  {fpath.relative_to(cache_dir)} ({size} bytes)")

    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--inferencer-type",
    default="devmate_sdk",
    type=click.Choice(INFERENCER_CHOICES),
    help="Which inferencer to test retry behavior for.",
)
@click.option(
    "--request",
    default="Explain what the DevmateSDKInferencer does in 2 sentences.",
    help="Prompt text to send to the inferencer.",
)
@click.option(
    "--request-file",
    default=None,
    help=(
        "Path to a file containing the request text (overrides --request). "
        "Can be relative to cwd or a filename in the test_dual_inferencer/ "
        "directory (e.g., request_simple_transformer.txt)."
    ),
)
@click.option(
    "--root-folder",
    default=None,
    help="Root folder for the agent. Defaults to cwd.",
)
@click.option(
    "--model",
    default="claude-sonnet-4-5",
    help="Model to use.",
)
@click.option(
    "--initial-idle-timeout",
    default=600,
    type=int,
    help="Initial idle timeout in seconds. Doubled on each retry.",
)
@click.option(
    "--total-timeout",
    default=7200,
    type=int,
    help="Total timeout in seconds (caps all retries).",
)
@click.option(
    "--max-retry",
    default=3,
    type=int,
    help="Maximum number of retry attempts.",
)
@click.option(
    "--workspace",
    default=None,
    help="Workspace directory. Auto-generated if not specified.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level.",
)
@click.option(
    "--retry-prompt-mode",
    default="original",
    type=click.Choice(["original", "simple_retry", "retry_with_original"]),
    help="How to transform the prompt on retry. "
    "'simple_retry' sends a short resume message (tests session awareness). "
    "'retry_with_original' sends resume + original prompt.",
)
@click.option(
    "--two-phase-test",
    is_flag=True,
    default=False,
    help="Enable two-phase session awareness test: "
    "Phase 1: Complete a simple task to establish session context. "
    "Phase 2: Run with short timeout to trigger retries, testing session memory.",
)
@click.option(
    "--phase1-prompt",
    default="My secret number is 42. Please acknowledge this briefly and remember it.",
    help="Prompt for phase 1 (establishing session context).",
)
@click.option(
    "--phase1-timeout",
    default=120,
    type=int,
    help="Generous timeout for phase 1 to ensure completion (seconds).",
)
@click.option(
    "--phase2-prompt",
    default="What was my secret number? Just tell me the number.",
    help="Prompt for phase 2 (testing session memory).",
)
def main(
    inferencer_type: str,
    request: str,
    request_file: Optional[str],
    root_folder: Optional[str],
    model: str,
    initial_idle_timeout: int,
    total_timeout: int,
    max_retry: int,
    workspace: Optional[str],
    log_level: str,
    retry_prompt_mode: str,
    two_phase_test: bool,
    phase1_prompt: str,
    phase1_timeout: int,
    phase2_prompt: str,
) -> None:
    """Test retry mechanism with incremental idle timeout doubling."""
    # 1. Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # 2. Resolve request
    if request_file is not None:
        request_path = Path(request_file)
        if not request_path.is_file():
            request_path = _SCRIPT_DIR / request_file
        if not request_path.is_file():
            raise FileNotFoundError(
                f"Request file not found: {request_file} "
                f"(also tried {_SCRIPT_DIR / request_file})"
            )
        logger.info("Reading request from file: %s", request_path)
        request_text = request_path.read_text()
    else:
        request_text = request

    # 3. Set up workspace
    if workspace is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_path = Path(
            f"./_workspace/test_retry_{inferencer_type}_{timestamp}"
        )
    else:
        workspace_path = Path(workspace)

    logger.info("Workspace: %s", workspace_path)
    paths = setup_workspace(workspace_path)

    # 4. Resolve root folder
    if root_folder is None:
        root_folder = os.getcwd()

    # 5. Set up JsonLogger for structured logging
    from rich_python_utils.common_objects.debuggable import LoggerConfig
    from rich_python_utils.io_utils.json_io import JsonLogger, SpaceExtMode

    json_logger = JsonLogger(
        file_path=str(paths["logs_dir"] / "session.jsonl"),
        append=True,
        is_artifact=True,
        parts_min_size=0,
        space_ext_mode=SpaceExtMode.MOVE,
        parts_file_namer=lambda obj: obj.get("type", "")
        if isinstance(obj, dict)
        else "",
    )
    inferencer_logger = [
        (json_logger, LoggerConfig(pass_item_key_as="parts_key_path_root")),
        print,
    ]

    # 6. Create inferencer with retry enabled
    logger.info(
        "Creating %s: max_retry=%d, initial_idle_timeout=%d, "
        "total_timeout=%d, model=%s",
        inferencer_type,
        max_retry,
        initial_idle_timeout,
        total_timeout,
        model,
    )

    inferencer = create_inferencer(
        inferencer_type=inferencer_type,
        root_folder=root_folder,
        model=model,
        max_retry=max_retry,
        initial_idle_timeout=initial_idle_timeout,
        total_timeout=total_timeout,
        cache_folder=str(paths["cache_dir"]),
        inferencer_logger=inferencer_logger,
    )

    # 7. Build inference_args with on_retry_callback
    inference_args = {
        "idle_timeout_seconds": initial_idle_timeout,
        "retry_prompt_mode": retry_prompt_mode,
    }
    inference_args["on_retry_callback"] = make_retry_callback(
        inference_args, f"RetryTest[{inferencer_type}]"
    )

    # 8. Save test configuration
    config = {
        "inferencer_type": inferencer_type,
        "request": request_text[:500],
        "root_folder": root_folder,
        "model": model,
        "initial_idle_timeout": initial_idle_timeout,
        "total_timeout": total_timeout,
        "max_retry": max_retry,
        "retry_prompt_mode": retry_prompt_mode,
        "timestamp": datetime.now().isoformat(),
    }
    (paths["results_dir"] / "config.json").write_text(
        json.dumps(config, indent=2)
    )

    # 9. Run inference with retry
    logger.info("Starting inference with retry (idle_timeout doubling)...")
    logger.info(
        "Timeout schedule: %s",
        " → ".join(
            str(initial_idle_timeout * (2**i)) for i in range(max_retry)
        ),
    )

    result = {
        "inferencer_type": inferencer_type,
        "success": False,
        "response": None,
        "retries_observed": 0,
        "retry_prompt_mode": retry_prompt_mode,
        "error": None,
    }

    # Check if two-phase test is requested
    if two_phase_test:
        # Run two-phase session awareness test
        two_phase_result = asyncio.run(
            run_two_phase_test(
                inferencer=inferencer,
                phase1_prompt=phase1_prompt,
                phase1_timeout=phase1_timeout,
                phase2_prompt=phase2_prompt,
                phase2_timeout=initial_idle_timeout,
                retry_prompt_mode=retry_prompt_mode,
                max_retry=max_retry,
                total_timeout=total_timeout,
            )
        )

        # Save two-phase results
        (paths["results_dir"] / "two_phase_result.json").write_text(
            json.dumps(two_phase_result, indent=2, default=str)
        )

        # Print two-phase summary
        print_two_phase_summary(workspace_path, two_phase_result)
        return

    async def run():
        try:
            response = await inferencer.ainfer(
                request_text,
                **inference_args,
            )
            result["success"] = True
            result["response"] = str(response)
            logger.info(
                "Inference succeeded! Response length: %d", len(str(response))
            )
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            logger.error(
                "Inference failed after all retries: %s", result["error"]
            )

    asyncio.run(run())

    # 10. Detect how many retries happened by counting cache files
    cache_dir = paths["cache_dir"]
    cache_files = []
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.startswith("stream_") and f.endswith(".txt"):
                cache_files.append(Path(root) / f)
    result["retries_observed"] = max(0, len(cache_files) - 1)

    # 11. Save results
    result_serializable = {k: v for k, v in result.items()}
    (paths["results_dir"] / "result.json").write_text(
        json.dumps(result_serializable, indent=2, default=str)
    )

    # 12. Print summary
    print_summary(workspace_path, result)


if __name__ == "__main__":
    main()
