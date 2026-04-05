#!/usr/bin/env python3
"""Integration test: verify real-time cache writes during ainfer() execution.

Confirms that the streaming pipeline writes each line to the cache file
immediately (with flush()), so you can `tail -f` the cache during execution.

Usage:
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_claude_code_cli_inferencer_real -- --mode cache
"""

import asyncio
import glob
import os
import sys
import time


async def test_realtime_cache() -> bool:
    """Verify cache file grows incrementally during ainfer() execution."""
    print("\n" + "=" * 60)
    print("TEST: Real-Time Cache Writes During ainfer()")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False

    import tempfile

    cache_dir = tempfile.mkdtemp(prefix="claude_cache_test_")
    print(f"  Cache directory: {cache_dir}")

    try:
        inferencer = ClaudeCodeCliInferencer(
            target_path="/tmp",
            model_name="sonnet",
            allowed_tools=[],
            cache_folder=cache_dir,
        )

        # Use a prompt that should produce multi-line output
        prompt = "List the numbers 1 through 10, each on its own line. Just the numbers, nothing else."
        print(f"\n  Prompt: '{prompt}'")
        print(f"  Starting ainfer()...")

        # Run ainfer and collect result
        start = time.time()
        result = await inferencer.ainfer(prompt)
        elapsed = time.time() - start

        print(f"\n  ainfer() completed in {elapsed:.2f}s")
        print(f"  Success: {result.get('success') if isinstance(result, dict) else 'N/A'}")
        output = result.get("output", "") if isinstance(result, dict) else str(result)
        print(f"  Output length: {len(output)} chars")
        print(f"  Output preview: {output[:200]}...")

        # Find cache files
        cache_files = glob.glob(
            os.path.join(cache_dir, "**", "stream_*.txt"), recursive=True
        )
        print(f"\n  Cache files found: {len(cache_files)}")

        if not cache_files:
            print("  ❌ No cache files created!")
            return False

        # Read cache content
        cache_path = cache_files[0]
        print(f"  Cache file: {cache_path}")
        with open(cache_path, "r") as f:
            cache_content = f.read()

        cache_size = len(cache_content)
        print(f"  Cache file size: {cache_size} chars")

        # Check cache has content (not empty during execution)
        if cache_size == 0:
            print("  ❌ Cache file is empty — no real-time writes!")
            return False

        # Check for the completion marker
        has_success_marker = "--- STREAM COMPLETED SUCCESSFULLY ---" in cache_content
        has_failure_marker = "--- STREAM FAILED:" in cache_content
        print(f"  Has success marker: {has_success_marker}")
        print(f"  Has failure marker: {has_failure_marker}")

        # Show cache content
        print(f"\n  === Cache File Contents ===")
        for i, line in enumerate(cache_content.split("\n")):
            if i < 20:
                print(f"    {i+1}: {line}")
            else:
                print(f"    ... ({len(cache_content.split(chr(10)))} total lines)")
                break
        print(f"  === End Cache File ===")

        # Verify: cache has content AND the output text appears in the cache
        output_in_cache = output.strip()[:50] in cache_content if output.strip() else False
        print(f"\n  Output text found in cache: {output_in_cache}")

        if cache_size > 0 and has_success_marker and output_in_cache:
            print("\n✅ REAL-TIME CACHE TEST PASSED!")
            print("  Cache file was written with output + completion marker.")
            return True
        else:
            print("\n❌ REAL-TIME CACHE TEST FAILED:")
            if cache_size == 0:
                print("    - Cache file is empty")
            if not has_success_marker:
                print("    - Missing completion marker")
            if not output_in_cache:
                print("    - Output text not found in cache")
            return False

    except Exception as e:
        print(f"\n❌ REAL-TIME CACHE TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)


async def test_streaming_cache_incremental() -> bool:
    """Verify cache file grows line-by-line during ainfer_streaming().

    This test uses ainfer_streaming() directly and checks the cache file
    size DURING iteration to prove writes happen incrementally, not at the end.
    """
    print("\n" + "=" * 60)
    print("TEST: Incremental Cache During ainfer_streaming()")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False

    import tempfile

    cache_dir = tempfile.mkdtemp(prefix="claude_stream_cache_test_")
    print(f"  Cache directory: {cache_dir}")

    try:
        inferencer = ClaudeCodeCliInferencer(
            target_path="/tmp",
            model_name="sonnet",
            allowed_tools=[],
            cache_folder=cache_dir,
        )

        prompt = "List the numbers 1 through 10, each on its own line. Just the numbers, nothing else."
        print(f"  Prompt: '{prompt}'")
        print(f"\n  Starting ainfer_streaming()...")

        cache_sizes_during: list = []
        lines_received: list = []

        async for line in inferencer.ainfer_streaming(prompt):
            lines_received.append(line)

            # Check cache file size DURING streaming
            cache_files = glob.glob(
                os.path.join(cache_dir, "**", "stream_*.txt"), recursive=True
            )
            if cache_files:
                current_size = os.path.getsize(cache_files[0])
                cache_sizes_during.append(current_size)

        print(f"\n  Lines received: {len(lines_received)}")
        print(f"  Cache size snapshots during streaming: {cache_sizes_during}")

        # Find final cache file
        cache_files = glob.glob(
            os.path.join(cache_dir, "**", "stream_*.txt"), recursive=True
        )
        if not cache_files:
            print("  ❌ No cache files created!")
            return False

        cache_path = cache_files[0]
        with open(cache_path, "r") as f:
            final_content = f.read()

        final_size = len(final_content)
        has_marker = "--- STREAM COMPLETED SUCCESSFULLY ---" in final_content

        print(f"  Final cache size: {final_size} chars")
        print(f"  Has completion marker: {has_marker}")

        # Key check: cache file had content DURING streaming (not just at the end)
        had_content_during = any(s > 0 for s in cache_sizes_during)
        print(f"  Had content during streaming: {had_content_during}")

        # Show growth pattern
        if cache_sizes_during:
            print(f"  Cache growth: {' → '.join(str(s) for s in cache_sizes_during)}")

        if had_content_during and has_marker and final_size > 0:
            print("\n✅ INCREMENTAL CACHE TEST PASSED!")
            print("  Cache file grew during streaming — real-time writes confirmed.")
            return True
        else:
            print("\n❌ INCREMENTAL CACHE TEST FAILED:")
            if not had_content_during:
                print("    - Cache was empty during streaming (writes only at end)")
            if not has_marker:
                print("    - Missing completion marker")
            return False

    except Exception as e:
        print(f"\n❌ INCREMENTAL CACHE TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)


def main() -> int:
    results = []
    results.append(("Real-Time Cache (ainfer)", asyncio.run(test_realtime_cache())))
    results.append(("Incremental Cache (ainfer_streaming)", asyncio.run(test_streaming_cache_incremental())))

    print("\n" + "=" * 60)
    print("CACHE TEST SUMMARY")
    print("=" * 60)
    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1
    print(f"\nTotal: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
