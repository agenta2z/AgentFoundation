"""Reproduce the production rovodev failure mode using the example harness.

Production failure (server_20260615_194631_8e0863a8, turn_002, "hello"):
  - rovodev was spawned with --output-file
  - output file stayed EMPTY for 120s
  - watchdog killed the process
  - server fell back to raw stdout (the banner) — displayed as assistant answer

Production config (from openteam/server/backends/factories.py:258-264):
    RovoDevCliInferencer(
        target_path=ctx.working_dir,          # real workspace, not tempdir
        idle_timeout_seconds=600,
        tool_use_idle_timeout_seconds=600,
        cache_folder=ctx.cache_dir,           # ★ NOT in the existing example
        enable_legacy=True,
    )
  → ainfer_streaming(...)                     # production path

This script mirrors that config exactly and runs streaming inference. It also
sweeps the 3 hypotheses (cache off/on; output_file explicit vs auto; real
workspace vs tempdir) so we can pin which variable triggers the empty-file bug.

Run:
    python example_rovodev_streaming_with_output_file.py --scenario all
    python example_rovodev_streaming_with_output_file.py --scenario B_cache_only
    python example_rovodev_streaming_with_output_file.py --scenario C_production_exact

Honest assessment criteria:
    PASS = output_file is non-empty AND contains the LLM answer (not banner)
    FAIL = output_file is empty after timeout OR contains only banner text
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Make AgentFoundation + RichPythonUtils importable (mirrors sibling examples)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_agent_foundation_root = os.path.normpath(
    os.path.join(_script_dir, "..", "..", "..", "..", "..", "..", "..")
)
_src_dir = os.path.join(_agent_foundation_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_rich_utils_src = os.path.normpath(
    os.path.join(_agent_foundation_root, "..", "RichPythonUtils", "src")
)
if os.path.isdir(_rich_utils_src) and _rich_utils_src not in sys.path:
    sys.path.insert(0, _rich_utils_src)

# Deferred import — paths are now configured
def _import_rovodev():
    from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
        RovoDevCliInferencer,
    )
    return RovoDevCliInferencer


# Production exact constants (from factories.py:258-264)
PROD_IDLE_TIMEOUT = 600
PROD_TOOL_USE_TIMEOUT = 600


def _banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


_BANNER_MARKERS = (
    "Working in",
    "Creating agent...",
    "Started ",
    "MCP servers",
    "Turn off prompt collection",
    "Jira projects:",
    "Using model:",
    "Session context:",
    "\u2517",          # box-drawing chars from the TUI banner
    "\u2501",
)


def _is_banner_only(text: str) -> bool:
    """True if every non-empty line of ``text`` is a known TUI banner marker."""
    if not text:
        return False
    return all(
        not line.strip() or any(m in line for m in _BANNER_MARKERS)
        for line in text.splitlines()
    )


def _verdict(scenario, inferencer, explicit_output_file, chunks, total_chars,
             elapsed, error):
    """Honest pass/fail classification using the inferencer's documented API.

    Per ``RovoDevCliInferencer`` (rovodev_cli_inferencer.py:131, 581, 606-628):
      * Class invariant ``streams_differ_from_final_output = True`` — the
        clean LLM answer is NOT the streamed stdout; it is captured into
        ``--output-file`` and surfaced via ``get_final_output()``.
      * ``get_final_output()`` is the canonical post-inference accessor.
        It reads from ``self._last_clean_output`` which is populated by
        ``ainfer_streaming()`` BEFORE the auto-injected temp file is deleted
        (see docstring at line 606-614).
      * For explicit ``output_file=`` callers the file persists and we can
        verify it on disk too.

    Verdict ladder (most authoritative first):
      1. ``get_final_output()`` returns clean, non-banner text  → PASS
      2. explicit ``output_file`` exists and contains clean text → PASS
      3. otherwise classify FAIL with the specific failure mode.
    """
    # Layer 1: the documented post-inference accessor
    try:
        final = inferencer.get_final_output()
    except Exception as e:  # pragma: no cover — defensive
        final = None
        error = error or f"get_final_output raised {type(e).__name__}: {e}"

    # Layer 2: the documented backing attribute (in case get_final_output
    # returns None due to a regression; lets us see the cache state)
    last_clean = getattr(inferencer, "_last_clean_output", None)

    # Layer 3: explicit caller-supplied output_file, if any
    of_text = ""
    of_size = 0
    of_exists = False
    if explicit_output_file:
        p = Path(explicit_output_file)
        of_exists = p.exists()
        if of_exists:
            try:
                of_text = p.read_text(encoding="utf-8", errors="replace")
                of_size = len(of_text)
            except OSError as e:
                of_text = f"(read error: {e})"

    # Decide the verdict using the documented hierarchy
    if error:
        verdict = "ERROR"
    elif final and final.strip() and not _is_banner_only(final):
        verdict = "PASS (get_final_output)"
    elif explicit_output_file and of_size and not _is_banner_only(of_text):
        verdict = "PASS (explicit output_file)"
    elif last_clean and last_clean.strip() and not _is_banner_only(last_clean):
        verdict = "PASS (_last_clean_output cached)"
    elif final and _is_banner_only(final):
        verdict = "FAIL (final output is banner only)"
    elif explicit_output_file and of_size and _is_banner_only(of_text):
        verdict = "FAIL (explicit output_file is banner only)"
    elif final is None and last_clean is None and of_size == 0:
        verdict = "FAIL (no clean output anywhere)"
    else:
        verdict = "FAIL (empty clean output)"

    print()
    print(f"  ── Verdict for {scenario!r}: {verdict} ──")
    print(f"     get_final_output():    {(final or '')[:120]!r}")
    print(f"     _last_clean_output:    {(last_clean or '')[:120]!r}")
    print(f"     explicit_output_file:  {explicit_output_file}")
    print(f"     explicit_of_exists:    {of_exists}  size={of_size} bytes")
    if of_size:
        print(f"     explicit_of_preview:   {of_text[:120]!r}")
    print(f"     stream chunks={chunks}  total_chars={total_chars}  elapsed={elapsed:.1f}s")
    if error:
        print(f"     error:                 {error}")
    print()

    return {
        "scenario": scenario,
        "verdict": verdict,
        "final_output_len": len(final or ""),
        "last_clean_len": len(last_clean or ""),
        "explicit_of_size": of_size,
        "chunks": chunks,
        "total_chars": total_chars,
        "elapsed": elapsed,
        "error": error,
    }


def run_one(scenario: str, target_path: str, output_file: Optional[str],
            cache_folder: Optional[str], query: str) -> dict:
    """Construct + invoke rovodev with the given config; capture results."""
    _banner(f"Scenario: {scenario}")
    print(f"  target_path:   {target_path}")
    print(f"  output_file:   {output_file}  ({'explicit' if output_file else 'auto-inject'})")
    print(f"  cache_folder:  {cache_folder}")
    print(f"  query:         {query!r}")
    print()

    RovoDevCliInferencer = _import_rovodev()
    inferencer = RovoDevCliInferencer(
        target_path=target_path,
        output_file=output_file,        # None → auto-injected (production behavior)
        idle_timeout_seconds=PROD_IDLE_TIMEOUT,
        tool_use_idle_timeout_seconds=PROD_TOOL_USE_TIMEOUT,
        cache_folder=cache_folder,
        enable_legacy=True,
    )

    print(f"  Rovo Dev: ", end="", flush=True)
    start = time.time()
    chunks = 0
    total_chars = 0
    error = None
    try:
        for chunk in inferencer.infer_streaming(query):
            if chunk.strip():
                # Echo a single dot per chunk to keep output readable
                print(".", end="", flush=True)
                chunks += 1
                total_chars += len(chunk)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    elapsed = time.time() - start
    print()

    # Authoritative verdict uses the inferencer's documented post-inference API
    # (``get_final_output()`` + ``_last_clean_output``) rather than guessing the
    # auto-injected temp-file location, which is intentionally cleaned up.
    # For explicit ``output_file=`` callers we ALSO verify the on-disk file.
    return _verdict(scenario, inferencer, output_file, chunks, total_chars,
                    elapsed, error)


def _scenarios(query: str) -> list[dict]:
    """Build the scenario sweep — each varies ONE production-vs-example axis."""
    tmp = tempfile.mkdtemp(prefix="rovodev_repro_")
    cache_dir = tempfile.mkdtemp(prefix="rovodev_cache_")
    real_workspace = "/Users/tchen7/MyProjects/CoreProjects/AgentFoundation"
    explicit_of = os.path.join(tmp, "explicit_output.md")

    return [
        # Control: the WORKING example config (no cache, tempdir target, auto-output_file)
        {"id": "A_control_example",
         "target_path": tmp, "output_file": None, "cache_folder": None},

        # Hypothesis 2: cache_folder is the culprit
        {"id": "B_cache_only",
         "target_path": tmp, "output_file": None, "cache_folder": cache_dir},

        # Production exact: real workspace + cache + auto output_file
        {"id": "C_production_exact",
         "target_path": real_workspace, "output_file": None, "cache_folder": cache_dir},

        # Explicit output_file, no cache (rules out auto-injection bug)
        {"id": "D_explicit_of_no_cache",
         "target_path": tmp, "output_file": explicit_of, "cache_folder": None},

        # Real workspace, no cache (isolates target_path effect)
        {"id": "E_real_workspace_no_cache",
         "target_path": real_workspace, "output_file": None, "cache_folder": None},
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="all",
                   choices=["all", "A_control_example", "B_cache_only",
                            "C_production_exact", "D_explicit_of_no_cache",
                            "E_real_workspace_no_cache"])
    p.add_argument("--query", default="Say hello in 3 words.")
    args = p.parse_args()

    all_scenarios = _scenarios(args.query)
    if args.scenario != "all":
        all_scenarios = [s for s in all_scenarios if s["id"] == args.scenario]

    results = []
    for s in all_scenarios:
        try:
            results.append(run_one(s["id"], s["target_path"], s["output_file"],
                                   s["cache_folder"], args.query))
        except Exception as e:
            print(f"  ✗ scenario {s['id']!r} crashed: {type(e).__name__}: {e}")
            results.append({"scenario": s["id"], "verdict": "CRASH", "error": str(e)})

    # Summary
    _banner("SUMMARY")
    print(f"  {'Scenario':<32} {'Verdict':<35} {'final_len':>9} {'cached_len':>10} {'expl_of':>7}")
    print("  " + "-" * 96)
    for r in results:
        print(f"  {r['scenario']:<32} {r.get('verdict', '?'):<35} "
              f"{r.get('final_output_len', '-'):>9} "
              f"{r.get('last_clean_len', '-'):>10} "
              f"{r.get('explicit_of_size', '-'):>7}")
    print()

    # Diagnostic interpretation — pin the variable that flips PASS → FAIL.
    # We classify on the verdict prefix (PASS / FAIL / ERROR) which is stable
    # across the richer sub-labels emitted by ``_verdict``.
    def _status(v: str) -> str:
        if v.startswith("PASS"):
            return "PASS"
        if v.startswith("FAIL"):
            return "FAIL"
        return "ERROR"

    print("  Diagnostic interpretation:")
    by_id = {r["scenario"]: _status(r.get("verdict", "")) for r in results}
    a = by_id.get("A_control_example")
    b = by_id.get("B_cache_only")
    c = by_id.get("C_production_exact")
    d = by_id.get("D_explicit_of_no_cache")
    e = by_id.get("E_real_workspace_no_cache")

    if a and b:
        if a == "PASS" and b == "FAIL":
            print("    → cache_folder is the failure trigger "
                  "(A passes without it, B fails with it).")
        elif a == "FAIL" and b == "FAIL":
            print("    → cache_folder is NOT the trigger; failure is independent.")
        elif a == "PASS" and b == "PASS":
            print("    → cache_folder alone does not reproduce the failure.")
    if d == "PASS" and (a == "FAIL" or e == "FAIL"):
        print("    → auto-injected output_file is the trigger "
              "(explicit output_file passes, auto-injected fails).")
    if e == "FAIL" and a == "PASS":
        print("    → real workspace target_path (not cache, not auto-inject) "
              "triggers the failure.")
    if c == "FAIL" and all(s == "PASS" for s in [a, b, d, e] if s is not None):
        print("    → only the production-exact combination fails; the failure "
              "is a multi-axis interaction.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
