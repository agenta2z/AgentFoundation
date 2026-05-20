#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Real-inference integration test: MetamateSDKInferencer + metamate_standalone.

Drives the SDK inferencer with ``use_standalone=True`` so the client
class is resolved to ``metamate_standalone.cli.metamate_graphql.MetamateGraphQLClient``
instead of the upstream ``//msl/metamate/cli:metamate_graphql``. Sends a
real query through ``engine_start_v2`` + polling, prints the response.

Run:
    buck2 run //_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/external/metamate:test_metamate_standalone_inferencer
    buck2 run //_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/external/metamate:test_metamate_standalone_inferencer -- --query "Reply with: hi"

Tests three modes:
  - sync: ``inferencer.infer(query)``
  - async-single: ``await inferencer.ainfer(query)``
  - streaming: ``async for chunk in inferencer.ainfer_streaming(query): ...``

Exit code is 0 iff ALL three modes return a non-empty response AND the
client class actually resolved to the standalone (verified by inspecting
``MetamateGraphQLClient.__module__``).
"""

import argparse
import asyncio
import logging
import os
import sys
import time


def _ensure_metamate_cat_env() -> None:
    """Mint a CAT once and put it in ``METAMATE_CAT`` for the standalone.

    The standalone library is BYO-CAT (no self-minting). When this test
    runs on a devserver, we mint a token via Meta's CryptoCAT util once
    here in the test fixture and export it to the env so every
    ``MetamateSDKInferencer(use_standalone=True)`` call below picks it
    up through the standalone's ``cat_provider`` chain.
    """
    if os.environ.get("METAMATE_CAT", "").strip():
        print("[fixture] METAMATE_CAT already set; skipping mint.")
        return
    try:
        from libfb.py.interngraph.auth.interngraph_crypto_auth_token_util import (
            InternGraphCryptoAuthTokenUtil as AuthUtil,
        )
    except ImportError as e:
        print(
            f"[fixture] cannot mint CAT (no libfb.interngraph dep): {e}\n"
            "Set METAMATE_CAT to a serialized token list to run this test."
        )
        sys.exit(2)
    # 1874465445999900 is MetaMate's InternGraph app_id (mirrors
    # metamate_standalone.cli.utils.DEFAULT_INTERNGRAPH_APP).
    token = AuthUtil.get_serialized_token_list_for_current_unix_user(
        1874465445999900, token_timeout_seconds=86400
    )
    os.environ["METAMATE_CAT"] = token
    print(f"[fixture] minted CAT bundle ({len(token)} chars) → METAMATE_CAT")


def _resolve_and_verify_standalone_used() -> str:
    """Return the dotted module name of the resolved MetamateGraphQLClient.

    Asserts it lives under ``metamate_standalone.*`` (not the upstream
    ``msl.metamate.*``). This is the strongest guarantee we can give
    without instrumenting the SDK inferencer itself.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.common import (
        resolve_metamate_client_cls,
    )

    cls = resolve_metamate_client_cls(use_standalone=True)
    mod = cls.__module__
    assert mod.startswith("metamate_standalone."), (
        f"resolve_metamate_client_cls(True) returned class from {mod!r}; "
        f"expected something under metamate_standalone.*"
    )
    return mod


def test_sync(query: str) -> bool:
    print("\n" + "=" * 60)
    print("TEST 1: sync infer() via standalone client")
    print("=" * 60)
    from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
        MetamateSDKInferencer,
    )

    inf = MetamateSDKInferencer(
        use_standalone=True,
        total_timeout_seconds=120,
        idle_timeout_seconds=120,
        poll_interval_seconds=3.0,
    )
    start = time.time()
    try:
        result = inf.infer(query)
    except Exception as e:
        print(f"FAIL ({type(e).__name__}): {e}")
        return False
    elapsed = time.time() - start
    text = str(result)
    print(f"Got {len(text)} chars in {elapsed:.1f}s")
    print(f"Session id: {inf.active_session_id}")
    print("-" * 40)
    print(text[:500] + ("..." if len(text) > 500 else ""))
    print("-" * 40)
    ok = len(text.strip()) > 0
    print("PASS" if ok else "FAIL: empty response")
    return ok


async def test_async_single(query: str) -> bool:
    print("\n" + "=" * 60)
    print("TEST 2: async ainfer() via standalone client")
    print("=" * 60)
    from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
        MetamateSDKInferencer,
    )

    inf = MetamateSDKInferencer(
        use_standalone=True,
        total_timeout_seconds=120,
        idle_timeout_seconds=120,
        poll_interval_seconds=3.0,
    )
    start = time.time()
    try:
        result = await inf.ainfer(query)
    except Exception as e:
        print(f"FAIL ({type(e).__name__}): {e}")
        return False
    elapsed = time.time() - start
    text = str(result)
    print(f"Got {len(text)} chars in {elapsed:.1f}s")
    print("-" * 40)
    print(text[:500] + ("..." if len(text) > 500 else ""))
    print("-" * 40)
    ok = len(text.strip()) > 0
    print("PASS" if ok else "FAIL: empty response")
    return ok


async def test_streaming(query: str) -> bool:
    print("\n" + "=" * 60)
    print("TEST 3: streaming ainfer_streaming() via standalone client")
    print("=" * 60)
    from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
        MetamateSDKInferencer,
    )

    inf = MetamateSDKInferencer(
        use_standalone=True,
        total_timeout_seconds=120,
        idle_timeout_seconds=120,
        poll_interval_seconds=3.0,
    )
    start = time.time()
    chunks: list[str] = []
    try:
        async for chunk in inf.ainfer_streaming(query):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\nFAIL ({type(e).__name__}): {e}")
        return False
    elapsed = time.time() - start
    text = "".join(chunks)
    print(f"\n  -> {len(chunks)} chunks, {len(text)} chars, {elapsed:.1f}s")
    ok = len(text.strip()) > 0
    print("PASS" if ok else "FAIL: empty stream")
    return ok


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Reply with exactly: pong")
    parser.add_argument(
        "--skip-streaming",
        action="store_true",
        help="Run only sync + async-single (skip streaming).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("METAMATE STANDALONE INFERENCER — END-TO-END TEST")
    print("=" * 60)

    # The truly-standalone client is BYO-CAT — mint one for it.
    _ensure_metamate_cat_env()

    mod = _resolve_and_verify_standalone_used()
    print(f"Standalone client class resolved to: {mod}")

    results: list[tuple[str, bool]] = []
    results.append(("sync infer()", test_sync(args.query)))
    results.append(("async ainfer()", asyncio.run(test_async_single(args.query))))
    if not args.skip_streaming:
        results.append(("ainfer_streaming()", asyncio.run(test_streaming(args.query))))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = 0
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if ok:
            passed += 1
    print(f"\nTotal: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
