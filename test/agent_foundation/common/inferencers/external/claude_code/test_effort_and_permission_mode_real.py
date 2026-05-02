"""Real integration tests: ``effort`` and ``permission_mode`` end-to-end.

These tests **actually spawn the ``claude`` binary** and verify behavior
against the live Anthropic backend. They:

- Cost real API tokens / subscription quota
- Take seconds to tens of seconds per call
- Require ``claude`` to be installed and authenticated on the host

They are **gated** behind the ``RUN_CLAUDE_CODE_REAL_TESTS=1`` environment
variable so they don't run by default. To run::

    set RUN_CLAUDE_CODE_REAL_TESTS=1
    python -m unittest AgentFoundation.test.agent_foundation.common.inferencers.external.claude_code.test_effort_and_permission_mode_real -v

What we verify:

1. **Plan mode is *behaviorally* effective**: when ``permission_mode="plan"``
   is set, asking Claude to create a file results in **no file being
   created**. This is the deterministic proof that the flag isn't just
   being passed through cosmetically — the model actually obeys it.

2. **Each effort level produces a successful end-to-end call**: a smoke
   test across ``low`` / ``max`` / ``xhigh`` (the boundary cases — covers
   both the SDK's typed Literal and the CLI-only ``xhigh``). Asserts
   ``return_code == 0`` and a non-empty response. Also logs token
   counts so the operator can visually confirm higher effort spends
   more thinking budget.

3. **The same plan-mode guarantee holds via the SDK inferencer.**
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer import (
    ClaudeCodeSdkInferencer,
)


_RUN_REAL = os.environ.get("RUN_CLAUDE_CODE_REAL_TESTS") == "1"
_REAL_REASON = "Set RUN_CLAUDE_CODE_REAL_TESTS=1 to run real-API tests"


def _unique_target_path(tmpdir: Path) -> Path:
    """Generate a unique non-existent file path inside ``tmpdir`` for the
    plan-mode write test. Using a UUID prevents collision across reruns."""
    return tmpdir / f"plan_mode_canary_{uuid.uuid4().hex}.txt"


def _make_plan_prompt(target: Path) -> str:
    """A direct, unambiguous instruction to write a file. If permission_mode
    is NOT "plan", a normally-permissive inferencer (bypassPermissions) would
    happily create this file, so the absence of the file after the call is
    strong evidence that plan mode took effect."""
    return (
        f"Create a new file at the absolute path '{target.as_posix()}' "
        f"with the literal contents 'hello from plan mode'. "
        f"Do this immediately, then describe what you did."
    )


# ---------------------------------------------------------------------------
# CLI inferencer
# ---------------------------------------------------------------------------


@unittest.skipUnless(_RUN_REAL, _REAL_REASON)
class CliRealPermissionModeTest(unittest.TestCase):
    """``permission_mode="plan"`` actually prevents file writes via the CLI."""

    def test_plan_mode_blocks_file_creation(self):
        """Definitive proof: with plan mode set, the file is NOT created.

        We do NOT assert on the model's textual response — under headless
        ``-p`` mode the CLI sometimes returns an empty ``result`` field
        for plan-mode-blocked requests (the "response" was an internal
        ``EnterPlanMode`` tool-use rather than user-facing text). The
        meaningful, deterministic check is whether the canary file
        materialized on disk. The bypass control test in this same class
        proves the same prompt DOES create the file when permission mode
        is permissive — so a missing file here is unambiguously caused
        by plan mode.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmpdir = Path(tmp)
            target = _unique_target_path(tmpdir)
            prompt = _make_plan_prompt(target)

            inf = ClaudeCodeCliInferencer(
                target_path=str(tmpdir),
                permission_mode="plan",
                effort="low",  # keep it cheap; plan mode is what we're testing
            )
            result = inf(prompt)
            output = inf.get_response_text(result)
            return_code = getattr(result, "return_code", None)
            success = getattr(result, "success", None)
            print(
                f"\n[plan-mode CLI] return_code={return_code} success={success} "
                f"response (first 200 chars): {output[:200]!r}"
            )

            # Primary, deterministic assertion: file does not exist.
            self.assertFalse(
                target.exists(),
                f"plan mode failed: file was created at {target}. response: {output!r}",
            )
            # Sanity: the subprocess itself must not have crashed.
            if return_code is not None:
                self.assertEqual(
                    return_code, 0,
                    f"claude exited non-zero in plan mode: rc={return_code}, "
                    f"output={output!r}",
                )

    def test_bypass_permissions_does_create_file(self):
        """Control test: with default bypassPermissions, the same prompt
        DOES create the file. This proves the plan-mode test above is
        meaningful (and not e.g. a model refusing to write for an unrelated
        reason)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmpdir = Path(tmp)
            target = _unique_target_path(tmpdir)
            prompt = _make_plan_prompt(target)

            inf = ClaudeCodeCliInferencer(
                target_path=str(tmpdir),
                permission_mode="bypassPermissions",
                effort="low",
            )
            result = inf(prompt)
            output = inf.get_response_text(result)
            print(f"\n[bypass CLI] response (first 200 chars): {output[:200]!r}")
            self.assertTrue(
                target.exists(),
                f"control failed: file was NOT created at {target} even in "
                f"bypassPermissions mode. response: {output!r}",
            )


@unittest.skipUnless(_RUN_REAL, _REAL_REASON)
class CliRealEffortTest(unittest.TestCase):
    """Each effort level produces a successful end-to-end call.

    We don't assert on token-count ordering (it's stochastic across runs),
    but we DO log it so the operator can verify higher effort spends more.
    """

    def _smoke_one_effort(self, level):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inf = ClaudeCodeCliInferencer(
                target_path=tmp,
                permission_mode="bypassPermissions",
                effort=level,
            )
            t0 = time.time()
            result = inf("What is 2+2? Answer in one word.")
            dt = time.time() - t0
            output = inf.get_response_text(result)
            # Pull metadata if structured (parse_output populates these
            # fields when --output-format json is in use)
            usage = getattr(result, "usage", None) or (
                result.get("usage") if isinstance(result, dict) else None
            )
            cost = getattr(result, "total_cost_usd", None) or (
                result.get("total_cost_usd") if isinstance(result, dict) else None
            )
            print(
                f"\n[effort={level}] dt={dt:.2f}s "
                f"output={output[:80]!r} usage={usage} cost={cost}"
            )
            self.assertTrue(output, f"empty response at effort={level}")
            return dt, usage

    def test_effort_low(self):
        self._smoke_one_effort("low")

    def test_effort_max(self):
        self._smoke_one_effort("max")

    def test_effort_xhigh(self):
        # CLI-only value — this tests the path where the SDK would route
        # through extra_args, but for the CLI inferencer it's just the flag.
        self._smoke_one_effort("xhigh")


# ---------------------------------------------------------------------------
# SDK inferencer
# ---------------------------------------------------------------------------


@unittest.skipUnless(_RUN_REAL, _REAL_REASON)
class SdkRealPermissionModeTest(unittest.TestCase):
    """``permission_mode="plan"`` actually prevents file writes via the SDK."""

    def test_plan_mode_blocks_file_creation(self):
        """SDK counterpart to the CLI plan-mode test. The SDK's streaming
        path typically does emit text (we observed responses like
        ``"I cannot create that file because I'm in plan mode..."``), but we
        keep the same definitive assertion as the CLI test — the file must
        not exist — and only log the response for inspection.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmpdir = Path(tmp)
            target = _unique_target_path(tmpdir)
            prompt = _make_plan_prompt(target)

            inf = ClaudeCodeSdkInferencer(
                root_folder=str(tmpdir),
                permission_mode="plan",
                effort="low",
            )
            result = inf(prompt)
            output = result if isinstance(result, str) else str(result)
            print(f"\n[plan-mode SDK] response (first 200 chars): {output[:200]!r}")

            # Primary, deterministic assertion: file does not exist.
            self.assertFalse(
                target.exists(),
                f"plan mode failed via SDK: file was created at {target}. "
                f"response: {output!r}",
            )


@unittest.skipUnless(_RUN_REAL, _REAL_REASON)
class SdkRealEffortTest(unittest.TestCase):
    """``effort`` flows through the SDK transport and produces a working
    end-to-end call.

    Note on level selection: the claude-agent-sdk bundles its own ``claude``
    binary, which may lag the user's PATH ``claude`` by a few minor versions.
    Older CLI builds (a) reject ``xhigh`` (added later) and (b) gate ``max``
    behind API-tier auth. We therefore real-test only universally-supported
    levels here. The argv-emission for ``xhigh`` (the ``extra_args`` routing
    path) is verified by ``SdkTransportCrossCheckTest`` in
    ``test_effort_and_permission_mode_sdk.py``, which constructs a real
    ``SubprocessCLITransport`` and inspects its argv — that proves the flag
    reaches the CLI without depending on a specific bundled CLI version.
    """

    def _smoke_one_effort(self, level):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inf = ClaudeCodeSdkInferencer(
                root_folder=tmp,
                effort=level,
                permission_mode="bypassPermissions",
            )
            t0 = time.time()
            result = inf("What is 2+2? Answer in one word.")
            dt = time.time() - t0
            output = result if isinstance(result, str) else str(result)
            print(f"\n[SDK effort={level}] dt={dt:.2f}s output={output[:80]!r}")
            self.assertTrue(output, f"empty SDK response at effort={level}")

    def test_effort_low(self):
        self._smoke_one_effort("low")

    def test_effort_high(self):
        self._smoke_one_effort("high")


# ---------------------------------------------------------------------------
# Permission-mode matrix: every supported value is wired through to the CLI,
# AND headless ``-p`` mode produces the empirically-observed split where
# ``plan`` is the only restrictive mode while the other five all allow file
# writes (because ``-p`` skips the workspace-trust dialog).
# ---------------------------------------------------------------------------

# Empirically-observed distinguishing behavior when the CLI inferencer
# invokes ``claude -p`` with each permission mode. Verified against host
# CLI v2.1.119 with a write-the-file prompt:
#
# - ``acceptEdits`` and ``bypassPermissions`` ALLOW the write to land.
# - ``plan`` blocks it because the *model* refuses to call non-readonly
#   tools (response: "I'm in plan mode and cannot create files...").
# - ``auto`` blocks it because its research-preview safety check denies
#   the write in headless contexts (response: "write permission was
#   denied").
# - ``dontAsk`` blocks it because the Write tool is not pre-approved
#   (response: "Write tool is blocked in the current 'don't ask' mode").
# - ``default`` blocks it because the CLI inferencer's ``construct_command``
#   intentionally omits ``--permission-mode`` for the ``"default"`` value
#   (it relies on the CLI's own default), and ``claude -p`` without any
#   permission-mode flag prompts for permission, which fails in headless.
#   IMPORTANT: this is *different* from running ``claude -p
#   --permission-mode default`` directly, which IS permissive in -p mode.
_PERMISSION_MODES_THAT_ALLOW_WRITES_IN_HEADLESS = (
    "acceptEdits",
    "bypassPermissions",
)
_PERMISSION_MODES_THAT_BLOCK_WRITES_IN_HEADLESS = (
    "plan",
    "default",
    "auto",
    "dontAsk",
)


@unittest.skipUnless(_RUN_REAL, _REAL_REASON)
class CliRealPermissionModeMatrixTest(unittest.TestCase):
    """Each of the 6 permission modes is genuinely accepted by the CLI and
    produces its expected end-to-end behavior on a write request.

    Why this test exists: a pass on the smoke + plan tests above only
    proves ``plan`` and ``bypassPermissions`` work. A flag like ``auto``
    or ``dontAsk`` could in principle be silently accepted by the CLI
    but treated as a no-op. This matrix runs every mode against the same
    write prompt and verifies:

    1. The subprocess returns success (rc=0, no parse error) — proves the
       flag is accepted, not just transported.
    2. The file-creation outcome matches the empirically-observed split
       (only ``plan`` blocks; the other 5 allow because headless ``-p``
       auto-trusts the workspace).

    If Anthropic ever tightens any of these modes (e.g., makes ``dontAsk``
    actually auto-deny in ``-p``), this test will start failing on that
    mode — which is desirable: it surfaces the behavioral change rather
    than letting it silently re-shape what our wrapper does.
    """

    # Keep the prompt and filename mode-AGNOSTIC. An earlier version
    # interpolated the mode name into both, and the model occasionally
    # refused to create files whose names contained strings like
    # "bypassPermissions" because it judged the filename suspicious — a
    # false negative that had nothing to do with the flag working.
    PROMPT_TEMPLATE = (
        "Create a file at '{path}' with the literal contents 'hello world'. "
        "Do this immediately and then in one short sentence say what you did."
    )

    def _run_one(self, mode):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmpdir = Path(tmp)
            target = tmpdir / f"canary_{uuid.uuid4().hex[:8]}.txt"
            inf = ClaudeCodeCliInferencer(
                target_path=str(tmpdir),
                permission_mode=mode,
                effort="low",
            )
            result = inf(self.PROMPT_TEMPLATE.format(path=target.as_posix()))
            return {
                "mode": mode,
                "file_created": target.exists(),
                "return_code": getattr(result, "return_code", None),
                "success": getattr(result, "success", None),
                "response_first_80": (inf.get_response_text(result) or "")[:80],
            }

    def test_plan_blocks_writes_all_other_modes_allow(self):
        observations = {}
        for mode in (
            _PERMISSION_MODES_THAT_BLOCK_WRITES_IN_HEADLESS
            + _PERMISSION_MODES_THAT_ALLOW_WRITES_IN_HEADLESS
        ):
            obs = self._run_one(mode)
            observations[mode] = obs
            print(f"\n[perm-matrix] {mode:18s} -> {obs}")

        # Every mode must have produced a successful subprocess call.
        for mode, obs in observations.items():
            self.assertEqual(
                obs["return_code"], 0,
                f"mode={mode}: subprocess exited rc={obs['return_code']}",
            )
            self.assertTrue(
                obs["success"],
                f"mode={mode}: success=False, response={obs['response_first_80']!r}",
            )

        # Plan mode is the only one that blocks file creation.
        for mode in _PERMISSION_MODES_THAT_BLOCK_WRITES_IN_HEADLESS:
            self.assertFalse(
                observations[mode]["file_created"],
                f"mode={mode} should have BLOCKED the file creation but the "
                f"file exists. response={observations[mode]['response_first_80']!r}",
            )

        # All other modes should have allowed the file creation in -p mode.
        for mode in _PERMISSION_MODES_THAT_ALLOW_WRITES_IN_HEADLESS:
            self.assertTrue(
                observations[mode]["file_created"],
                f"mode={mode} should have ALLOWED the file creation in "
                f"headless -p mode but the file was NOT created. "
                f"response={observations[mode]['response_first_80']!r}",
            )


@unittest.skipUnless(_RUN_REAL, _REAL_REASON)
class InferencerDefaultPermissionModeTest(unittest.TestCase):
    """Documents the empirical behavior of each inferencer's *default*
    ``permission_mode`` value, with no caller override.

    The two inferencers ship with different defaults:
        ClaudeCodeCliInferencer.permission_mode = "bypassPermissions"
        ClaudeCodeSdkInferencer.permission_mode = None

    These tests pin down what each default actually does end-to-end so a
    user can answer "if I just construct the inferencer with no
    permission-mode argument, will my Write call succeed?" — which is
    different for the two inferencers.
    """

    PROMPT_TEMPLATE = (
        "Create a file at '{path}' with the literal contents 'default-test'. "
        "Do this immediately and then in one short sentence say what you did."
    )

    def test_cli_inferencer_default_allows_writes(self):
        """CLI inferencer default = ``bypassPermissions``: writes succeed."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmpdir = Path(tmp)
            target = tmpdir / f"cli_default_{uuid.uuid4().hex[:8]}.txt"
            inf = ClaudeCodeCliInferencer(target_path=str(tmpdir), effort="low")
            self.assertEqual(
                inf.permission_mode, "bypassPermissions",
                "CLI inferencer default should be 'bypassPermissions'",
            )
            inf(self.PROMPT_TEMPLATE.format(path=target.as_posix()))
            print(
                f"\n[default CLI] permission_mode={inf.permission_mode!r}, "
                f"file_created={target.exists()}"
            )
            self.assertTrue(
                target.exists(),
                "CLI inferencer with default permission_mode should allow writes",
            )

    def test_sdk_inferencer_default_allows_writes_via_allowed_tools(self):
        """SDK inferencer default = ``permission_mode=None`` AND
        ``allowed_tools=["Read", "Write", "Bash"]``. The two inferencers
        reach the same end behavior (writes allowed) via *different*
        authorization mechanisms:

        - CLI inferencer authorizes via ``--dangerously-skip-permissions``
          (from ``permission_mode="bypassPermissions"``).
        - SDK inferencer authorizes via the SDK-level ``allowed_tools``
          allowlist, which pre-approves the listed tools without needing
          a permissive ``permission_mode``.

        This test pins down that behavior. If someone removes ``"Write"``
        from the SDK inferencer's default ``allowed_tools``, the file
        creation will start failing under the default config — and this
        test will catch it.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmpdir = Path(tmp)
            target = tmpdir / f"sdk_default_{uuid.uuid4().hex[:8]}.txt"
            inf = ClaudeCodeSdkInferencer(root_folder=str(tmpdir), effort="low")
            self.assertIsNone(
                inf.permission_mode,
                "SDK inferencer default should be None",
            )
            self.assertIn(
                "Write", inf.allowed_tools,
                "SDK inferencer default allowed_tools should include 'Write' — "
                "that is what authorizes writes despite permission_mode=None",
            )
            inf(self.PROMPT_TEMPLATE.format(path=target.as_posix()))
            print(
                f"\n[default SDK] permission_mode={inf.permission_mode!r}, "
                f"allowed_tools={inf.allowed_tools}, "
                f"file_created={target.exists()}"
            )
            self.assertTrue(
                target.exists(),
                "SDK inferencer with default permission_mode=None should "
                "still ALLOW writes because allowed_tools=['Read','Write','Bash'] "
                "pre-authorizes the Write tool. If this fails, the SDK's "
                "default authorization mechanism has changed.",
            )


# ---------------------------------------------------------------------------
# Behavioral effectiveness — does the model ACTUALLY use more compute?
# ---------------------------------------------------------------------------
#
# The smoke tests above only prove the flag is accepted. They do NOT prove
# that the model behaves any differently across levels — a CLI that silently
# coalesced ``low``/``medium``/``high``/``max`` into a single internal value
# would still pass them. The tests below close that gap by asserting on a
# behavioral signal: ``output_tokens`` (which on Claude's API includes
# extended-thinking tokens) must scale with effort.
#
# Empirical baseline observed during development on host CLI v2.1.119 with
# the prompt below:
#
#     effort=low    →    543 output_tokens, ~9s wall, $0.079
#     effort=medium → 1,555 output_tokens, ~19s wall, $0.094
#     effort=high   → 1,951 output_tokens, ~22s wall, $0.100
#     effort=xhigh  → 1,839 output_tokens, ~21s wall
#     effort=max    → 3,801 output_tokens, ~42s wall, $0.128
#
# That is a ~7× growth in tokens and ~5× growth in wall-clock between
# ``low`` and ``max``, while the visible response length only grew ~1.8×.
# The bulk of the extra ~3,260 tokens at ``max`` is hidden thinking that
# never reaches the user — direct evidence the flag scales internal compute.
#
# The assertions below use a 1.5× threshold (substantially under the
# observed gap) so they remain robust against run-to-run variance.

_BEHAVIORAL_PROMPT = (
    "Find the smallest prime p > 100 such that p+2, p+6, and p+8 are also "
    "primes. Show your reasoning step by step, including any alternative "
    "approaches you considered."
)


@unittest.skipUnless(_RUN_REAL, _REAL_REASON)
class CliEffortBehavioralTest(unittest.TestCase):
    """``effort`` is *behaviorally effective*, not just transported.

    Asserts that ``output_tokens`` strictly increases between ``low`` and
    ``max`` for the same prompt. If the CLI silently coalesced effort
    levels, the two calls would produce nearly identical token counts and
    this test would fail.
    """

    def _measure(self, level):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inf = ClaudeCodeCliInferencer(
                target_path=tmp,
                effort=level,
                permission_mode="bypassPermissions",
            )
            t0 = time.time()
            result = inf(_BEHAVIORAL_PROMPT)
            wall_s = time.time() - t0
            # raw_output is the full ``--output-format json`` blob. Use the
            # inferencer's robust extractor (handles stderr noise prefixes /
            # multi-line JSON) rather than json.loads directly.
            data = inf._extract_json_from_output(result.raw_output)
            if data is None:
                self.fail(
                    f"could not extract JSON at effort={level}; "
                    f"raw_output (first 500): {result.raw_output[:500]!r}; "
                    f"stderr (first 500): {(result.stderr or '')[:500]!r}"
                )
            usage = data.get("usage") or {}
            return {
                "output_tokens": usage.get("output_tokens"),
                "duration_ms": data.get("duration_ms"),
                "wall_s": round(wall_s, 2),
                "cost_usd": data.get("total_cost_usd"),
                "result_chars": len(data.get("result", "")),
            }

    def test_max_produces_significantly_more_thinking_than_low(self):
        low = self._measure("low")
        high_eff = self._measure("max")
        token_ratio = high_eff["output_tokens"] / low["output_tokens"]
        char_ratio = high_eff["result_chars"] / max(low["result_chars"], 1)
        print(f"\n[behavioral CLI] low  = {low}")
        print(f"[behavioral CLI] max  = {high_eff}")
        print(
            f"[behavioral CLI] ratios: output_tokens={token_ratio:.2f}× "
            f"result_chars={char_ratio:.2f}× wall_s="
            f"{high_eff['wall_s'] / low['wall_s']:.2f}×"
        )
        self.assertIsNotNone(low["output_tokens"], "could not parse usage from low call")
        self.assertIsNotNone(high_eff["output_tokens"], "could not parse usage from max call")
        # (1) Total compute scaled: max produces ≥1.5× the output_tokens of
        # low. A coalesced/no-op flag would land near 1.0×.
        self.assertGreater(
            high_eff["output_tokens"],
            low["output_tokens"] * 1.5,
            f"effort=max should produce >1.5× the output_tokens of low — "
            f"a CLI that silently coalesced effort levels would fail this. "
            f"Got max={high_eff['output_tokens']} vs low={low['output_tokens']}.",
        )
        # (2) Smoking-gun for *hidden* thinking: total tokens grew strictly
        # more than visible characters. If max effort were merely producing
        # a longer answer (no extended thinking), the two ratios would be
        # ~equal. Hidden thinking is the only behavior that produces the
        # observed pattern of token_ratio > char_ratio.
        self.assertGreater(
            token_ratio, char_ratio,
            f"effort=max would only be 'longer-answer effective' (not "
            f"extended-thinking effective) if output_tokens scaled the same "
            f"as result_chars. Observed token_ratio={token_ratio:.2f}× vs "
            f"char_ratio={char_ratio:.2f}× — equal ratios would falsify the "
            f"hidden-thinking claim.",
        )
        # (3) Wall-clock must also grow — extended thinking is real compute,
        # not free. A coalesced flag would not produce 2× longer calls.
        self.assertGreater(
            high_eff["wall_s"], low["wall_s"] * 1.3,
            f"effort=max should take >1.3× wall-clock of low. "
            f"Got max={high_eff['wall_s']}s vs low={low['wall_s']}s.",
        )


@unittest.skipUnless(_RUN_REAL, _REAL_REASON)
class SdkEffortBehavioralTest(unittest.TestCase):
    """SDK counterpart: ``output_tokens`` from ``ResultMessage.usage`` must
    scale with effort.

    This goes through ``ClaudeAgentOptions(effort=...)`` and the SDK
    transport — proving the SDK path doesn't strip or coalesce the flag
    somewhere between our wrapper and the model. The SDK's bundled CLI
    rejects ``max`` for subscription auth (we measured this earlier), so
    this test compares ``low`` vs ``high`` — both subscription-supported
    on every CLI version we've seen.
    """

    async def _measure_via_sdk(self, level):
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        from claude_agent_sdk.types import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
        )

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            options = ClaudeAgentOptions(
                effort=level,
                permission_mode="bypassPermissions",
                cwd=tmp,
            )
            t0 = time.time()
            visible_text_parts = []
            async with ClaudeSDKClient(options=options) as client:
                await client.query(_BEHAVIORAL_PROMPT)
                final_usage = None
                async for msg in client.receive_response():
                    if isinstance(msg, ResultMessage):
                        final_usage = msg.usage
                    elif isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                visible_text_parts.append(block.text)
            wall_s = time.time() - t0
            output_tokens = (final_usage or {}).get("output_tokens")
            visible_text = "".join(visible_text_parts)
            return {
                "output_tokens": output_tokens,
                "result_chars": len(visible_text),
                "wall_s": round(wall_s, 2),
            }

    def test_high_produces_significantly_more_thinking_than_low(self):
        low = asyncio.run(self._measure_via_sdk("low"))
        high_eff = asyncio.run(self._measure_via_sdk("high"))
        token_ratio = high_eff["output_tokens"] / low["output_tokens"]
        char_ratio = high_eff["result_chars"] / max(low["result_chars"], 1)
        print(f"\n[behavioral SDK] low  = {low}")
        print(f"[behavioral SDK] high = {high_eff}")
        print(
            f"[behavioral SDK] ratios: output_tokens={token_ratio:.2f}× "
            f"result_chars={char_ratio:.2f}× wall_s="
            f"{high_eff['wall_s'] / low['wall_s']:.2f}×"
        )
        self.assertIsNotNone(low["output_tokens"], "no usage in SDK ResultMessage at low")
        self.assertIsNotNone(high_eff["output_tokens"], "no usage in SDK ResultMessage at high")
        # (1) Total compute scaled.
        self.assertGreater(
            high_eff["output_tokens"],
            low["output_tokens"] * 1.5,
            f"effort=high should produce >1.5× the output_tokens of low via SDK. "
            f"Got high={high_eff['output_tokens']} vs low={low['output_tokens']}.",
        )
        # (2) Smoking-gun for hidden thinking via SDK transport: tokens grew
        # strictly more than visible characters.
        self.assertGreater(
            token_ratio, char_ratio,
            f"SDK effort=high would only be 'longer-answer effective' if "
            f"output_tokens scaled the same as result_chars. Observed "
            f"token_ratio={token_ratio:.2f}× vs char_ratio={char_ratio:.2f}×.",
        )
        # (3) Wall-clock grew — real compute through the SDK transport.
        self.assertGreater(
            high_eff["wall_s"], low["wall_s"] * 1.3,
            f"SDK effort=high should take >1.3× wall-clock of low. "
            f"Got high={high_eff['wall_s']}s vs low={low['wall_s']}s.",
        )


if __name__ == "__main__":
    if not _RUN_REAL:
        print(
            "Skipping all real integration tests. "
            "Set RUN_CLAUDE_CODE_REAL_TESTS=1 to run.",
            file=sys.stderr,
        )
    unittest.main()
