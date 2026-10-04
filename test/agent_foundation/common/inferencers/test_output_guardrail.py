"""Unit tests for the output guardrail (LLM-based quality judge).

Covers:
- output_guardrail_inferencer=None (default): zero behavioral change
- Guardrail judge returns "PASS": output accepted normally
- Guardrail judge returns "FAIL: reason": output rejected → recovery fires
- _parse_guardrail_verdict overridable by subclass
- Guardrail fail-open: judge exception → output accepted (no crash)
- v5 Fix #1: rendered prompt reaches judge via ``_fallback_state`` ContextVar
  (no new instance state), with ``_guardrail_input_text`` overridable hook
- v5 Fix #2: rejected response persisted via ``log_debug`` before fail-fast
- v5 Fix #4: internal recovery re-runs the RENDERED prompt, not the raw seed
- v5 Fix #5: ``HopelessOutputError`` is terminal (non-retryable), AND the
  rejected response is still persisted before the raise
"""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_foundation.common.inferencers.inferencer_base import (
    _current_fallback_state,
    HopelessOutputError,
    InferencerBase,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import (
    FallbackMode,
    OutputValidationExhaustedError,
)


@attrs
class _StubInferencer(InferencerBase):
    """A minimal leaf inferencer for testing."""

    _responses = attrib(factory=list, init=False)
    _call_count = attrib(default=0, init=False)

    def set_responses(self, responses):
        self._responses = list(responses)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._responses):
            r = self._responses[idx]
            if isinstance(r, Exception):
                raise r
            return r
        return f"response_{idx}"


@attrs
class _StubJudge(InferencerBase):
    """A stub guardrail judge that returns configurable verdicts."""

    _verdict = attrib(default="PASS")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._verdict


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _render_under(inf, rendered_input, output):
    token = _current_fallback_state.set({"rendered_input": rendered_input})
    try:
        return inf._render_guardrail_prompt(output)
    finally:
        _current_fallback_state.reset(token)


class TestGuardrailDisabledByDefault(unittest.TestCase):
    def test_no_guardrail_accepts_normally(self):
        inf = _StubInferencer()
        inf.set_responses(["good output"])
        result = inf.infer("input")
        self.assertEqual(str(result), "good output")

    def test_no_guardrail_async(self):
        inf = _StubInferencer()
        inf.set_responses(["good output"])
        result = _run(inf.ainfer("input"))
        self.assertEqual(str(result), "good output")


class TestGuardrailAccepts(unittest.TestCase):
    def test_pass_verdict_accepts(self):
        judge = _StubJudge(verdict="PASS")
        inf = _StubInferencer(output_guardrail_inferencer=judge)
        inf.set_responses(["good output"])
        result = inf.infer("input")
        self.assertEqual(str(result), "good output")

    def test_pass_verdict_async(self):
        judge = _StubJudge(verdict="PASS")
        inf = _StubInferencer(output_guardrail_inferencer=judge)
        inf.set_responses(["good output"])
        result = _run(inf.ainfer("input"))
        self.assertEqual(str(result), "good output")


class TestGuardrailRejects(unittest.TestCase):
    def test_fail_verdict_triggers_retry(self):
        """When the judge says FAIL, the inferencer retries via recovery."""
        judge = _StubJudge(verdict="FAIL: output is just narration")
        inf = _StubInferencer(
            output_guardrail_inferencer=judge,
            max_retry=2,
        )
        inf.set_responses(["bad narration", "also bad", "still bad"])
        # All attempts rejected → should exhaust retries and raise/return default
        # With default_return_or_raise=None and all attempts failing validation,
        # execute_with_retry raises the last exception.
        with self.assertRaises(Exception):
            inf.infer("input")
        # The inferencer was called multiple times (retried)
        self.assertGreater(inf._call_count, 1)

    def test_fail_then_pass_accepts_second(self):
        """First attempt rejected, second attempt passes the judge."""
        call_count = [0]

        @attrs
        class _FlipJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                call_count[0] += 1
                return "FAIL: too short" if call_count[0] == 1 else "PASS"

        judge = _FlipJudge()
        inf = _StubInferencer(
            output_guardrail_inferencer=judge,
            max_retry=3,
        )
        inf.set_responses(["bad narration", "good complete plan"])
        result = inf.infer("input")
        self.assertEqual(str(result), "good complete plan")
        self.assertEqual(inf._call_count, 2)


@attrs
class _CountingJudge(_StubJudge):
    calls = attrib(default=0, init=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.calls += 1
        return super()._infer(inference_input, inference_config, **kwargs)


class TestGuardrailSingleAttempt(unittest.TestCase):
    """``fallback_mode=NEVER`` with the default ``max_retry=1`` makes one
    attempt; sync ``infer`` still judges it, as ``ainfer`` does."""

    MODES = {
        "sync": lambda inf: inf.infer("input"),
        "async": lambda inf: _run(inf.ainfer("input")),
    }

    def _single_attempt(self, verdict):
        judge = _CountingJudge(verdict=verdict)
        inf = _StubInferencer(
            output_guardrail_inferencer=judge, fallback_mode=FallbackMode.NEVER
        )
        inf.set_responses(["draft"])
        return inf, judge

    def test_pass_is_judged_once(self):
        for mode, call in self.MODES.items():
            with self.subTest(mode=mode):
                inf, judge = self._single_attempt("PASS")
                self.assertEqual(str(call(inf)), "draft")
                self.assertEqual(judge.calls, 1)
                self.assertEqual(inf._call_count, 1)

    def test_retry_verdict_raises(self):
        for mode, call in self.MODES.items():
            with self.subTest(mode=mode):
                inf, _ = self._single_attempt("RETRY: narration only")
                with self.assertRaises(OutputValidationExhaustedError):
                    call(inf)
                self.assertEqual(inf._call_count, 1)

    def test_update_verdict_is_published_degraded(self):
        for mode, call in self.MODES.items():
            with self.subTest(mode=mode):
                inf, _ = self._single_attempt("UPDATE: expand it")
                with patch.object(inf, "log_warning") as warned:
                    self.assertEqual(str(call(inf)), "draft")
                events = [
                    c.args[0]
                    for c in warned.call_args_list
                    if isinstance(c.args[0], dict)
                    and c.args[0].get("event") == "DEGRADED_OUTPUT"
                ]
                self.assertEqual(len(events), 1)


class TestGuardrailReasoningFirstVerdict(unittest.TestCase):
    """End-to-end: the recovery judge reasons first and states its verdict last.
    The guardrail must route on the tail verdict, through the full infer flow."""

    def test_reasoning_first_pass_accepted(self):
        """A reasoning-first reply ending in PASS is ACCEPTED — no recovery. This
        is the real production misparse that discarded a valid deliverable."""
        judge = _StubJudge(
            verdict=(
                "RETRY/UPDATE/PASS decision requires checking the task's "
                "deliverable path — it specifies one, so I judge the file.\n\n"
                "The file is a fully-formed, on-topic artifact with a well-formed "
                "proposal_index fence; no corruption.\n\nVerdict: PASS"
            )
        )
        inf = _StubInferencer(output_guardrail_inferencer=judge)
        inf.set_responses(["the good deliverable"])
        result = inf.infer("input")
        self.assertEqual(str(result), "the good deliverable")
        self.assertEqual(inf._call_count, 1)

    def test_reasoning_first_retry_triggers_recovery(self):
        """A reasoning-first reply ending in RETRY still routes to recovery."""
        call_count = [0]

        @attrs
        class _FlipJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    return (
                        "The transcript only narrates and never writes the "
                        "file.\n\nVerdict: RETRY: nothing was produced."
                    )
                return "Reasoning: the deliverable is present.\nVerdict: PASS"

        judge = _FlipJudge()
        inf = _StubInferencer(output_guardrail_inferencer=judge, max_retry=3)
        inf.set_responses(["narration only", "the real deliverable"])
        result = inf.infer("input")
        self.assertEqual(str(result), "the real deliverable")
        self.assertEqual(inf._call_count, 2)


class TestGuardrailFailOpen(unittest.TestCase):
    def test_judge_exception_accepts_output(self):
        """If the judge itself crashes, fail-open: accept the output."""

        @attrs
        class _CrashingJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                raise RuntimeError("judge crashed")

        judge = _CrashingJudge()
        inf = _StubInferencer(output_guardrail_inferencer=judge)
        inf.set_responses(["some output"])
        result = inf.infer("input")
        self.assertEqual(str(result), "some output")


class TestParseGuardrailVerdict(unittest.TestCase):
    def test_pass_variants(self):
        inf = _StubInferencer()
        self.assertTrue(inf._parse_guardrail_verdict("PASS"))
        self.assertTrue(inf._parse_guardrail_verdict("pass"))
        self.assertTrue(inf._parse_guardrail_verdict("PASS - looks good"))

    def test_handler_routing(self):
        """Verdict returns handler name strings (Option C: retry / update)."""
        inf = _StubInferencer()
        # New canonical verdicts.
        self.assertEqual(
            inf._parse_guardrail_verdict("RETRY: completely wrong"), "retry"
        )
        self.assertEqual(inf._parse_guardrail_verdict("UPDATE: incomplete"), "update")
        # Legacy verdicts remain mapped (backward-compat; never silent-accept).
        self.assertEqual(
            inf._parse_guardrail_verdict("RESTART: completely wrong"), "retry"
        )
        self.assertEqual(
            inf._parse_guardrail_verdict("RETRY_WITH_REFERENCE: incomplete"), "retry"
        )
        self.assertEqual(inf._parse_guardrail_verdict("CONTINUE: truncated"), "update")
        self.assertEqual(inf._parse_guardrail_verdict("FAIL: generic"), "retry")

    def test_overridable(self):
        """Subclass can override the verdict parser."""

        @attrs
        class _CustomVerdict(_StubInferencer):
            def _parse_guardrail_verdict(self, judge_response):
                return (
                    "approved" if "APPROVED" in str(judge_response).upper() else "retry"
                )

        inf = _CustomVerdict()
        self.assertEqual(inf._parse_guardrail_verdict("APPROVED: ok"), "approved")
        self.assertEqual(inf._parse_guardrail_verdict("REJECTED: bad"), "retry")

    def test_quote_or_fence_wrapped_verdict_not_silent_pass(self):
        """C2: a judge that wraps the verdict in quotes/backticks must still route,
        not fall through to the silent-PASS default (worst failure for a guardrail)."""
        inf = _StubInferencer()
        self.assertEqual(inf._parse_guardrail_verdict('"UPDATE: x"'), "update")
        self.assertEqual(inf._parse_guardrail_verdict("`RETRY: y`"), "retry")
        self.assertEqual(inf._parse_guardrail_verdict("'PASS'"), True)

    def test_unrecognized_verdict_warns_and_accepts(self):
        """C2: a truly unrecognized verdict still accepts (back-compat) but is logged
        (a guardrail blind spot worth surfacing), not silently swallowed."""
        inf = _StubInferencer()
        logger_name = "agent_foundation.common.inferencers.inferencer_base"
        with self.assertLogs(logger_name, level="WARNING") as log:
            self.assertTrue(inf._parse_guardrail_verdict("mumble mumble"))
        self.assertTrue(any("unrecognized" in m.lower() for m in log.output))

    def test_reasoning_first_verdict_read_from_tail(self):
        """The recovery judge reasons FIRST and states its verdict LAST, so the
        verdict is read from the tail, not the head. A reasoning preamble that
        merely opens with a keyword must not be misread — regression for the real
        production misparse (a PASS reply that began 'RETRY/UPDATE/PASS decision
        ...' was routed as retry, discarding a valid deliverable)."""
        inf = _StubInferencer()
        real_pass = (
            "RETRY/UPDATE/PASS decision requires checking whether the task "
            "specifies a deliverable file path — it does, so I judge the file.\n\n"
            "The deliverable is a fully-formed, on-topic artifact that closes with "
            "a well-formed proposal_index JSON fence. No truncation or corruption."
            "\n\nPASS"
        )
        self.assertTrue(inf._parse_guardrail_verdict(real_pass))
        real_retry = (
            "The task requires writing the deliverable, but the transcript is "
            "narration-only and no file exists.\n\nRETRY: nothing was produced."
        )
        self.assertEqual(inf._parse_guardrail_verdict(real_retry), "retry")
        real_update = (
            "Substantive through Section 4, but the proposal_index fence is "
            "missing and must be added.\n\nUPDATE: add the proposal_index fence."
        )
        self.assertEqual(inf._parse_guardrail_verdict(real_update), "update")

    def test_verdict_label_read_from_final_line(self):
        """New judge.jinja2 contract: reasoning first, then a final
        'Verdict: <verdict>' line. The parser keys on the LAST 'Verdict:' label
        and tolerates the historical 'Vertdict' misspelling."""
        inf = _StubInferencer()
        self.assertTrue(
            inf._parse_guardrail_verdict(
                "Reasoning: file present, fence well-formed.\nVerdict: PASS"
            )
        )
        self.assertEqual(
            inf._parse_guardrail_verdict(
                "Reasoning: substantive but the fence is missing.\n"
                "Verdict: UPDATE: add the proposal_index fence."
            ),
            "update",
        )
        self.assertEqual(
            inf._parse_guardrail_verdict(
                "Reasoning: narration-only, no file.\nVerdict: RETRY: start fresh."
            ),
            "retry",
        )
        # Reasoning that name-drops keywords must not fool the tail reader.
        self.assertTrue(
            inf._parse_guardrail_verdict(
                "Reasoning: not a RETRY and not an UPDATE case.\nVerdict: PASS"
            )
        )
        # Tolerate the 'Vertdict' misspelling (label typo only).
        self.assertEqual(
            inf._parse_guardrail_verdict(
                "Reasoning: incomplete work.\nVertdict: UPDATE: finish it."
            ),
            "update",
        )


class TestGuardrailOutputText(unittest.TestCase):
    """Fix 1 (keystone): the guardrail judges the written DELIVERABLE FILE, not the
    (truncatable) stdout transcript. ``_guardrail_output_text`` sources the file when
    an ``output_path`` deliverable exists, else falls back to the response transcript.
    """

    def _make(self, op_path):
        @attrs
        class _OutputPathInferencer(_StubInferencer):
            _op_path = attrib(default=None)

            def resolve_output_path(self):
                return self._op_path

        return _OutputPathInferencer(op_path=op_path)

    def test_reads_deliverable_file_when_present(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Real deliverable\nsubstantive content\n")
            path = f.name
        try:
            inf = self._make(path)
            out = inf._guardrail_output_text({"output": "TRUNCATED TRANSCRIPT"})
            self.assertIn("substantive content", out)
            self.assertIn(path, out)  # "# Agent-written deliverable (<path>):"
            self.assertNotIn("TRUNCATED TRANSCRIPT", out)  # file replaced transcript
        finally:
            os.unlink(path)

    def test_falls_back_to_transcript_when_no_file(self):
        inf = self._make(None)
        self.assertEqual(
            inf._guardrail_output_text({"output": "the transcript"}), "the transcript"
        )
        # Also when the path is set but the file does not exist.
        inf2 = self._make("/nonexistent/path/output.md")
        self.assertEqual(inf2._guardrail_output_text({"output": "xcript"}), "xcript")

    def test_falls_back_when_file_empty(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            path = f.name  # empty file (size 0)
        try:
            inf = self._make(path)
            self.assertEqual(
                inf._guardrail_output_text({"output": "fallback text"}),
                "fallback text",
            )
        finally:
            os.unlink(path)

    def test_feeds_full_large_file_no_truncation(self):
        import tempfile

        # A large deliverable (>40 KB) must be fed to the judge IN FULL — no head+tail
        # elision — so a large trailing fence (e.g. a 28-84 KB proposal_index) is never
        # mangled mid-JSON into a false "malformed" reject. Let the judge model handle
        # the length.
        big = "A" * 32768 + "B" * 20000 + "Z" * 8192
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(big)
            path = f.name
        try:
            inf = self._make(path)
            out = inf._guardrail_output_text({"output": "t"})
            self.assertNotIn("elided", out)  # NO truncation marker
            self.assertIn(big, out)  # the COMPLETE content is included
            self.assertIn("B" * 100, out)  # the middle (previously elided) is present
        finally:
            os.unlink(path)

    def test_judge_receives_file_not_transcript(self):
        """End-to-end wiring: with a truncated transcript but a substantive file, the
        judge's prompt carries the FILE content (so it can PASS), not the transcript.
        """
        import tempfile

        recorded = []

        @attrs
        class _RecordingJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                recorded.append(str(inp))
                return "PASS"

        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("SUBSTANTIVE_DELIVERABLE_MARKER content\n")
            path = f.name
        try:
            inf = self._make(path)
            inf.output_guardrail_inferencer = _RecordingJudge()
            inf.set_responses(["TRUNCATED_MID_TOOL_SNIPPET"])
            inf.infer("seed input")
            self.assertTrue(recorded, "judge should have been invoked")
            self.assertIn("SUBSTANTIVE_DELIVERABLE_MARKER", recorded[0])
            self.assertNotIn("TRUNCATED_MID_TOOL_SNIPPET", recorded[0])
        finally:
            os.unlink(path)

    # ── Channel labeling: a LOCAL agent's absent/empty file is stated as a NEUTRAL
    # FACT, never as an "expected deliverable" verdict — the framework cannot know
    # whether the input asked for a file (``output_path`` is blanket-cascaded to
    # every leaf), so it states what it knows and leaves the call to the judge.
    # Gated on has_local_access so a no-local agent (deliverable inline in
    # <Response>) is never remarked on at all. ──

    _VERDICT_WORDS = ("WAS NOT WRITTEN", "EXPECTED DELIVERABLE")

    def _assert_no_verdict(self, out):
        for word in self._VERDICT_WORDS:
            self.assertNotIn(word, out, f"guardrail must not assert intent: {word!r}")

    def test_local_missing_file_stated_as_neutral_fact(self):
        inf = self._make("/nonexistent/path/output.md")
        inf.has_local_access = True
        out = inf._guardrail_output_text({"output": "the transcript"})
        self._assert_no_verdict(out)
        self.assertIn("No file exists at", out)
        self.assertIn("/nonexistent/path/output.md", out)
        # The disclaimer is what stops the judge reading absence as a failure.
        self.assertIn("does NOT imply", out)
        self.assertIn("the transcript", out)  # transcript still appended for context

    def test_local_empty_file_stated_as_neutral_fact(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            path = f.name  # empty (size 0)
        try:
            inf = self._make(path)
            inf.has_local_access = True
            out = inf._guardrail_output_text({"output": "fallback text"})
            self._assert_no_verdict(out)
            self.assertIn("No file exists at", out)
            self.assertIn("fallback text", out)
        finally:
            os.unlink(path)

    def test_local_present_file_feeds_deliverable(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("real content\n")
            path = f.name
        try:
            inf = self._make(path)
            inf.has_local_access = True
            out = inf._guardrail_output_text({"output": "t"})
            self.assertIn("real content", out)
            self._assert_no_verdict(out)
            self.assertNotIn("No file exists at", out)
        finally:
            os.unlink(path)

    def test_no_local_missing_file_not_remarked(self):
        # A no-local agent's deliverable is legitimately inline in <Response> (its
        # file is materialized later by _finalize_output), so remarking on the
        # absent file would be pure noise: the transcript is returned bare.
        inf = self._make("/nonexistent/path/output.md")
        self.assertFalse(inf.has_local_access)  # default False
        out = inf._guardrail_output_text({"output": "inline deliverable"})
        self._assert_no_verdict(out)
        self.assertEqual(out, "inline deliverable")


class TestJudgePromptRendering(unittest.TestCase):
    def test_render_guardrail_prompt_contains_input_and_output(self):
        inf = _StubInferencer()
        prompt = _render_under(inf, "original task request", "the agent's output text")
        self.assertIn("original task request", prompt)
        self.assertIn("the agent's output text", prompt)
        self.assertIn("PASS", prompt)
        self.assertIn("RETRY", prompt)
        self.assertIn("UPDATE", prompt)


class TestGuardrailUnifiedRendering(unittest.TestCase):
    """The guardrail prompt should render via the inferencer's OWN template_manager
    (unified with continue/retry_with_reference recovery prompts), not only the
    standalone recovery TemplateManager."""

    def test_renders_recovery_judge_via_own_template_manager(self):
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
            ClaudeCodeCliInferencer,
        )
        from agent_foundation.common.inferencers.constants.paths import (
            DEFAULT_PROMPT_TEMPLATES_DIR,
        )
        from rich_python_utils.string_utils.formatting.template_manager import (
            TemplateManager,
        )

        tm = TemplateManager(
            templates=[str(DEFAULT_PROMPT_TEMPLATES_DIR)], active_template_type="main"
        )
        inf = ClaudeCodeCliInferencer(
            template_manager=tm, template_root_space="plan", template_key="initial"
        )
        rendered = _render_under(
            inf, "the original task", {"output": "PARTIAL_OUTPUT_MARKER"}
        )

        # Rendered the recovery/judge template (verdict contract present)…
        self.assertIn("sanity judge", rendered.lower())
        self.assertIn("PASS", rendered)
        self.assertIn("RETRY", rendered)
        self.assertIn("UPDATE", rendered)
        # …with the real input + output woven in…
        self.assertIn("the original task", rendered)
        self.assertIn("PARTIAL_OUTPUT_MARKER", rendered)
        # …and did NOT double-wrap with the planning template.
        self.assertNotIn("You are tasked with creating artifacts", rendered)


class TestGuardrailNoDoubleWrap(unittest.TestCase):
    """The guardrail prompt is already fully rendered (recovery/judge). A judge
    that carries its own (planning) template must NOT wrap it a second time."""

    def _judge(self, **kwargs):
        # A judge that "carries a template" — its _render_prompt would wrap the
        # prompt in a planning template if the judge rendered its input.
        @attrs
        class _TemplatedJudge(InferencerBase):
            _seen_prompt = attrib(default=None, init=False)

            def _render_prompt(self, inference_input, extra_feed=None):
                return f"You are tasked with creating artifacts: [{inference_input}]"

            def _infer(self, inference_input, inference_config=None, **kwargs):
                # Record the prompt the judge actually executes.
                object.__setattr__(self, "_seen_prompt", str(inference_input))
                return "PASS"

        judge = _TemplatedJudge(**kwargs)
        judge.template_manager = object()
        return judge

    def test_templated_judge_runs_the_prerendered_prompt_unmodified(self):
        judge = self._judge()
        template_manager = judge.template_manager

        inf = _StubInferencer(output_guardrail_inferencer=judge)
        inf.set_responses(["the output"])
        result = inf.infer("input")
        self.assertEqual(str(result), "the output")

        # The judge executed the fully-rendered judge prompt verbatim (no second
        # wrap) — and its definition was not modified to achieve that (B22).
        self.assertIs(judge.template_manager, template_manager)
        self.assertIsNotNone(judge._seen_prompt)
        self.assertIn("sanity judge", judge._seen_prompt.lower())
        self.assertNotIn("You are tasked with creating artifacts", judge._seen_prompt)

    def test_judge_input_preprocessor_is_applied_once_before_the_call(self):
        judge = self._judge(input_preprocessor=lambda prompt: f"PRE<{prompt}>")
        inf = _StubInferencer(output_guardrail_inferencer=judge)
        inf.set_responses(["the output"])
        inf.infer("input")
        self.assertTrue(judge._seen_prompt.startswith("PRE<"))
        self.assertFalse(judge._seen_prompt.startswith("PRE<PRE<"))
        self.assertIn("sanity judge", judge._seen_prompt.lower())


# =============================================================================
# v5 Fix #1 — rendered prompt reaches the judge via ``_fallback_state`` ContextVar
# (no new instance state); ``_guardrail_input_text`` is overridable.
# =============================================================================


class TestGuardrailFix1_RenderedInputViaContextVar(unittest.TestCase):
    def test_render_pulls_from_fallback_state(self):
        """The judge's input is the rendered prompt published in ``_fallback_state``."""
        inf = _StubInferencer()
        prompt = _render_under(inf, "POST_RENDER_FULL_PROMPT", "agent output here")
        self.assertIn("POST_RENDER_FULL_PROMPT", prompt)
        # Output still flows through.
        self.assertIn("agent output here", prompt)

    def test_render_never_uses_an_earlier_calls_input(self):
        """With no ``_fallback_state`` active, a finished call's input must not
        leak into the judge prompt."""
        inf = _StubInferencer()
        inf.set_responses(["earlier output"])
        inf.infer("EARLIER_CALL_INPUT")
        self.assertIsNone(_current_fallback_state.get(None))
        prompt = inf._render_guardrail_prompt("agent output")
        self.assertNotIn("EARLIER_CALL_INPUT", prompt)
        self.assertIn("agent output", prompt)

    def test_render_falls_back_to_empty_when_nothing_set(self):
        """No ``_fallback_state`` active → empty input, no crash."""
        inf = _StubInferencer()
        self.assertIsNone(_current_fallback_state.get(None))
        prompt = inf._render_guardrail_prompt("only the output is set")
        self.assertIn("only the output is set", prompt)
        # Template still rendered — verdict contract present.
        self.assertIn("PASS", prompt)

    def test_guardrail_input_text_default_is_identity(self):
        """The default ``_guardrail_input_text`` returns the rendered text unchanged.

        This is the contract that prevents the false-RESTART bug: extracting
        only ``<UserRequest>`` would strip the ``output_path`` instruction the
        judge needs to verify deliverables (per ``recovery/judge.jinja2:3``).
        """
        inf = _StubInferencer()
        full_rendered = (
            "<UserRequest>do thing X</UserRequest>\n"
            "## Output Requirements\nWrite to /tmp/x.md"
        )
        self.assertEqual(inf._guardrail_input_text(full_rendered), full_rendered)

    def test_guardrail_input_text_overridable(self):
        """Subclasses can override to shape what the judge sees — opt-in trim."""

        @attrs
        class _TrimInferencer(_StubInferencer):
            def _guardrail_input_text(self, rendered):
                # Toy extractor: drop everything after the marker.
                marker = "## Output Requirements"
                return rendered.split(marker, 1)[0].strip()

        inf = _TrimInferencer()
        full = (
            "<UserRequest>do thing X</UserRequest>\n"
            "## Output Requirements\nWrite to /tmp/x.md"
        )
        fs = {"rendered_input": full}
        token = _current_fallback_state.set(fs)
        try:
            prompt = inf._render_guardrail_prompt("agent_out")
        finally:
            _current_fallback_state.reset(token)
        # Override applied — output_path trimmed.
        self.assertIn("do thing X", prompt)
        self.assertNotIn("/tmp/x.md", prompt)

    def test_end_to_end_judge_sees_rendered_input(self):
        """End-to-end: a templated leaf wraps the raw input; the judge sees the wrap."""

        @attrs
        class _TemplatedStub(_StubInferencer):
            # Pretend we have a template manager so _render_prompt runs.
            def _render_prompt(self, inference_input, *args, **kwargs):
                return f"TEMPLATE_PREAMBLE\n<UserRequest>{inference_input}</UserRequest>\n## Output Requirements\nWrite to /tmp/foo.md"

            # Required for the _render_prompt seam to actually fire.
            def supports_prompt_rendering(self):
                return True

        seen = {}

        @attrs
        class _CapturingJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                seen["prompt"] = str(inp)
                return "PASS"

        judge = _CapturingJudge()
        inf = _TemplatedStub(output_guardrail_inferencer=judge)
        inf.set_responses(["the agent's output"])
        result = _run(inf.ainfer("RAW_USER_REQUEST"))
        self.assertEqual(str(result), "the agent's output")
        # Judge saw the RENDERED prompt (template wrapper + raw input + path).
        self.assertIn("TEMPLATE_PREAMBLE", seen["prompt"])
        self.assertIn("RAW_USER_REQUEST", seen["prompt"])
        self.assertIn("/tmp/foo.md", seen["prompt"])


# =============================================================================
# v5 Fix #2 — rejected responses are persisted (one InferenceResponse parts
# file per reject, named ``call_<cid>_reject<k>`` under verbose correlation).
# =============================================================================


class TestGuardrailFix2_PersistOnRejection(unittest.TestCase):
    def test_reject_calls_log_debug_with_parts_file_namer(self):
        """Verbose correlation ON: each rejection logs an InferenceResponse with
        a ``parts_file_namer`` that encodes ``call_<cid>_reject<k>``."""

        judge = _StubJudge(verdict="RESTART: nope")
        inf = _StubInferencer(
            output_guardrail_inferencer=judge,
            max_retry=2,
        )
        inf.set_responses(["bad1", "bad2", "bad3"])

        captured = []

        def _capture(item, name, **kw):
            captured.append((name, item, kw.get("parts_file_namer")))

        with (
            patch.dict(os.environ, {"RESEARCH_PROPOSE__VERBOSE_CORRELATION": "1"}),
            patch.object(inf, "log_debug", side_effect=_capture),
        ):
            with self.assertRaises(Exception):
                inf.infer("input")

        # Multiple InferenceResponse logs (one per rejection).
        responses = [c for c in captured if c[0] == "InferenceResponse"]
        self.assertGreaterEqual(len(responses), 2)
        # Each carries a parts_file_namer encoding the reject index.
        for i, (_name, item, namer) in enumerate(responses, start=1):
            self.assertIsNotNone(namer, "verbose corr should set parts_file_namer")
            named = namer(item)
            self.assertTrue(
                named.startswith("call_") and f"_reject{i}" in named,
                f"expected call_<cid>_reject{i}, got {named!r}",
            )

    def test_reject_persists_even_without_verbose_correlation(self):
        """Verbose correlation OFF: still persist (defensive default)."""
        judge = _StubJudge(verdict="RESTART: nope")
        inf = _StubInferencer(
            output_guardrail_inferencer=judge,
            max_retry=1,
        )
        inf.set_responses(["bad1", "bad2"])

        captured = []

        def _capture(item, name, **kw):
            captured.append((name, item, kw.get("parts_file_namer")))

        # Make sure env var is unset.
        env = {
            k: v
            for k, v in os.environ.items()
            if k != "RESEARCH_PROPOSE__VERBOSE_CORRELATION"
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(inf, "log_debug", side_effect=_capture),
        ):
            with self.assertRaises(Exception):
                inf.infer("input")

        responses = [c for c in captured if c[0] == "InferenceResponse"]
        self.assertGreaterEqual(len(responses), 1)
        # No parts_file_namer when correlation is off — default naming.
        for _name, _item, namer in responses:
            self.assertIsNone(namer)

    def test_counter_in_fallback_state_increments(self):
        """The ``guardrail_reject_attempt`` counter bumps on each rejection."""
        # Drive _run_output_guardrail directly with a populated _fallback_state.
        judge = _StubJudge(verdict="RESTART: nope")
        inf = _StubInferencer(output_guardrail_inferencer=judge)
        fs = {"call_id": "deadbeef", "guardrail_reject_attempt": 0}
        token = _current_fallback_state.set(fs)
        try:
            with (
                patch.object(inf, "log_debug"),
                patch.dict(os.environ, {"RESEARCH_PROPOSE__VERBOSE_CORRELATION": "1"}),
            ):
                _run(inf._run_output_guardrail("first bad output"))
                self.assertEqual(fs["guardrail_reject_attempt"], 1)
                _run(inf._run_output_guardrail("second bad output"))
                self.assertEqual(fs["guardrail_reject_attempt"], 2)
        finally:
            _current_fallback_state.reset(token)


# =============================================================================
# Q2 — on rejection, the judge's <reason> (verdict text after the UPDATE:/RETRY:
# prefix) is stashed on the per-call _fallback_state as ``guardrail_reason`` for
# recovery/update.jinja2. Extraction is regex-free (``re`` is not imported here).
# =============================================================================


class TestGuardrailReasonStash(unittest.TestCase):
    def _reason_after(self, verdict, sync=False):
        judge = _StubJudge(verdict=verdict)
        inf = _StubInferencer(output_guardrail_inferencer=judge)
        fs = {"guardrail_reject_attempt": 0}
        token = _current_fallback_state.set(fs)
        try:
            with patch.object(inf, "log_debug"):
                if sync:
                    inf._run_output_guardrail_sync("bad output")
                else:
                    _run(inf._run_output_guardrail("bad output"))
        finally:
            _current_fallback_state.reset(token)
        return fs.get("guardrail_reason")

    def test_update_colon_reason(self):
        self.assertEqual(
            self._reason_after("UPDATE: missing wrapper"), "missing wrapper"
        )

    def test_update_reason_preserves_inner_colon(self):
        # Only the leading prefix + separator chars are stripped; inner colons stay.
        self.assertEqual(self._reason_after("UPDATE: x: y"), "x: y")

    def test_update_space_separator_no_colon(self):
        self.assertEqual(self._reason_after("UPDATE incomplete"), "incomplete")

    def test_bare_update_yields_none(self):
        self.assertIsNone(self._reason_after("UPDATE"))

    def test_retry_reason_captured(self):
        self.assertEqual(self._reason_after("RETRY: off topic"), "off topic")

    def test_sync_path_stashes_reason(self):
        self.assertEqual(self._reason_after("UPDATE: fix it", sync=True), "fix it")

    def test_extract_helper_direct(self):
        """Extraction is an overridable helper (``_extract_guardrail_reason``), shared
        by the async + sync guardrail paths (DRY, no duplicated inline logic)."""
        inf = _StubInferencer()
        self.assertEqual(inf._extract_guardrail_reason("UPDATE: x: y"), "x: y")
        self.assertEqual(inf._extract_guardrail_reason("RETRY off topic"), "off topic")
        self.assertIsNone(inf._extract_guardrail_reason("UPDATE"))
        self.assertIsNone(inf._extract_guardrail_reason(""))

    def test_extract_reason_from_labeled_verdict_line(self):
        """Reason extraction reads the same tail segment as the parser, so the
        concrete fix text after the keyword is captured even when the judge
        reasons first (feeds ``recovery/update.jinja2``'s ``{{ reason }}``)."""
        inf = _StubInferencer()
        self.assertEqual(
            inf._extract_guardrail_reason(
                "Reasoning: the fence is missing.\n"
                "Verdict: UPDATE: add the proposal_index fence."
            ),
            "add the proposal_index fence.",
        )
        self.assertEqual(
            inf._extract_guardrail_reason(
                "Reasoning: no file written.\nVerdict: RETRY: write the file."
            ),
            "write the file.",
        )


# =============================================================================
# v5 Fix #4 — internal recovery re-runs the rendered prompt (closure-local),
# not the raw pre-render seed. External fallback inferencers still get raw.
# =============================================================================


class TestGuardrailFix4_RecoveryReceivesRenderedPrompt(unittest.TestCase):
    def test_recovery_receives_rendered_not_raw(self):
        """After a rejection, the recovery call to ``_ainfer_recovery`` passes
        the RENDERED prompt (post-``_render_prompt``), not the raw input."""

        seen_recovery_inputs = []

        @attrs
        class _TemplatedRecoveryStub(_StubInferencer):
            def _render_prompt(self, inference_input, *args, **kwargs):
                return f"RENDERED({inference_input})"

            def supports_prompt_rendering(self):
                return True

            async def _ainfer_recovery(
                self,
                inference_input,
                last_exception,
                last_partial_output,
                inference_config=None,
                **kwargs,
            ):
                # Recovery override — capture and return a final good output.
                seen_recovery_inputs.append(inference_input)
                return "recovered good output"

        # Judge: reject first attempt; pass on recovery.
        verdicts = ["RESTART: try again", "PASS"]
        idx = [0]

        @attrs
        class _SequencedJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                v = verdicts[idx[0]]
                idx[0] += 1
                return v

        judge = _SequencedJudge()
        inf = _TemplatedRecoveryStub(
            output_guardrail_inferencer=judge,
            max_retry=2,
        )
        inf.set_responses(["first attempt output"])
        result = _run(inf.ainfer("RAW_REQUEST"))
        self.assertEqual(str(result), "recovered good output")
        # Recovery was called with the RENDERED prompt, NOT the raw seed.
        self.assertEqual(len(seen_recovery_inputs), 1)
        self.assertIn("RENDERED(", seen_recovery_inputs[0])
        self.assertIn("RAW_REQUEST", seen_recovery_inputs[0])

    def test_external_fallback_still_receives_original(self):
        """External fallback inferencers (via ``fallback_inferencer``) keep
        getting the raw ``original_input`` so they can re-render with their
        own template. Internal recovery's rendered prompt does NOT leak."""

        seen_external = []

        @attrs
        class _ExternalFallback(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                seen_external.append(str(inp))
                return "external recovery output"

        ext = _ExternalFallback()

        @attrs
        class _TemplatedStub(_StubInferencer):
            def _render_prompt(self, inference_input, *args, **kwargs):
                return f"RENDERED({inference_input})"

            def supports_prompt_rendering(self):
                return True

        # Judge REJECTS the primary/internal-recovery outputs but ACCEPTS the
        # external fallback's output — so the external fallback deterministically
        # fires and the run SUCCEEDS (this removes the original vacuous-pass:
        # the old test's assertions were gated behind ``if seen_external:`` with
        # no guarantee it ever ran).
        @attrs
        class _PassExternalJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                return (
                    "PASS"
                    if "external recovery output" in str(inp)
                    else "RESTART: nope"
                )

        inf = _TemplatedStub(
            output_guardrail_inferencer=_PassExternalJudge(),
            max_retry=1,
            fallback_inferencer=ext,
        )
        inf.set_responses(["primary out 1", "primary out 2"])
        result = _run(inf.ainfer("RAW_REQUEST"))
        # The external fallback produced the accepted output.
        self.assertEqual(str(result), "external recovery output")
        # It MUST have actually fired — no vacuous pass (the bug being fixed).
        self.assertTrue(seen_external, "external fallback should have fired")
        # …and it received the RAW original input, NOT the internal RENDERED prompt.
        for got in seen_external:
            self.assertNotIn("RENDERED(", got, "external got rendered!")
            self.assertIn("RAW_REQUEST", got)


# =============================================================================
# v5 Fix #5 — ``HopelessOutputError`` is terminal (non-retryable). Fix #2's
# persist runs BEFORE the raise so the rejected response is still captured.
# =============================================================================


class TestGuardrailFix5_HopelessTerminal(unittest.TestCase):
    def test_hopeless_propagates_terminally_async(self):
        """N identical empty outputs → ``HopelessOutputError`` short-circuits
        immediately (does NOT spin to ``max_retry``)."""
        judge = _StubJudge(verdict="RESTART: empty narration")
        inf = _StubInferencer(
            output_guardrail_inferencer=judge,
            max_retry=10,  # large budget — fix must short-circuit before this
            guardrail_empty_fail_fast_n=2,  # trip on second identical empty
        )
        # All responses identical & empty-shaped (< 200 chars normalized).
        inf.set_responses([""] * 10)
        with self.assertRaises(HopelessOutputError):
            _run(inf.ainfer("input"))
        # Should have stopped well before max_retry (2 attempts ~ trigger fail-fast)
        self.assertLessEqual(inf._call_count, 4)

    def test_hopeless_propagates_terminally_sync(self):
        judge = _StubJudge(verdict="RESTART: empty")
        inf = _StubInferencer(
            output_guardrail_inferencer=judge,
            max_retry=10,
            guardrail_empty_fail_fast_n=2,
        )
        inf.set_responses([""] * 10)
        with self.assertRaises(HopelessOutputError):
            inf.infer("input")
        self.assertLessEqual(inf._call_count, 4)

    def test_hopeless_response_persisted_before_raise(self):
        """Fix #2 ordering — the offending response is persisted BEFORE the
        ``HopelessOutputError`` is raised."""
        judge = _StubJudge(verdict="RESTART: empty narration")
        inf = _StubInferencer(
            output_guardrail_inferencer=judge,
            max_retry=10,
            guardrail_empty_fail_fast_n=2,
        )
        inf.set_responses([""] * 10)

        captured = []

        def _capture(item, name, **kw):
            captured.append(name)

        with patch.object(inf, "log_debug", side_effect=_capture):
            with self.assertRaises(HopelessOutputError):
                inf.infer("input")
        # At least one InferenceResponse persist happened (Fix #2's persist
        # ran before the raise). If Fix #5 were applied without Fix #2's
        # ordering guarantee, this would fail.
        self.assertGreaterEqual(
            sum(1 for n in captured if n == "InferenceResponse"),
            1,
            f"expected at least one reject persist, got logs: {captured}",
        )


class TestGuardrailFailFastWindowPerCall(unittest.TestCase):
    """B5 — the empty-fingerprint window spans the retries of one call and is
    never carried into the next call on the same instance."""

    def _leaf(self):
        """Each call's empties stay below ``n``; two calls together reach it."""
        inf = _StubInferencer(
            output_guardrail_inferencer=_StubJudge(verdict="RESTART: empty"),
            max_retry=2,
            fallback_mode=FallbackMode.NEVER,
            guardrail_empty_fail_fast_n=4,
        )
        inf.set_responses([""] * 8)
        return inf

    def test_previous_call_empty_not_counted_sync(self):
        inf = self._leaf()
        with self.assertRaises(OutputValidationExhaustedError):
            inf.infer("input")
        per_call = inf._call_count
        self.assertLess(per_call, 4)
        with self.assertRaises(OutputValidationExhaustedError):
            inf.infer("input")
        self.assertEqual(inf._call_count, 2 * per_call)

    def test_previous_call_empty_not_counted_async(self):
        inf = self._leaf()
        with self.assertRaises(OutputValidationExhaustedError):
            _run(inf.ainfer("input"))
        per_call = inf._call_count
        self.assertLess(per_call, 4)
        with self.assertRaises(OutputValidationExhaustedError):
            _run(inf.ainfer("input"))
        self.assertEqual(inf._call_count, 2 * per_call)

    def test_window_lives_in_fallback_state(self):
        inf = _StubInferencer(
            output_guardrail_inferencer=_StubJudge(verdict="RESTART: empty"),
            guardrail_empty_fail_fast_n=2,
        )
        before = dict(vars(inf))
        with patch.object(inf, "log_debug"):
            for _ in range(2):
                fs = {}
                token = _current_fallback_state.set(fs)
                try:
                    self.assertEqual(inf._run_output_guardrail_sync(""), "retry")
                    self.assertEqual(fs["guardrail_empty_fingerprints"], [""])
                finally:
                    _current_fallback_state.reset(token)
        self.assertEqual(vars(inf).keys(), before.keys())


# =============================================================================
# v5 Fix #1 / #4 — REVIEWER-context coverage: the render seam's ``extra_feed``
# branch (a reviewer renders its prompt from seed + extra_feed). The judge AND
# the recovery must see the FULL rendered prompt (incl. extra_feed), not the
# bare pre-render seed. This exercises the exact production path the earlier
# tests missed — none used ``extra_feed`` (the reviewer bug's root: the seed and
# the rendered prompt genuinely diverge only when extra_feed is present).
# =============================================================================


@attrs
class _ReviewRenderStub(_StubInferencer):
    """A leaf whose ``_render_prompt`` weaves BOTH the raw input and the
    ``extra_feed`` (the artifact under review) into a review prompt — mirroring
    how the real reviewer renders ``state["inference_input"]`` + the review
    feed. The bare seed and ``rendered_input`` (the full review prompt)
    DIVERGE, which is what makes the Fix #1/#4 assertions revert-sensitive."""

    def _render_prompt(self, inference_input, extra_feed=None, **kwargs):
        artifact = ""
        if isinstance(extra_feed, dict):
            artifact = str(extra_feed.get("artifact", extra_feed))
        elif extra_feed is not None:
            artifact = str(extra_feed)
        return (
            "REVIEW_PREAMBLE: You are reviewing artifacts.\n"
            f"<UserRequest>{inference_input}</UserRequest>\n"
            f"<ArtifactUnderReview>{artifact}</ArtifactUnderReview>\n"
            "Now start your review."
        )

    def supports_prompt_rendering(self):
        return True


class TestGuardrailFix1_ReviewerExtraFeed(unittest.TestCase):
    """Fix #1 in the reviewer path: the judge must see the rendered REVIEW
    prompt INCLUDING the ``extra_feed`` artifact — not the bare seed."""

    def test_judge_sees_extra_feed_rendered_prompt(self):
        seen = {}

        @attrs
        class _CapturingJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                seen["prompt"] = str(inp)
                return "PASS"

        judge = _CapturingJudge()
        inf = _ReviewRenderStub(output_guardrail_inferencer=judge)
        inf.set_responses(["a structured critique"])
        result = _run(
            inf.ainfer(
                "ORIGINAL_SEED_TASK",
                extra_feed={"artifact": "ARTIFACT_UNDER_REVIEW_MARKER"},
            )
        )
        self.assertEqual(str(result), "a structured critique")
        jp = seen["prompt"]
        # The judge saw the RENDERED review prompt: review framing + the
        # extra_feed artifact. NEITHER is in the bare seed, so a revert to
        # judging the seed fails these.
        self.assertIn("REVIEW_PREAMBLE", jp)
        self.assertIn("ARTIFACT_UNDER_REVIEW_MARKER", jp)
        self.assertIn("Now start your review", jp)
        self.assertIn("ORIGINAL_SEED_TASK", jp)  # seed embedded within the render


class TestGuardrailFix4_ReviewerExtraFeedRecovery(unittest.TestCase):
    """Fix #4 in the reviewer path: on a RESTART, recovery must re-issue the
    rendered REVIEW prompt (incl. extra_feed), NOT the bare seed — otherwise the
    reviewer 're-proposes' against the seed instead of re-reviewing."""

    def test_recovery_receives_extra_feed_rendered_prompt(self):
        seen_recovery = []

        @attrs
        class _RecoveringReviewStub(_ReviewRenderStub):
            async def _ainfer_recovery(
                self,
                inference_input,
                last_exception,
                last_partial_output,
                inference_config=None,
                **kwargs,
            ):
                seen_recovery.append(str(inference_input))
                return "recovered critique"

        verdicts = ["RESTART: looks like a review not a proposal", "PASS"]
        idx = [0]

        @attrs
        class _SequencedJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                v = verdicts[min(idx[0], len(verdicts) - 1)]
                idx[0] += 1
                return v

        inf = _RecoveringReviewStub(
            output_guardrail_inferencer=_SequencedJudge(),
            max_retry=2,
        )
        inf.set_responses(["first critique"])
        result = _run(
            inf.ainfer(
                "ORIGINAL_SEED_TASK",
                extra_feed={"artifact": "ARTIFACT_UNDER_REVIEW_MARKER"},
            )
        )
        self.assertEqual(str(result), "recovered critique")
        self.assertEqual(len(seen_recovery), 1)
        # Recovery re-issued the RENDERED review prompt (with extra_feed), NOT
        # the bare seed → reverting to raw ``inp`` (= original_input = seed)
        # drops these markers and fails.
        self.assertIn("REVIEW_PREAMBLE", seen_recovery[0])
        self.assertIn("ARTIFACT_UNDER_REVIEW_MARKER", seen_recovery[0])
        self.assertIn("ORIGINAL_SEED_TASK", seen_recovery[0])


# =============================================================================
# The output-guardrail must survive a role switch (MFDual promotes a flow leaf
# to the 'reviewer' role via ``switch_role``; the guardrail must stay active so
# the reviewer's output is still judged).
# =============================================================================


class TestGuardrailSurvivesRoleSwitch(unittest.TestCase):
    def test_output_guardrail_survives_switch_role(self):
        judge = _StubJudge(verdict="PASS")
        inf = _StubInferencer(output_guardrail_inferencer=judge)
        self.assertIs(inf.output_guardrail_inferencer, judge)
        # Promote to reviewer (as MFDual does). ``switch_role`` must not drop or
        # replace the guardrail.
        inf.switch_role("review_inferencer")
        self.assertIs(inf.output_guardrail_inferencer, judge)
        # …and it still functions after the switch.
        inf.set_responses(["good output"])
        self.assertEqual(str(inf.infer("x")), "good output")


# =============================================================================
# Gap 1 — the leaf-only output guardrail must not be attachable to an
# orchestrator. Layer A: fail-fast at construction. Layer B: runtime skip+warn
# for a post-construction assignment (the slot is public/mutable).
# =============================================================================


@attrs
class _StubOrchestrator(InferencerBase):
    """A minimal orchestrator stub: overriding ``_iter_child_inferencers`` makes
    ``_is_orchestrator()`` return True. Implements ``_infer`` so it is constructable."""

    def _iter_child_inferencers(self):
        return iter(())

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "orchestrator output"


class TestGuardrailForbiddenOnOrchestrator(unittest.TestCase):
    def test_construction_fails_fast_on_orchestrator_with_guardrail(self):
        # Sanity: the stub is detected as an orchestrator.
        self.assertTrue(_StubOrchestrator()._is_orchestrator())
        # Layer A: attaching a guardrail at construction raises.
        with self.assertRaises(ValueError) as cm:
            _StubOrchestrator(output_guardrail_inferencer=_StubJudge())
        self.assertIn("orchestrator", str(cm.exception).lower())

    def test_orchestrator_without_guardrail_constructs(self):
        # No guardrail → constructs fine (no false positives).
        inf = _StubOrchestrator()
        self.assertIsNone(inf.output_guardrail_inferencer)

    def test_leaf_with_guardrail_unaffected(self):
        # Regression: a leaf (overrides no child-iteration primitive) still works.
        self.assertFalse(_StubInferencer()._is_orchestrator())
        inf = _StubInferencer(output_guardrail_inferencer=_StubJudge(verdict="PASS"))
        inf.set_responses(["good output"])
        self.assertEqual(str(inf.infer("input")), "good output")

    def test_runtime_skip_and_warn_on_post_init_assignment(self):
        # Layer B: the slot is mutable — a guardrail assigned AFTER construction
        # bypasses Layer A. The runtime guard skips (accept) + warns, and must NOT
        # invoke the judge (on an orchestrator the judge's verdict re-raises).
        recorded = []

        @attrs
        class _RecordingJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                recorded.append(str(inp))
                return "RETRY: nope"

        inf = _StubOrchestrator()  # constructs cleanly (no guardrail)
        inf.output_guardrail_inferencer = _RecordingJudge()  # post-init mutation

        logger_name = "agent_foundation.common.inferencers.inferencer_base"
        with self.assertLogs(logger_name, level="WARNING") as log:
            self.assertTrue(_run(inf._run_output_guardrail("some output")))
            self.assertTrue(inf._run_output_guardrail_sync("some output"))
        self.assertEqual(recorded, [], "judge must NOT be invoked on an orchestrator")
        self.assertTrue(any("orchestrator" in m.lower() for m in log.output))


# =============================================================================
# Unified expected_extraction registry — emitter-side self-validation (warn-only,
# observability). source=output (deliverable file) / response (stdout); kind=
# content / control. Consumption of control fences stays in flow_parsers (unchanged).
# =============================================================================


class TestExpectedExtraction(unittest.TestCase):
    def _warn_msgs(self, inf, response=None):
        """Run _run_expected_extraction and capture formatted _logger.warning messages."""
        from agent_foundation.common.inferencers import inferencer_base as _ib

        msgs = []

        def _cap(fmt, *args, **kw):
            try:
                msgs.append(fmt % args)
            except Exception:
                msgs.append(str(fmt))

        with patch.object(_ib._logger, "warning", side_effect=_cap):
            inf._run_expected_extraction(response)
        return [m for m in msgs if "extraction_issue" in m]

    def _mk(self, op_path, specs):
        @attrs
        class _ExtractInferencer(_StubInferencer):
            _op_path = attrib(default=None)

            def resolve_output_path(self):
                return self._op_path

        inf = _ExtractInferencer(op_path=op_path)
        inf.expected_extraction = specs
        return inf

    _SPEC_OUT = [{"label": "proposal_index", "source": "output", "kind": "content"}]
    _SPEC_RESP = [{"label": "winner_pick", "source": "response", "kind": "control"}]

    def test_output_content_present_no_warn(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write('prose\n\n```json proposal_index\n{"total_count": 1}\n```\n')
            path = f.name
        try:
            self.assertEqual(self._warn_msgs(self._mk(path, self._SPEC_OUT)), [])
        finally:
            os.unlink(path)

    def test_output_content_missing_warns(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("prose only — no fence\n")
            path = f.name
        try:
            msgs = self._warn_msgs(self._mk(path, self._SPEC_OUT))
            self.assertTrue(
                msgs and "proposal_index" in msgs[0] and "content" in msgs[0]
            )
        finally:
            os.unlink(path)

    def test_response_control_present_no_warn(self):
        inf = self._mk(None, self._SPEC_RESP)
        resp = {"raw_output": 'x\n```json winner_pick\n{"winner_index": 0}\n```\n'}
        self.assertEqual(self._warn_msgs(inf, resp), [])

    def test_response_control_missing_warns(self):
        inf = self._mk(None, self._SPEC_RESP)
        msgs = self._warn_msgs(inf, {"raw_output": "no fence here"})
        self.assertTrue(msgs and "winner_pick" in msgs[0] and "control" in msgs[0])

    def test_output_source_no_file_skips(self):
        # No deliverable file on the output channel → nothing to check here (a missing
        # expected deliverable is surfaced by _guardrail_output_text, not double-warned).
        self.assertEqual(self._warn_msgs(self._mk(None, self._SPEC_OUT)), [])

    def test_none_registry_is_noop(self):
        self.assertEqual(self._warn_msgs(self._mk(None, None)), [])

    def test_content_fallback_to_source_suppresses_warn(self):
        # content fence absent, but fallback_to_source=True → the whole (present)
        # source text stands in as the content, so a missing fence is NOT flagged.
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("prose only — no fence\n")
            path = f.name
        try:
            specs = [
                {
                    "label": "proposal_index",
                    "source": "output",
                    "kind": "content",
                    "fallback_to_source": True,
                }
            ]
            self.assertEqual(self._warn_msgs(self._mk(path, specs)), [])
        finally:
            os.unlink(path)

    def test_control_ignores_fallback_to_source(self):
        # control fences ALWAYS flag when missing — a cross-node consumer depends on
        # them and cannot fall back to raw text; fallback_to_source is forced off.
        specs = [
            {
                "label": "winner_pick",
                "source": "response",
                "kind": "control",
                "fallback_to_source": True,
            }
        ]
        msgs = self._warn_msgs(self._mk(None, specs), {"raw_output": "no fence here"})
        self.assertTrue(msgs and "winner_pick" in msgs[0] and "control" in msgs[0])


class TestBlockPersistence(unittest.TestCase):
    """``persist_to``: the registry EMITS the extracted block as a structured
    sidecar in the node's own ``outputs/``. Still never gates."""

    FENCE = 'prose\n\n```json decomposed_subtasks\n{"subtasks": [{"id": 1}], "gaps": "none"}\n```\n'

    def _mk_ws(self, specs):
        """A stub leaf with a real workspace, registry, and a response channel."""
        import tempfile

        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        root = tempfile.mkdtemp()
        ws = InferencerWorkspace(root=root)
        ws.ensure_dirs()
        inf = _StubInferencer()
        inf._workspace = ws
        inf.expected_extraction = specs
        return inf, ws

    _SPEC = [
        {
            "label": "decomposed_subtasks",
            "source": "response",
            "kind": "content",
            "persist_to": "decomposed_subtasks.json",
        }
    ]

    def test_persists_extracted_block(self):
        import json

        inf, ws = self._mk_ws(self._SPEC)
        inf._run_expected_extraction({"raw_output": self.FENCE})
        target = ws.output_path("decomposed_subtasks.json")
        self.assertTrue(os.path.isfile(target), "block should be persisted")
        self.assertEqual(
            json.load(open(target, encoding="utf-8")),
            {"subtasks": [{"id": 1}], "gaps": "none"},
        )

    def test_no_fence_writes_nothing(self):
        inf, ws = self._mk_ws(self._SPEC)
        inf._run_expected_extraction({"raw_output": "no fence here"})
        self.assertFalse(os.path.isfile(ws.output_path("decomposed_subtasks.json")))

    def test_without_persist_to_is_noop(self):
        # Byte-identical to pre-persist behavior: validate only, emit nothing.
        specs = [
            {"label": "decomposed_subtasks", "source": "response", "kind": "content"}
        ]
        inf, ws = self._mk_ws(specs)
        inf._run_expected_extraction({"raw_output": self.FENCE})
        self.assertEqual(os.listdir(ws.outputs_dir), [])

    def test_unsafe_persist_to_is_rejected(self):
        # Confined to the node's own outputs/: traversal and absolute paths refused.
        for bad in ("../escape.json", "/tmp/escape.json", "a/../../escape.json"):
            inf, ws = self._mk_ws(
                [
                    {
                        "label": "decomposed_subtasks",
                        "source": "response",
                        "kind": "content",
                        "persist_to": bad,
                    }
                ]
            )
            inf._run_expected_extraction({"raw_output": self.FENCE})
            self.assertEqual(
                os.listdir(ws.outputs_dir), [], f"{bad!r} must not be written"
            )
            self.assertFalse(os.path.exists("/tmp/escape.json"))

    def test_persist_failure_never_gates(self):
        # A write error is logged, not raised — persistence is emission, not a gate.
        inf, _ws = self._mk_ws(self._SPEC)
        with patch(
            "agent_foundation.common.inferencers.inferencer_base.open",
            side_effect=OSError("disk full"),
            create=True,
        ):
            inf._run_expected_extraction({"raw_output": self.FENCE})  # must not raise


class TestBlockAccessors(unittest.TestCase):
    """Stateless block accessors — text in, object out, nothing cached."""

    FENCE_A = '```json blk\n{"v": "A"}\n```'
    FENCE_B = '```json blk\n{"v": "B"}\n```'

    def test_block_text_dict_and_missing(self):
        inf = _StubInferencer()
        self.assertIn('"v": "A"', inf.block_text(self.FENCE_A, "blk"))
        self.assertEqual(inf.block_dict(self.FENCE_A, "blk"), {"v": "A"})
        self.assertIsNone(inf.block_dict("no fence", "blk"))
        self.assertIsNone(inf.block_text(None, "blk"))

    def test_block_parsed_without_parser_returns_dict(self):
        inf = _StubInferencer()
        self.assertEqual(inf.block_parsed(self.FENCE_A, "blk"), {"v": "A"})

    def test_block_parsed_uses_registered_parser(self):
        @attrs
        class _WithParser(_StubInferencer):
            BLOCK_PARSERS = {"blk": "_take_v"}

            def _take_v(self, data):
                return data["v"]

        self.assertEqual(_WithParser().block_parsed(self.FENCE_A, "blk"), "A")

    def test_accessors_are_stateless(self):
        # The guard against the shared/re-roled instance hazard: nothing carries over
        # between calls, so a reused leaf can never serve another node's block.
        inf = _StubInferencer()
        self.assertEqual(inf.block_dict(self.FENCE_A, "blk"), {"v": "A"})
        self.assertEqual(inf.block_dict(self.FENCE_B, "blk"), {"v": "B"})
        self.assertIsNone(inf.block_dict("nothing", "blk"))
        self.assertFalse(
            [a for a in vars(inf) if "block" in a.lower()],
            "block accessors must not cache state on the instance",
        )


class TestPromoteExhaustedUpdate(unittest.TestCase):
    """A terminal UPDATE publishes; a terminal RETRY still fails.

    The judge contract defines UPDATE as "real, on-topic work is present but it
    is incomplete … We preserve it", and PASS as content that "need not be
    complete, deep, perfectly formatted" — so the two differ in degree, not in
    kind. Before this, exhausting the retry budget on UPDATE raised and
    ``_finalize_output`` never ran, making substantive work indistinguishable
    from producing nothing. RETRY means "fundamentally unusable", so it is the
    one verdict that must still be discarded.
    """

    # Bodies must differ (identical consecutive outputs trip the
    # ``guardrail_empty_fail_fast_n`` hopeless-loop guard) but share a prefix, so
    # the assertion holds whatever the recovery chain does to the attempt count.
    PREFIX = "substantive but incomplete draft"

    def _drafts(self):
        return [f"{self.PREFIX} #{i}" for i in range(10)]

    def test_terminal_update_is_published(self):
        judge = _StubJudge(verdict="UPDATE: expand the dataset tracing section")
        inf = _StubInferencer(output_guardrail_inferencer=judge, max_retry=2)
        inf.set_responses(self._drafts())

        result = inf.infer("input")

        self.assertTrue(
            str(result).startswith(self.PREFIX),
            f"the UPDATE body should be published, not discarded; got {result!r}",
        )
        self.assertGreater(inf._call_count, 1, "retries should still be spent first")

    def test_terminal_update_is_published_async(self):
        judge = _StubJudge(verdict="UPDATE: expand the dataset tracing section")
        inf = _StubInferencer(output_guardrail_inferencer=judge, max_retry=2)
        inf.set_responses(self._drafts())

        result = asyncio.run(inf.ainfer("input"))

        self.assertTrue(str(result).startswith(self.PREFIX), f"got {result!r}")

    def test_terminal_retry_still_raises(self):
        """RETRY = "narration-only, nothing to preserve" — must stay terminal."""
        judge = _StubJudge(verdict="RETRY: narration only, no actual work")
        inf = _StubInferencer(output_guardrail_inferencer=judge, max_retry=2)
        inf.set_responses(["narration", "more narration"])

        with self.assertRaises(OutputValidationExhaustedError):
            inf.infer("input")

    def test_opt_out_restores_old_behaviour(self):
        judge = _StubJudge(verdict="UPDATE: incomplete")
        inf = _StubInferencer(
            output_guardrail_inferencer=judge,
            max_retry=2,
            promote_exhausted_update=False,
        )
        inf.set_responses(["draft", "draft two"])

        with self.assertRaises(OutputValidationExhaustedError):
            inf.infer("input")

    def test_update_then_pass_is_unaffected(self):
        """The normal converge-to-PASS path must not change."""
        calls = [0]

        @attrs
        class _FlipJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                calls[0] += 1
                return "UPDATE: expand it" if calls[0] == 1 else "PASS"

        inf = _StubInferencer(output_guardrail_inferencer=_FlipJudge(), max_retry=3)
        inf.set_responses(["partial", "complete"])

        self.assertEqual(str(inf.infer("input")), "complete")

    def test_degraded_output_is_logged(self):
        """Promotion must never be silent."""
        judge = _StubJudge(verdict="UPDATE: trace the remaining datasets")
        inf = _StubInferencer(output_guardrail_inferencer=judge, max_retry=2)
        inf.set_responses(["draft", "draft two"])

        with patch.object(inf, "log_warning") as warned:
            inf.infer("input")

        events = [
            c.args[0]
            for c in warned.call_args_list
            if isinstance(c.args[0], dict)
            and c.args[0].get("event") == "DEGRADED_OUTPUT"
        ]
        self.assertEqual(len(events), 1, "expected exactly one DEGRADED_OUTPUT record")
        self.assertIn("trace the remaining datasets", str(events[0]["outstanding"]))


if __name__ == "__main__":
    unittest.main()
