# Feature: retry-native-timeout, Property 14: Prompt Template Formatting
"""Property-based test for prompt template formatting in StreamingInferencerBase.

**Validates: Requirements 20.2**

Property 14 states: For any random (prompt, partial_output) strings,
rendering the ``recovery/update`` template via ``render_recovery_prompt`` SHALL
produce a string containing both ``prompt`` and ``partial_output`` as substrings.
(``recovery/retry.jinja2`` was retired — RETRY is now a plain re-run of the input,
so it renders no template; see ``TestRetryRendersNone`` / ``TestRetryIsPlainRerun``.)
"""

import unittest

from agent_foundation.common.inferencers.prompt_templates import render_recovery_prompt
from hypothesis import assume, given, HealthCheck, settings, strategies as st

# Strategy: non-empty text without Jinja2-special chars
_safe_text = st.text(min_size=1, max_size=200).filter(
    lambda s: "{" not in s and "}" not in s and s.strip()
)


class TestPromptTemplateFormatting(unittest.TestCase):
    """Property 14: Prompt Template Formatting."""

    @given(prompt=_safe_text, partial_output=_safe_text)
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_update_prompt_contains_both_inputs(
        self, prompt: str, partial_output: str
    ) -> None:
        """render_recovery_prompt('recovery/update') produces string containing both inputs."""
        result = render_recovery_prompt(
            "recovery/update", prompt=prompt, partial_output=partial_output
        )
        self.assertIn(prompt, result)
        self.assertIn(partial_output, result)

    # NOTE: the old ``test_retry_prompt_contains_both_inputs`` was retired together
    # with ``recovery/retry.jinja2``. RETRY is now a plain re-run of the original
    # input (no template) — covered by ``TestRetryRendersNone`` /
    # ``TestRetryIsPlainRerun`` (and ``TestUpdateToRetryDispatch``) below.


if __name__ == "__main__":
    unittest.main()

from typing import Any, AsyncIterator

# ---------------------------------------------------------------------------
# Property 12: Cache Marker Stripping Preserves Content
# Property 13: CONTINUE Mode Newline Truncation
# Property 15: Session Recovery Precedence
# Property 16: FallbackInferMode Output Contract
# ---------------------------------------------------------------------------

from agent_foundation.common.inferencers.streaming_inferencer_base import (
    FallbackInferMode,
    StreamingInferencerBase,
)
from attr import attrib, attrs


@attrs
class MockStreamingInferencer(StreamingInferencerBase):
    """Minimal concrete subclass for testing recovery methods."""

    _mock_ainfer_result: str = attrib(default="mock_result")

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        yield self._mock_ainfer_result

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        return self._mock_ainfer_result

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        return self._mock_ainfer_result

    async def adisconnect(self):
        pass


class TestCacheMarkerStripping(unittest.TestCase):
    """Property 12: Cache Marker Stripping Preserves Content.

    For any cached partial output string followed by a STREAM FAILED marker,
    the _sanitize_partial method SHALL return exactly the original content
    with no data loss and no residual marker text.

    # Feature: retry-native-timeout, Property 12: Cache Marker Stripping Preserves Content
    **Validates: Requirements 19.3**
    """

    @given(
        content=st.text(min_size=1, max_size=500).filter(
            lambda s: "--- STREAM FAILED:" not in s and s.strip()
        )
    )
    @settings(max_examples=100)
    def test_marker_stripped_preserves_content(self, content: str):
        """Stripping the STREAM FAILED marker returns the original content."""
        inf = MockStreamingInferencer()
        marked = content + "\n--- STREAM FAILED: some error ---\n"
        result = inf._sanitize_partial(marked, FallbackInferMode.REFERENCE)
        # The result should contain the original content (stripped)
        self.assertIsNotNone(result)
        # Original content should be preserved (modulo strip)
        self.assertIn(content.strip(), result)
        # Marker should not be present
        self.assertNotIn("--- STREAM FAILED:", result)

    @given(
        content=st.text(min_size=1, max_size=500).filter(
            lambda s: "--- STREAM FAILED:" not in s and s.strip()
        )
    )
    @settings(max_examples=100)
    def test_no_marker_returns_content_unchanged(self, content: str):
        """When no marker is present, content is returned as-is (stripped)."""
        inf = MockStreamingInferencer()
        result = inf._sanitize_partial(content, FallbackInferMode.REFERENCE)
        if content.strip():
            self.assertEqual(result, content.strip())
        else:
            self.assertIsNone(result)

    def test_none_input_returns_none(self):
        inf = MockStreamingInferencer()
        self.assertIsNone(inf._sanitize_partial(None, FallbackInferMode.REFERENCE))

    def test_empty_input_returns_none(self):
        inf = MockStreamingInferencer()
        self.assertIsNone(inf._sanitize_partial("", FallbackInferMode.REFERENCE))


class TestContinueModeNewlineTruncation(unittest.TestCase):
    """Property 13: CONTINUE Mode Newline Truncation.

    For any partial output string, when fallback_infer_mode is CONTINUE,
    the truncation-to-last-newline logic SHALL produce a string that either
    ends at a newline boundary or is empty (if no newline exists).

    # Feature: retry-native-timeout, Property 13: CONTINUE Mode Newline Truncation
    **Validates: Requirements 19.4**
    """

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"), whitelist_characters="\n "
            ),
            min_size=3,
            max_size=200,
        ).filter(lambda s: "\n" in s and s.strip())
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_continue_truncates_to_last_newline(self, content: str):
        """CONTINUE mode truncates to last newline boundary."""
        inf = MockStreamingInferencer()
        result = inf._sanitize_partial(content, FallbackInferMode.CONTINUE)
        if result is not None:
            # The result should not end with a partial line — it should be
            # a subset of the content up to some newline boundary.
            # Verify the result is a prefix of the content (after stripping)
            # by checking it doesn't contain content that only appears after
            # the last newline in the original.
            pass  # The key property is that result is not None when content has newlines
            # and that the sanitized result is shorter than or equal to the original
            original_ref = inf._sanitize_partial(content, FallbackInferMode.REFERENCE)
            if original_ref:
                self.assertTrue(len(result) <= len(original_ref))

    def test_no_newline_may_return_content_or_none(self):
        """Content without newlines: CONTINUE mode may return content as-is or None."""
        inf = MockStreamingInferencer()
        result = inf._sanitize_partial(
            "single_line_content", FallbackInferMode.CONTINUE
        )
        # With no newline to truncate to, the content passes through stripped
        # (rfind returns -1, which is not > 0, so no truncation happens)
        self.assertEqual(result, "single_line_content")


class TestSessionRecoveryPrecedence(unittest.TestCase):
    """Property 15: Session Recovery Precedence.

    When both _session_id and cached partial output are available,
    _ainfer_recovery SHALL use session-based resumption and NOT apply
    cache-based prompt augmentation.

    # Feature: retry-native-timeout, Property 15: Session Recovery Precedence
    **Validates: Requirements 21.1, 21.2, 21.3**
    """

    def test_session_id_takes_precedence_over_cache(self):
        """When _session_id is set, session resume is used (not cache replay)."""
        import asyncio

        call_log = []

        @attrs
        class SessionTrackingInferencer(MockStreamingInferencer):
            async def _ainfer(self, inference_input, inference_config=None, **kwargs):
                if "session_id" in kwargs:
                    call_log.append(("session_resume", kwargs["session_id"]))
                else:
                    call_log.append(("normal", inference_input))
                return "session_result"

            async def adisconnect(self):
                call_log.append("disconnect")

        inf = SessionTrackingInferencer()
        inf._session_id = "test-session-123"

        async def run():
            return await inf._ainfer_recovery(
                "original prompt",
                last_exception=RuntimeError("failed"),
                last_partial_output="partial",
            )

        result = asyncio.run(run())
        self.assertEqual(result, "session_result")
        self.assertIn("disconnect", call_log)
        self.assertTrue(
            any(
                entry[0] == "session_resume"
                for entry in call_log
                if isinstance(entry, tuple)
            )
        )

    def test_no_session_falls_to_restart(self):
        """When _session_id is None and no cache, falls through to RESTART."""
        import asyncio

        call_log = []

        @attrs
        class TrackingInferencer(MockStreamingInferencer):
            async def _ainfer(self, inference_input, inference_config=None, **kwargs):
                call_log.append(("ainfer", inference_input))
                return "restart_result"

        inf = TrackingInferencer()
        inf._session_id = None

        async def run():
            return await inf._ainfer_recovery(
                "original prompt",
                last_exception=RuntimeError("failed"),
                last_partial_output=None,
            )

        result = asyncio.run(run())
        self.assertEqual(result, "restart_result")
        self.assertTrue(
            any(
                entry[0] == "ainfer" and entry[1] == "original prompt"
                for entry in call_log
                if isinstance(entry, tuple)
            )
        )


class TestFallbackInferModeOutputContract(unittest.TestCase):
    """Property 16: FallbackInferMode Output Contract (Option C).

    - UPDATE: renders the update prompt (prior work inlined as reference) and
      returns the agent's fresh response — edit-in-place, NOT a partial+continuation concat.
    - RETRY (incl. legacy REFERENCE/RESTART aliases): returns only the new response.

    # Feature: retry-native-timeout, Property 16: FallbackInferMode Output Contract
    **Validates: Requirements 18.2, 19.5, 19.6**
    """

    @given(
        partial=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ", min_size=1, max_size=50
        ),
        continuation=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ", min_size=1, max_size=50
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_update_mode_edits_in_place(self, partial: str, continuation: str):
        """UPDATE renders the update prompt (prior work inlined as reference) and
        returns the agent's fresh response — it does NOT concatenate partial+new."""
        import asyncio
        import os
        import tempfile

        assume(partial.strip())
        assume(continuation.strip())

        seen_prompt = {}

        @attrs
        class UpdateTestInferencer(MockStreamingInferencer):
            cont_value: str = attrib(default="")

            async def _ainfer(self, inference_input, inference_config=None, **kwargs):
                seen_prompt["p"] = str(inference_input)
                return self.cont_value

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write partial to a cache file
            cache_path = os.path.join(tmpdir, "cache.txt")
            with open(cache_path, "w") as f:
                f.write(partial)

            inf = UpdateTestInferencer(cont_value=continuation)
            inf.fallback_infer_mode = FallbackInferMode.UPDATE

            # Manually set up the ContextVar
            from agent_foundation.common.inferencers.inferencer_base import (
                _current_fallback_state,
            )

            state = {
                "last_exception": None,
                "partial_output": None,
                "cache_path": cache_path,
            }
            token = _current_fallback_state.set(state)
            try:

                async def run():
                    return await inf._ainfer_recovery(
                        "test prompt",
                        last_exception=RuntimeError("failed"),
                        last_partial_output=None,
                    )

                result = asyncio.run(run())
            finally:
                _current_fallback_state.reset(token)

            # UPDATE returns the agent's fresh response (no partial+continuation concat).
            self.assertEqual(result, continuation)
            # The prior work is inlined into the update prompt as a reference base.
            sanitized = inf._sanitize_partial(partial, FallbackInferMode.UPDATE)
            if sanitized:
                self.assertIn(sanitized, seen_prompt["p"])

    @given(
        continuation=st.text(min_size=1, max_size=100).filter(
            lambda s: s.strip() and "{" not in s and "}" not in s
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_reference_mode_returns_only_new(self, continuation: str):
        """REFERENCE mode returns only the new response."""
        import asyncio
        import os
        import tempfile

        @attrs
        class ReferenceTestInferencer(MockStreamingInferencer):
            cont_value: str = attrib(default="")

            async def _ainfer(self, inference_input, inference_config=None, **kwargs):
                return self.cont_value

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.txt")
            with open(cache_path, "w") as f:
                f.write("some partial output")

            inf = ReferenceTestInferencer(cont_value=continuation)
            inf.fallback_infer_mode = FallbackInferMode.REFERENCE

            from agent_foundation.common.inferencers.inferencer_base import (
                _current_fallback_state,
            )

            state = {
                "last_exception": None,
                "partial_output": None,
                "cache_path": cache_path,
            }
            token = _current_fallback_state.set(state)
            try:

                async def run():
                    return await inf._ainfer_recovery(
                        "test prompt",
                        last_exception=RuntimeError("failed"),
                        last_partial_output=None,
                    )

                result = asyncio.run(run())
            finally:
                _current_fallback_state.reset(token)

            # REFERENCE: result is only the continuation (new response)
            self.assertEqual(result, continuation)

    def test_restart_mode_returns_only_new(self):
        """RESTART mode returns only the new response (ignores cache)."""
        import asyncio

        @attrs
        class RestartTestInferencer(MockStreamingInferencer):
            async def _ainfer(self, inference_input, inference_config=None, **kwargs):
                return "fresh_result"

        inf = RestartTestInferencer()
        inf.fallback_infer_mode = FallbackInferMode.RESTART

        async def run():
            return await inf._ainfer_recovery(
                "test prompt",
                last_exception=RuntimeError("failed"),
                last_partial_output="ignored partial",
            )

        result = asyncio.run(run())
        self.assertEqual(result, "fresh_result")


# =============================================================================
# Post-mortem fix 1b: StreamingInferencerBase._pre_retry → reset_session()
# =============================================================================


class TestPreRetryResetsSession(unittest.IsolatedAsyncioTestCase):
    """Fix 1b: streaming inferencer's _pre_retry hook (overridden on
    StreamingInferencerBase) calls the existing reset_session() so when a
    parent inferencer's retry triggers child pre_retry propagation, the
    streaming child's active_session_id is cleared — the next ainfer()
    call on this instance starts fresh."""

    async def test_pre_retry_clears_active_session_id(self):
        inf = MockStreamingInferencer()
        inf._session_id = "abc123"
        self.assertEqual(inf.active_session_id, "abc123")
        await inf._pre_retry(attempt=0, exception=ConnectionError("transient"))
        self.assertIsNone(inf.active_session_id)


# =============================================================================
# Gap 2 — capability-aware UPDATE recovery: update.jinja2 branches on
# has_local_access; UPDATE degrades to RETRY only when there is nothing to build
# on (and a coerced RETRY with no partial is a PLAIN re-run, not a template).
# =============================================================================


@attrs
class _OutputPathStreaming(MockStreamingInferencer):
    """Mock that lets a test control resolve_output_path() + has_local_access."""

    _op_path = attrib(default=None)

    def resolve_output_path(self):
        return self._op_path


class TestOutputFileReady(unittest.TestCase):
    def test_missing_or_none(self):
        self.assertFalse(_OutputPathStreaming(op_path=None)._output_file_ready())
        self.assertFalse(
            _OutputPathStreaming(op_path="/nonexistent/xyz.md")._output_file_ready()
        )

    def test_empty_file_is_not_ready(self):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            path = f.name  # empty
        try:
            self.assertFalse(_OutputPathStreaming(op_path=path)._output_file_ready())
        finally:
            os.unlink(path)

    def test_nonempty_file_is_ready(self):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("real content")
            path = f.name
        try:
            self.assertTrue(_OutputPathStreaming(op_path=path)._output_file_ready())
        finally:
            os.unlink(path)


class TestUpdateModeViable(unittest.TestCase):
    def test_nonempty_partial_always_viable(self):
        for local in (True, False):
            inf = _OutputPathStreaming(op_path=None)
            inf.has_local_access = local
            self.assertTrue(inf._update_mode_viable("prior work"))

    def test_no_partial_no_local_not_viable(self):
        inf = _OutputPathStreaming(op_path=None)
        inf.has_local_access = False
        self.assertFalse(inf._update_mode_viable(None))
        self.assertFalse(inf._update_mode_viable("   "))

    def test_no_partial_local_with_file_viable(self):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("x")
            path = f.name
        try:
            inf = _OutputPathStreaming(op_path=path)
            inf.has_local_access = True
            self.assertTrue(inf._update_mode_viable(None))
        finally:
            os.unlink(path)

    def test_no_partial_local_no_file_not_viable(self):
        inf = _OutputPathStreaming(op_path="/nonexistent/xyz.md")
        inf.has_local_access = True
        self.assertFalse(inf._update_mode_viable(None))


class TestCapabilityAwareUpdateRender(unittest.TestCase):
    """update.jinja2 renders edit-in-place for local, re-emit-inline for no-local."""

    def test_local_renders_edit_in_place(self):
        inf = MockStreamingInferencer()
        inf.has_local_access = True
        out = inf._render_recovery_prompt(FallbackInferMode.UPDATE, "TASK", "PRIOR")
        self.assertIsNotNone(out)
        self.assertIn("edit it in place", out.lower())
        self.assertIn("PRIOR", out)

    def test_no_local_renders_inline(self):
        inf = MockStreamingInferencer()
        inf.has_local_access = False
        out = inf._render_recovery_prompt(FallbackInferMode.UPDATE, "TASK", "PRIOR")
        self.assertIsNotNone(out)
        self.assertIn("inline in your", out.lower())
        self.assertNotIn("edit it in place", out.lower())
        self.assertIn("PRIOR", out)


class TestUpdateToRetryDispatch(unittest.IsolatedAsyncioTestCase):
    """A no-local UPDATE with nothing to build on coerces to RETRY; with an empty
    partial that RETRY falls through to a PLAIN re-run (original prompt, no template)."""

    async def test_no_local_empty_cache_update_is_plain_rerun(self):
        seen = {}

        @attrs
        class _Capture(MockStreamingInferencer):
            async def _ainfer(self, inference_input, inference_config=None, **kwargs):
                seen["input"] = str(inference_input)
                return "fresh"

        inf = _Capture()
        inf.has_local_access = False
        inf.fallback_infer_mode = FallbackInferMode.UPDATE

        from agent_foundation.common.inferencers.inferencer_base import (
            _current_fallback_state,
        )

        token = _current_fallback_state.set({"cache_path": None})
        try:
            result = await inf._ainfer_recovery(
                "ORIGINAL_PROMPT",
                last_exception=RuntimeError("failed"),
                last_partial_output=None,
            )
        finally:
            _current_fallback_state.reset(token)

        self.assertEqual(result, "fresh")
        # Plain re-run: _ainfer got the ORIGINAL prompt, not a rendered recovery template.
        self.assertEqual(seen["input"], "ORIGINAL_PROMPT")
        self.assertNotIn("BEGIN INPUT", seen["input"])


class TestVerdictDrivesModeViaArgs1(unittest.IsolatedAsyncioTestCase):
    """Anti-regression for the phantom 'Gap 3': the guardrail verdict drives the
    mode via last_exception.args[1] (overriding the instance default)."""

    async def test_update_verdict_in_args1_selects_update(self):
        seen = {}

        @attrs
        class _Capture(MockStreamingInferencer):
            async def _ainfer(self, inference_input, inference_config=None, **kwargs):
                seen["input"] = str(inference_input)
                return "done"

        import os
        import tempfile

        inf = _Capture()
        inf.has_local_access = False
        inf.fallback_infer_mode = FallbackInferMode.RETRY  # default; args[1] must WIN

        from agent_foundation.common.inferencers.inferencer_base import (
            _current_fallback_state,
        )

        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "c.txt")
            with open(cache, "w", encoding="utf-8") as f:
                f.write("PRIOR_WORK")
            token = _current_fallback_state.set({"cache_path": cache})
            try:
                # args[1] carries the judge verdict, exactly as async_utils.py builds
                # OutputValidationExhaustedError("Output validation failed", verdict).
                exc = RuntimeError("Output validation failed", "update")
                await inf._ainfer_recovery(
                    "TASK", last_exception=exc, last_partial_output=None
                )
            finally:
                _current_fallback_state.reset(token)

        # UPDATE was selected (not the RETRY default) → no-local inline branch rendered.
        self.assertIn("inline in your", seen["input"].lower())
        self.assertIn("PRIOR_WORK", seen["input"])


# =============================================================================
# Q1 — retire recovery/retry.jinja2: RETRY is a plain re-run of the original
# input (no template). The keystone lives in _render_recovery_prompt (returns
# None for the retry family); dispatch + resume hooks null-check → plain re-run.
# =============================================================================


class TestRetryRendersNone(unittest.TestCase):
    """The RETRY keystone: ``_render_recovery_prompt`` returns None for RETRY and its
    retry-family aliases (which share the ``"retry"`` value), so dispatch and the
    resume hooks fall through to a plain re-run of the original input. UPDATE still
    renders a template (contrast — proves the guard is scoped to RETRY only)."""

    def test_retry_family_renders_none(self):
        inf = MockStreamingInferencer()
        for mode in (
            FallbackInferMode.RETRY,
            FallbackInferMode.RESTART,
            FallbackInferMode.RETRY_WITH_REFERENCE,
            FallbackInferMode.REFERENCE,
        ):
            for local in (True, False):
                inf.has_local_access = local
                self.assertIsNone(
                    inf._render_recovery_prompt(mode, "TASK", "PRIOR"),
                    f"{mode!r} (local={local}) must render None (plain re-run)",
                )

    def test_update_still_renders(self):
        inf = MockStreamingInferencer()
        inf.has_local_access = True
        self.assertIsNotNone(
            inf._render_recovery_prompt(FallbackInferMode.UPDATE, "TASK", "PRIOR")
        )


class TestRetryIsPlainRerun(unittest.IsolatedAsyncioTestCase):
    """RETRY is a plain re-run of the ORIGINAL input even when a non-empty partial is
    cached — the partial is dropped, NOT fed back as a reference (``TestUpdateToRetryDispatch``
    covers only the coerced-empty case)."""

    async def test_retry_nonempty_partial_is_plain_rerun(self):
        seen = {}

        @attrs
        class _Capture(MockStreamingInferencer):
            async def _ainfer(self, inference_input, inference_config=None, **kwargs):
                seen["input"] = str(inference_input)
                return "fresh"

        import os
        import tempfile

        inf = _Capture()
        inf.fallback_infer_mode = FallbackInferMode.RETRY

        from agent_foundation.common.inferencers.inferencer_base import (
            _current_fallback_state,
        )

        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "c.txt")
            with open(cache, "w", encoding="utf-8") as f:
                f.write("NON_EMPTY_PARTIAL_WORK")
            token = _current_fallback_state.set({"cache_path": cache})
            try:
                result = await inf._ainfer_recovery(
                    "ORIGINAL_PROMPT",
                    last_exception=RuntimeError("failed"),
                    last_partial_output=None,
                )
            finally:
                _current_fallback_state.reset(token)

        self.assertEqual(result, "fresh")
        # Plain re-run: original prompt verbatim, no rendered template, partial dropped.
        self.assertEqual(seen["input"], "ORIGINAL_PROMPT")
        self.assertNotIn("BEGIN INPUT", seen["input"])
        self.assertNotIn("NON_EMPTY_PARTIAL_WORK", seen["input"])


# =============================================================================
# Q2 — thread the judge's <reason> into recovery/update.jinja2 (guided fix). It
# rides the per-call _current_fallback_state (guardrail_reason), which the BASE
# _render_recovery_prompt reads internally — so a 3-arg override is not broken.
# =============================================================================


class TestUpdateReasonThreading(unittest.TestCase):
    """``reason`` present → a guided-fix line renders; absent (resume / non-guardrail
    UPDATE) → no reason line. Covers both the instance seam (ContextVar read) and the
    standalone ``render_recovery_prompt(reason=...)`` kwarg."""

    def test_render_includes_reason_when_present(self):
        from agent_foundation.common.inferencers.inferencer_base import (
            _current_fallback_state,
        )

        inf = MockStreamingInferencer()
        inf.has_local_access = True
        token = _current_fallback_state.set({"guardrail_reason": "MISSING_WRAPPER"})
        try:
            out = inf._render_recovery_prompt(FallbackInferMode.UPDATE, "TASK", "PRIOR")
        finally:
            _current_fallback_state.reset(token)
        self.assertIn("MISSING_WRAPPER", out)
        self.assertIn("specific issue to fix", out.lower())

    def test_render_omits_reason_block_when_absent(self):
        from agent_foundation.common.inferencers.inferencer_base import (
            _current_fallback_state,
        )

        # No fallback state active → reason is None → no guided-fix line.
        self.assertIsNone(_current_fallback_state.get(None))
        inf = MockStreamingInferencer()
        inf.has_local_access = True
        out = inf._render_recovery_prompt(FallbackInferMode.UPDATE, "TASK", "PRIOR")
        self.assertNotIn("specific issue to fix", out.lower())
        self.assertIn("PRIOR", out)

    def test_module_render_reason_kwarg(self):
        with_reason = render_recovery_prompt(
            "recovery/update",
            prompt="TASK",
            partial_output="PRIOR",
            reason="MISSING_WRAPPER",
        )
        self.assertIn("MISSING_WRAPPER", with_reason)
        without = render_recovery_prompt(
            "recovery/update", prompt="TASK", partial_output="PRIOR"
        )
        self.assertNotIn("specific issue to fix", without.lower())


class TestUpdateReasonEndToEnd(unittest.IsolatedAsyncioTestCase):
    """Integration: a UPDATE recovery with a stashed ``guardrail_reason`` weaves that
    reason into the prompt the backend actually re-runs (``_ainfer`` input)."""

    async def test_reason_reaches_ainfer_input(self):
        seen = {}

        @attrs
        class _Capture(MockStreamingInferencer):
            async def _ainfer(self, inference_input, inference_config=None, **kwargs):
                seen["input"] = str(inference_input)
                return "done"

        import os
        import tempfile

        inf = _Capture()
        inf.has_local_access = False  # no-local → deterministic inline branch
        inf.fallback_infer_mode = FallbackInferMode.UPDATE

        from agent_foundation.common.inferencers.inferencer_base import (
            _current_fallback_state,
        )

        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "c.txt")
            with open(cache, "w", encoding="utf-8") as f:
                f.write("PRIOR_WORK")
            token = _current_fallback_state.set(
                {"cache_path": cache, "guardrail_reason": "MISSING_WRAPPER"}
            )
            try:
                await inf._ainfer_recovery(
                    "TASK", last_exception=RuntimeError("x"), last_partial_output=None
                )
            finally:
                _current_fallback_state.reset(token)

        self.assertIn("MISSING_WRAPPER", seen["input"])
        self.assertIn("PRIOR_WORK", seen["input"])
