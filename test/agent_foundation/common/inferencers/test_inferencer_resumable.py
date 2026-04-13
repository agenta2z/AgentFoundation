"""Tests for making all inferencers Resumable.

Covers:
- isinstance(inferencer, Resumable) for all inferencer types
- Default attrs (enable_result_save=False, resume_with_saved_results=False)
- _get_result_path behavior with/without output_path
- StreamingInferencerBase: cache_folder fallback, _find_latest_cache, _load_cached_or_resume
- Diamond inheritance for DualInferencer/PTI/BTA
- Constructor accepts Resumable kwargs
"""

import hashlib
import os
import time
import unittest

from typing import Any, AsyncIterator, Optional

from attr import attrib, attrs

from rich_python_utils.common_objects.workflow.common.resumable import Resumable


# ---------------------------------------------------------------------------
# Minimal mock inferencers for testing
# ---------------------------------------------------------------------------

@attrs
class MockPlainInferencer:
    """Simulates a plain InferencerBase for testing _get_result_path.

    We can't instantiate InferencerBase directly (abstract), so we import
    the real mock from the existing test fixtures.
    """
    pass


# Import real inferencer classes lazily to avoid heavy imports at module level
def _get_inferencer_base():
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase
    return InferencerBase


def _get_streaming_base():
    from agent_foundation.common.inferencers.streaming_inferencer_base import (
        StreamingInferencerBase,
    )
    return StreamingInferencerBase


def _make_mock_streaming(cache_folder=None, **kwargs):
    """Create a minimal concrete StreamingInferencerBase for testing."""
    from agent_foundation.common.inferencers.streaming_inferencer_base import (
        StreamingInferencerBase,
    )

    @attrs
    class _MockStreaming(StreamingInferencerBase):
        async def _ainfer_streaming(self, prompt: str, **kw) -> AsyncIterator[str]:
            yield "mock"

        def _infer(self, inference_input, inference_config=None, **kw):
            return "mock"

        async def _ainfer(self, inference_input, inference_config=None, **kw):
            return "mock"

        async def adisconnect(self):
            pass

    return _MockStreaming(cache_folder=cache_folder, **kwargs)


def _make_mock_plain(**kwargs):
    """Create a minimal concrete InferencerBase for testing."""
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase

    @attrs
    class _MockPlain(InferencerBase):
        def _infer(self, inference_input, inference_config=None, **kw):
            return "mock"

        async def _ainfer(self, inference_input, inference_config=None, **kw):
            return "mock"

    return _MockPlain(**kwargs)


# ---------------------------------------------------------------------------
# Tests: isinstance + defaults
# ---------------------------------------------------------------------------

class TestIsInstanceResumable(unittest.TestCase):

    def test_plain_inferencer_is_resumable(self):
        inf = _make_mock_plain()
        self.assertIsInstance(inf, Resumable)

    def test_streaming_inferencer_is_resumable(self):
        inf = _make_mock_streaming()
        self.assertIsInstance(inf, Resumable)

    def test_defaults_no_behavior_change(self):
        inf = _make_mock_plain()
        self.assertFalse(inf.enable_result_save)
        self.assertFalse(inf.resume_with_saved_results)

    def test_constructor_accepts_resumable_kwargs(self):
        inf = _make_mock_plain(enable_result_save=True, checkpoint_mode='jsonfy')
        self.assertTrue(inf.enable_result_save)
        self.assertEqual(inf.checkpoint_mode, 'jsonfy')


# ---------------------------------------------------------------------------
# Tests: _get_result_path
# ---------------------------------------------------------------------------

class TestGetResultPath(unittest.TestCase):

    def test_raises_without_output_path(self):
        inf = _make_mock_plain(output_path=None)
        with self.assertRaises(NotImplementedError):
            inf._get_result_path("step_0")

    def test_with_output_path(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _make_mock_plain(output_path=tmpdir)
            result = inf._get_result_path("step_0")
            self.assertEqual(result, os.path.join(tmpdir, "step_0.pkl"))

    def test_streaming_fallback_to_cache_folder(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _make_mock_streaming(cache_folder=tmpdir)
            result = inf._get_result_path("step_0")
            self.assertEqual(result, os.path.join(tmpdir, "step_0.pkl"))

    def test_streaming_prefers_output_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "output")
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(out_dir)
            os.makedirs(cache_dir)
            inf = _make_mock_streaming(output_path=out_dir, cache_folder=cache_dir)
            result = inf._get_result_path("step_0")
            self.assertEqual(result, os.path.join(out_dir, "step_0.pkl"))


# ---------------------------------------------------------------------------
# Tests: _find_latest_cache
# ---------------------------------------------------------------------------

class TestFindLatestCache(unittest.TestCase):

    def _write_cache(self, cache_folder, class_name, prompt, content, delay=0):
        """Write a mock cache file matching _open_cache_file's naming convention."""
        import uuid
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        session_dir = os.path.join(
            cache_folder, class_name, f"mock-id_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(session_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        path = os.path.join(session_dir, f"stream_{unique_id}_{prompt_hash}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        if delay:
            time.sleep(delay)
        return path

    def test_no_cache_returns_none(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _make_mock_streaming(cache_folder=tmpdir)
            self.assertIsNone(inf._find_latest_cache("test prompt"))

    def test_finds_matching_cache(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use the actual class name that _find_latest_cache will look for
            inf = _make_mock_streaming(cache_folder=tmpdir)
            class_name = inf.__class__.__name__
            path = self._write_cache(tmpdir, class_name, "test prompt", "partial content")
            result = inf._find_latest_cache("test prompt")
            self.assertEqual(result, path)

    def test_filters_by_prompt_hash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _make_mock_streaming(cache_folder=tmpdir)
            class_name = inf.__class__.__name__
            self._write_cache(tmpdir, class_name, "other prompt", "wrong content")
            result = inf._find_latest_cache("test prompt")
            self.assertIsNone(result)  # different prompt hash

    def test_picks_most_recent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _make_mock_streaming(cache_folder=tmpdir)
            class_name = inf.__class__.__name__
            self._write_cache(tmpdir, class_name, "test prompt", "old content")
            time.sleep(0.05)  # ensure different mtime
            newer_path = self._write_cache(tmpdir, class_name, "test prompt", "new content")
            result = inf._find_latest_cache("test prompt")
            self.assertEqual(result, newer_path)


# ---------------------------------------------------------------------------
# Tests: _load_cached_or_resume
# ---------------------------------------------------------------------------

class TestLoadCachedOrResume(unittest.TestCase):

    def test_disabled_by_default(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _make_mock_streaming(
                cache_folder=tmpdir,
                resume_with_saved_results=False,  # default
            )
            self.assertIsNone(inf._load_cached_or_resume("test prompt"))

    def test_no_cache_returns_none(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _make_mock_streaming(
                cache_folder=tmpdir,
                resume_with_saved_results=True,
            )
            self.assertIsNone(inf._load_cached_or_resume("test prompt"))

    def test_completed_cache_returns_completed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _make_mock_streaming(
                cache_folder=tmpdir,
                resume_with_saved_results=True,
            )
            class_name = inf.__class__.__name__
            content = "Hello world\n--- STREAM COMPLETED SUCCESSFULLY ---\n"
            self._write_cache(tmpdir, class_name, "test prompt", content)
            result = inf._load_cached_or_resume("test prompt")
            self.assertIsNotNone(result)
            status, text = result
            self.assertEqual(status, 'completed')
            self.assertEqual(text, "Hello world")

    def test_failed_cache_returns_partial(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _make_mock_streaming(
                cache_folder=tmpdir,
                resume_with_saved_results=True,
            )
            class_name = inf.__class__.__name__
            content = "Partial output\n--- STREAM FAILED: connection dropped ---\n"
            self._write_cache(tmpdir, class_name, "test prompt", content)
            result = inf._load_cached_or_resume("test prompt")
            self.assertIsNotNone(result)
            status, text = result
            self.assertEqual(status, 'partial')
            self.assertIn("Partial output", text)

    def test_no_marker_treated_as_partial(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _make_mock_streaming(
                cache_folder=tmpdir,
                resume_with_saved_results=True,
            )
            class_name = inf.__class__.__name__
            content = "Abruptly stopped content"
            self._write_cache(tmpdir, class_name, "test prompt", content)
            result = inf._load_cached_or_resume("test prompt")
            self.assertIsNotNone(result)
            status, text = result
            self.assertEqual(status, 'partial')
            self.assertEqual(text, "Abruptly stopped content")

    def _write_cache(self, cache_folder, class_name, prompt, content):
        import uuid
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        session_dir = os.path.join(
            cache_folder, class_name, f"mock-id_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(session_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        path = os.path.join(session_dir, f"stream_{unique_id}_{prompt_hash}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path


# ---------------------------------------------------------------------------
# Tests: _load_result contract preserved
# ---------------------------------------------------------------------------

class TestLoadResultUntouched(unittest.TestCase):

    def test_load_result_returns_raw(self):
        """_load_result must return raw result (not a tuple) for Workflow engine."""
        import tempfile
        import pickle
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _make_mock_streaming(cache_folder=tmpdir)
            # Manually save a pickle result
            result_path = os.path.join(tmpdir, "test.pkl")
            with open(result_path, "wb") as f:
                pickle.dump("raw_result_string", f)
            loaded = inf._load_result("test", result_path)
            self.assertEqual(loaded, "raw_result_string")
            self.assertNotIsInstance(loaded, tuple)


# ---------------------------------------------------------------------------
# Tests: Diamond inheritance
# ---------------------------------------------------------------------------

class TestDiamondInheritance(unittest.TestCase):

    def test_dual_inferencer(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
            DualInferencer,
        )
        self.assertTrue(issubclass(DualInferencer, Resumable))
        mro_names = [c.__name__ for c in DualInferencer.__mro__]
        self.assertEqual(mro_names.count('Resumable'), 1)
        # Construction should work
        di = DualInferencer()
        self.assertFalse(di.enable_result_save)

    def test_plan_then_implement(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
            PlanThenImplementInferencer,
        )
        self.assertTrue(issubclass(PlanThenImplementInferencer, Resumable))
        mro_names = [c.__name__ for c in PlanThenImplementInferencer.__mro__]
        self.assertEqual(mro_names.count('Resumable'), 1)

    def test_breakdown_then_aggregate(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )
        self.assertTrue(issubclass(BreakdownThenAggregateInferencer, Resumable))
        mro_names = [c.__name__ for c in BreakdownThenAggregateInferencer.__mro__]
        self.assertEqual(mro_names.count('Resumable'), 1)


# ---------------------------------------------------------------------------
# Tests: worker_manages_resume detection
# ---------------------------------------------------------------------------

class TestWorkerManagesResumeDetection(unittest.TestCase):

    def test_streaming_inferencer_detected_as_resumable(self):
        """BTA's isinstance(worker, Resumable) check now returns True for streaming."""
        inf = _make_mock_streaming()
        self.assertIsInstance(inf, Resumable)
        # This is what BTA checks to decide worker_manages_resume
        self.assertTrue(isinstance(inf, Resumable))


if __name__ == "__main__":
    unittest.main()
