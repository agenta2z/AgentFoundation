"""Unit tests for the meta agent error hierarchy."""

import pytest

from agent_foundation.automation.meta_agent.errors import (
    GraphSynthesisError,
    InsufficientSuccessTracesError,
    MetaAgentError,
    PatternExtractionError,
    PipelineStageError,
    TraceAlignmentError,
    TraceCollectionError,
    TraceEvaluationError,
    TraceNormalizationError,
)


class TestMetaAgentError:
    def test_is_base_exception(self):
        err = MetaAgentError("something broke")
        assert isinstance(err, Exception)
        assert str(err) == "something broke"


class TestTraceCollectionError:
    def test_inherits_from_base(self):
        original = ValueError("bad value")
        err = TraceCollectionError(run_index=3, original_error=original)
        assert isinstance(err, MetaAgentError)

    def test_stores_attributes(self):
        original = RuntimeError("timeout")
        err = TraceCollectionError(run_index=5, original_error=original)
        assert err.run_index == 5
        assert err.original_error is original

    def test_message_format(self):
        original = RuntimeError("timeout")
        err = TraceCollectionError(run_index=2, original_error=original)
        assert "run 2" in str(err)
        assert "timeout" in str(err)


class TestTraceNormalizationError:
    def test_inherits_from_base(self):
        err = TraceNormalizationError("t-1", 4, "unknown type")
        assert isinstance(err, MetaAgentError)

    def test_stores_attributes(self):
        err = TraceNormalizationError("trace-abc", 7, "bad target")
        assert err.trace_id == "trace-abc"
        assert err.step_index == 7

    def test_message_format(self):
        err = TraceNormalizationError("t-1", 0, "missing field")
        msg = str(err)
        assert "t-1" in msg
        assert "step 0" in msg
        assert "missing field" in msg


class TestTraceAlignmentError:
    def test_inherits_from_base(self):
        err = TraceAlignmentError("too dissimilar")
        assert isinstance(err, MetaAgentError)

    def test_default_trace_ids(self):
        err = TraceAlignmentError("oops")
        assert err.trace_ids == []

    def test_stores_trace_ids(self):
        ids = ["t-1", "t-2"]
        err = TraceAlignmentError("diverged", trace_ids=ids)
        assert err.trace_ids == ids

    def test_message_format(self):
        err = TraceAlignmentError("score below threshold")
        assert "Alignment failed" in str(err)
        assert "score below threshold" in str(err)


class TestPatternExtractionError:
    def test_inherits_from_base(self):
        err = PatternExtractionError("loop detection failed")
        assert isinstance(err, MetaAgentError)

    def test_plain_message(self):
        err = PatternExtractionError("no patterns found")
        assert str(err) == "no patterns found"


class TestGraphSynthesisError:
    def test_inherits_from_base(self):
        err = GraphSynthesisError("cannot build graph")
        assert isinstance(err, MetaAgentError)

    def test_default_pattern_type(self):
        err = GraphSynthesisError("fail")
        assert err.pattern_type is None

    def test_stores_pattern_type(self):
        err = GraphSynthesisError("bad loop", pattern_type="loop")
        assert err.pattern_type == "loop"

    def test_message_format(self):
        err = GraphSynthesisError("invalid branch", pattern_type="branch")
        assert "Synthesis failed" in str(err)
        assert "invalid branch" in str(err)


class TestPipelineStageError:
    def test_inherits_from_base(self):
        original = RuntimeError("boom")
        err = PipelineStageError(stage="alignment", original_error=original)
        assert isinstance(err, MetaAgentError)

    def test_stores_attributes(self):
        original = ValueError("bad config")
        err = PipelineStageError(stage="normalization", original_error=original)
        assert err.stage == "normalization"
        assert err.original_error is original

    def test_message_format(self):
        original = TypeError("wrong type")
        err = PipelineStageError(stage="synthesis", original_error=original)
        msg = str(err)
        assert "synthesis" in msg
        assert "wrong type" in msg


class TestTraceEvaluationError:
    def test_inherits_from_base(self):
        err = TraceEvaluationError("t-1", "low quality")
        assert isinstance(err, MetaAgentError)

    def test_stores_attributes(self):
        err = TraceEvaluationError("trace-abc", "inferencer failed")
        assert err.trace_id == "trace-abc"

    def test_message_format(self):
        err = TraceEvaluationError("t-42", "score too low")
        msg = str(err)
        assert "t-42" in msg
        assert "score too low" in msg
        assert "Trace evaluation failed" in msg


class TestInsufficientSuccessTracesError:
    def test_inherits_from_base(self):
        err = InsufficientSuccessTracesError(required=3, actual=1, total=5)
        assert isinstance(err, MetaAgentError)

    def test_stores_attributes(self):
        err = InsufficientSuccessTracesError(required=5, actual=2, total=10)
        assert err.required == 5
        assert err.actual == 2
        assert err.total == 10

    def test_message_format(self):
        err = InsufficientSuccessTracesError(required=3, actual=1, total=5)
        msg = str(err)
        assert "1/5" in msg
        assert "3" in msg
        assert "Insufficient successful traces" in msg

    def test_zero_actual(self):
        err = InsufficientSuccessTracesError(required=2, actual=0, total=4)
        assert err.actual == 0
        assert "0/4" in str(err)


class TestCatchAllPattern:
    """Verify all errors can be caught with the base class."""

    def test_catch_all_subclasses(self):
        errors = [
            TraceCollectionError(0, RuntimeError("x")),
            TraceNormalizationError("t", 0, "x"),
            TraceAlignmentError("x"),
            PatternExtractionError("x"),
            TraceEvaluationError("t-1", "bad quality"),
            InsufficientSuccessTracesError(3, 1, 5),
            GraphSynthesisError("x"),
            PipelineStageError("s", RuntimeError("x")),
        ]
        for err in errors:
            with pytest.raises(MetaAgentError):
                raise err
