"""
Property-based test for ValidationResults success rate calculation.

Feature: meta-agent-workflow, Property 21: Validation success rate calculation

**Validates: Requirements 8.4**
"""

from hypothesis import given, settings, strategies as st

from science_modeling_tools.automation.meta_agent.models import (
    ValidationResult,
    ValidationResults,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

validation_result_st = st.builds(
    ValidationResult,
    input_data=st.none(),
    passed=st.booleans(),
)

validation_results_list_st = st.lists(validation_result_st, min_size=0, max_size=50)


# ---------------------------------------------------------------------------
# Property 21: Validation success rate calculation
# ---------------------------------------------------------------------------


class TestValidationSuccessRateProperty:
    """
    Property 21: Validation success rate calculation

    *For any* list of ValidationResults, the success_rate SHALL equal the
    count of passed results divided by the total count. If the list is
    empty, success_rate SHALL be 0.0.

    **Validates: Requirements 8.4**
    """

    @given(results=validation_results_list_st)
    @settings(max_examples=200)
    def test_success_rate_equals_passed_over_total(self, results):
        """success_rate == count(passed) / total for non-empty lists."""
        vr = ValidationResults(results=results)

        if len(results) == 0:
            assert vr.success_rate == 0.0
        else:
            expected = sum(1 for r in results if r.passed) / len(results)
            assert vr.success_rate == expected

    @given(results=st.lists(validation_result_st, min_size=1, max_size=50))
    @settings(max_examples=200)
    def test_success_rate_bounded_zero_to_one(self, results):
        """success_rate is always in [0.0, 1.0]."""
        vr = ValidationResults(results=results)
        assert 0.0 <= vr.success_rate <= 1.0

    def test_empty_list_returns_zero(self):
        """Empty results list yields 0.0."""
        vr = ValidationResults(results=[])
        assert vr.success_rate == 0.0

    @given(n=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100)
    def test_all_passed_gives_rate_one(self, n):
        """When every result passed, success_rate == 1.0."""
        results = [ValidationResult(input_data=None, passed=True) for _ in range(n)]
        vr = ValidationResults(results=results)
        assert vr.success_rate == 1.0

    @given(n=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100)
    def test_none_passed_gives_rate_zero(self, n):
        """When no result passed, success_rate == 0.0."""
        results = [ValidationResult(input_data=None, passed=False) for _ in range(n)]
        vr = ValidationResults(results=results)
        assert vr.success_rate == 0.0
