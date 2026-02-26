"""
Property-based test for synthetic data distinctness.

Feature: meta-agent-workflow, Property 3: Synthetic data distinctness

*For any* SyntheticDataProvider and count N >= 2, the generated N data sets
SHALL all be pairwise distinct (no two sets are equal).

**Validates: Requirements 1.3, 9.1**
"""

from typing import Any, Dict, List

from hypothesis import given, settings, strategies as st

from science_modeling_tools.automation.meta_agent.synthetic_data import (
    SyntheticDataProvider,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Count: N >= 2, capped to keep tests fast
count_st = st.integers(min_value=2, max_value=50)

# Supported type hints for schema fields
type_hint_st = st.sampled_from(["str", "int", "float", "bool"])

# Parameter schema: 1-5 fields with supported type hints
parameter_schema_st = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=("Ll",)),
        min_size=1,
        max_size=10,
    ),
    values=type_hint_st,
    min_size=1,
    max_size=5,
)


def _dict_key(d: Dict[str, Any]) -> str:
    """Create a hashable key from a dict for distinctness checking."""
    return str(sorted(d.items()))


# ---------------------------------------------------------------------------
# Property 3: Synthetic data distinctness
# ---------------------------------------------------------------------------


class TestSyntheticDataDistinctnessProperty:
    """
    Property 3: Synthetic data distinctness

    *For any* SyntheticDataProvider and count N >= 2, the generated N data
    sets SHALL all be pairwise distinct (no two sets are equal).

    **Validates: Requirements 1.3, 9.1**
    """

    @given(schema=parameter_schema_st, n=count_st)
    @settings(max_examples=100)
    def test_schema_based_generation_produces_distinct_data(
        self, schema: Dict[str, str], n: int
    ):
        """
        For any parameter schema and N >= 2, schema-based generation
        produces N pairwise-distinct data sets.
        """
        provider = SyntheticDataProvider(parameter_schema=schema)
        result = provider.generate(n)

        assert len(result) == n, (
            f"Expected {n} data sets, got {len(result)}"
        )

        keys = [_dict_key(item) for item in result]
        assert len(set(keys)) == n, (
            f"Expected {n} distinct data sets, got {len(set(keys))} unique "
            f"(schema={schema}, n={n})"
        )

    @given(n=count_st)
    @settings(max_examples=100)
    def test_custom_generator_returning_distinct_data(self, n: int):
        """
        For any N >= 2, a custom generator that returns distinct data
        passes through correctly.
        """
        def distinct_gen(count: int) -> List[Dict[str, Any]]:
            return [{"id": i, "value": f"item_{i}"} for i in range(count)]

        provider = SyntheticDataProvider(custom_generator=distinct_gen)
        result = provider.generate(n)

        assert len(result) == n
        keys = [_dict_key(item) for item in result]
        assert len(set(keys)) == n, (
            f"Custom generator should produce {n} distinct sets"
        )

    @given(schema=parameter_schema_st, n=count_st)
    @settings(max_examples=100)
    def test_pairwise_no_two_equal(
        self, schema: Dict[str, str], n: int
    ):
        """
        Explicit pairwise check: for any i != j, result[i] != result[j].
        """
        provider = SyntheticDataProvider(parameter_schema=schema)
        result = provider.generate(n)

        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                assert _dict_key(result[i]) != _dict_key(result[j]), (
                    f"Data sets at index {i} and {j} are equal: "
                    f"{result[i]} (schema={schema}, n={n})"
                )

    @given(n=count_st)
    @settings(max_examples=100)
    def test_bool_only_schema_still_distinct(self, n: int):
        """
        Even with a bool-only schema (only 2 natural distinct values),
        the provider guarantees N >= 2 distinct data sets via its
        fallback mechanism.
        """
        provider = SyntheticDataProvider(parameter_schema={"flag": "bool"})
        result = provider.generate(n)

        assert len(result) == n
        keys = [_dict_key(item) for item in result]
        assert len(set(keys)) == n, (
            f"Bool-only schema should still produce {n} distinct sets "
            f"via fallback mechanism"
        )
