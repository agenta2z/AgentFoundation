"""Unit tests for SyntheticDataProvider.

Tests cover:
- Custom generator is used when provided
- Parameter schema generates correct types
- Generated data sets are pairwise distinct
- ValueError on invalid count
- ValueError when no schema or generator configured
- Unsupported type hints handled gracefully

Requirements: 9.1, 9.2, 9.3
"""

from typing import Any, Dict, List

import pytest

from agent_foundation.automation.meta_agent.synthetic_data import (
    SyntheticDataProvider,
)


# ---------------------------------------------------------------------------
# Custom generator (Req 9.3)
# ---------------------------------------------------------------------------

class TestCustomGenerator:
    def test_custom_generator_used_when_provided(self):
        def my_gen(count: int) -> List[Dict[str, Any]]:
            return [{"x": i} for i in range(count)]

        provider = SyntheticDataProvider(custom_generator=my_gen)
        result = provider.generate(3)
        assert result == [{"x": 0}, {"x": 1}, {"x": 2}]

    def test_custom_generator_overrides_schema(self):
        def my_gen(count: int) -> List[Dict[str, Any]]:
            return [{"custom": True}] * count

        provider = SyntheticDataProvider(
            parameter_schema={"name": "str"},
            custom_generator=my_gen,
        )
        result = provider.generate(2)
        assert all(item == {"custom": True} for item in result)

    def test_custom_generator_receives_count(self):
        received = []

        def my_gen(count: int) -> List[Dict[str, Any]]:
            received.append(count)
            return [{}] * count

        provider = SyntheticDataProvider(custom_generator=my_gen)
        provider.generate(7)
        assert received == [7]


# ---------------------------------------------------------------------------
# Schema-based generation (Req 9.1, 9.2)
# ---------------------------------------------------------------------------

class TestSchemaGeneration:
    def test_str_type_generates_strings(self):
        provider = SyntheticDataProvider(parameter_schema={"name": "str"})
        result = provider.generate(3)
        assert len(result) == 3
        for item in result:
            assert isinstance(item["name"], str)
            assert len(item["name"]) > 0

    def test_int_type_generates_integers(self):
        provider = SyntheticDataProvider(parameter_schema={"age": "int"})
        result = provider.generate(3)
        for item in result:
            assert isinstance(item["age"], int)

    def test_float_type_generates_floats(self):
        provider = SyntheticDataProvider(parameter_schema={"score": "float"})
        result = provider.generate(3)
        for item in result:
            assert isinstance(item["score"], float)

    def test_bool_type_generates_booleans(self):
        provider = SyntheticDataProvider(parameter_schema={"active": "bool"})
        result = provider.generate(5)
        for item in result:
            assert isinstance(item["active"], bool)

    def test_multiple_fields(self):
        schema = {"name": "str", "age": "int", "score": "float", "active": "bool"}
        provider = SyntheticDataProvider(parameter_schema=schema)
        result = provider.generate(3)
        assert len(result) == 3
        for item in result:
            assert set(item.keys()) == {"name", "age", "score", "active"}

    def test_unsupported_type_returns_string_fallback(self):
        provider = SyntheticDataProvider(parameter_schema={"data": "complex"})
        result = provider.generate(2)
        for item in result:
            assert isinstance(item["data"], str)

    def test_correct_count_returned(self):
        provider = SyntheticDataProvider(parameter_schema={"x": "int"})
        for count in [1, 5, 10]:
            result = provider.generate(count)
            assert len(result) == count


# ---------------------------------------------------------------------------
# Pairwise distinctness (Req 9.1)
# ---------------------------------------------------------------------------

class TestDistinctness:
    def test_generated_data_pairwise_distinct(self):
        provider = SyntheticDataProvider(
            parameter_schema={"name": "str", "age": "int"}
        )
        result = provider.generate(10)
        # Convert to comparable form
        keys = [str(sorted(item.items())) for item in result]
        assert len(set(keys)) == len(keys)

    def test_single_item_is_trivially_distinct(self):
        provider = SyntheticDataProvider(parameter_schema={"x": "int"})
        result = provider.generate(1)
        assert len(result) == 1

    def test_bool_only_schema_handles_more_than_two(self):
        """With only bool fields, we can only get 2 distinct values.
        The fallback mechanism should handle this by adding _index."""
        provider = SyntheticDataProvider(parameter_schema={"flag": "bool"})
        result = provider.generate(3)
        assert len(result) == 3
        # All should be distinct (fallback adds _index for collisions)
        keys = [str(sorted(item.items())) for item in result]
        assert len(set(keys)) == 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_count_zero_raises_value_error(self):
        provider = SyntheticDataProvider(parameter_schema={"x": "int"})
        with pytest.raises(ValueError, match="count must be >= 1"):
            provider.generate(0)

    def test_negative_count_raises_value_error(self):
        provider = SyntheticDataProvider(parameter_schema={"x": "int"})
        with pytest.raises(ValueError):
            provider.generate(-1)

    def test_no_schema_no_generator_raises_value_error(self):
        provider = SyntheticDataProvider()
        with pytest.raises(ValueError, match="Either parameter_schema or custom_generator"):
            provider.generate(1)
