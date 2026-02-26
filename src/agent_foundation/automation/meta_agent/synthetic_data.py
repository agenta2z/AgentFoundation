"""
Synthetic Data Provider for the Meta Agent Workflow pipeline.

Generates varied input data for development-time agent runs to ensure
diverse trace coverage. Supports custom generators or default random
generation based on a parameter schema.

Requirements: 9.1, 9.2, 9.3
"""

from __future__ import annotations

import random
import string
from typing import Any, Callable, Dict, List, Optional


class SyntheticDataProvider:
    """
    Generates synthetic input data for agent runs.

    Supports two modes:

    1. **Custom generator** — when *custom_generator* is provided it is
       called with the requested count and its result is returned directly.
    2. **Schema-based generation** — when *parameter_schema* is provided
       (a ``Dict[str, str]`` mapping field names to type hints such as
       ``"str"``, ``"int"``, ``"float"``, ``"bool"``), the provider
       generates random values for each field.

    All generated data sets are guaranteed to be pairwise distinct.

    Parameters
    ----------
    parameter_schema:
        Mapping of field names to type hint strings.
        Supported types: ``"str"``, ``"int"``, ``"float"``, ``"bool"``.
    custom_generator:
        A callable ``(int) -> List[Dict[str, Any]]`` that produces
        *count* data sets. When provided, *parameter_schema* is ignored.
    """

    # Supported type hint strings and their generators
    _TYPE_GENERATORS: Dict[str, Callable[[int], Any]] = {}

    def __init__(
        self,
        parameter_schema: Optional[Dict[str, str]] = None,
        custom_generator: Optional[Callable[[int], List[Dict[str, Any]]]] = None,
    ):
        self._parameter_schema = parameter_schema
        self._custom_generator = custom_generator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate *count* sets of synthetic input data.

        Uses *custom_generator* if provided, otherwise generates based
        on *parameter_schema*.  All returned data sets are pairwise
        distinct.

        Parameters
        ----------
        count:
            Number of data sets to generate (must be >= 1).

        Returns
        -------
        List of dicts, one per data set.

        Raises
        ------
        ValueError
            If *count* < 1, or if neither *parameter_schema* nor
            *custom_generator* is configured.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")

        if self._custom_generator is not None:
            return self._custom_generator(count)

        if self._parameter_schema is not None:
            return self._generate_from_schema(count)

        raise ValueError(
            "Either parameter_schema or custom_generator must be provided"
        )

    # ------------------------------------------------------------------
    # Schema-based generation
    # ------------------------------------------------------------------

    def _generate_from_schema(self, count: int) -> List[Dict[str, Any]]:
        """Generate *count* pairwise-distinct data sets from the schema."""
        seen: set = set()
        results: List[Dict[str, Any]] = []
        max_attempts = count * 20  # safety limit to avoid infinite loops
        attempts = 0

        while len(results) < count and attempts < max_attempts:
            attempts += 1
            item = self._generate_single(len(results))
            key = self._dict_key(item)
            if key not in seen:
                seen.add(key)
                results.append(item)

        # If we couldn't generate enough distinct items via randomness,
        # force uniqueness by appending an index suffix.
        if len(results) < count:
            results = self._force_distinct(count)

        return results

    def _generate_single(self, index: int) -> Dict[str, Any]:
        """Generate a single data set from the parameter schema."""
        assert self._parameter_schema is not None
        item: Dict[str, Any] = {}
        for field_name, type_hint in self._parameter_schema.items():
            item[field_name] = self._generate_value(type_hint, index)
        return item

    def _generate_value(self, type_hint: str, index: int) -> Any:
        """Generate a random value for the given type hint."""
        hint = type_hint.strip().lower()

        if hint == "str":
            length = random.randint(3, 12)
            return "".join(random.choices(string.ascii_lowercase, k=length))

        if hint == "int":
            return random.randint(-1000, 1000)

        if hint == "float":
            return round(random.uniform(-1000.0, 1000.0), 4)

        if hint == "bool":
            return random.choice([True, False])

        # Unsupported type — return a string representation with index
        return f"{type_hint}_value_{index}"

    def _force_distinct(self, count: int) -> List[Dict[str, Any]]:
        """
        Fallback: generate *count* items guaranteed distinct by
        incorporating the index into each value.

        For types with limited cardinality (e.g. bool), an ``_index``
        field is appended to guarantee uniqueness.
        """
        assert self._parameter_schema is not None
        seen: set = set()
        results: List[Dict[str, Any]] = []
        for i in range(count):
            item: Dict[str, Any] = {}
            for field_name, type_hint in self._parameter_schema.items():
                item[field_name] = self._deterministic_value(type_hint, i)
            key = self._dict_key(item)
            if key in seen:
                # Add an index field to force uniqueness
                item["_index"] = i
            seen.add(self._dict_key(item))
            results.append(item)
        return results

    @staticmethod
    def _deterministic_value(type_hint: str, index: int) -> Any:
        """Generate a deterministic value that is unique per index."""
        hint = type_hint.strip().lower()

        if hint == "str":
            return f"value_{index}"
        if hint == "int":
            return index
        if hint == "float":
            return float(index) + 0.1
        if hint == "bool":
            return index % 2 == 0

        return f"{type_hint}_{index}"

    @staticmethod
    def _dict_key(d: Dict[str, Any]) -> str:
        """Create a hashable key from a dict for distinctness checking."""
        return str(sorted(d.items()))
