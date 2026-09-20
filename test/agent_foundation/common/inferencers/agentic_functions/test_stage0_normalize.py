# pyre-strict

"""Stage-0 normalization: ``extract_result_text`` over heterogeneous shapes."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent_foundation.common.response_parsers.result_text import extract_result_text


class ExtractResultTextTest(unittest.TestCase):
    def test_none_is_empty(self) -> None:
        self.assertEqual(extract_result_text(None), "")

    def test_plain_string(self) -> None:
        self.assertEqual(extract_result_text("hello"), "hello")

    def test_base_response_attr(self) -> None:
        self.assertEqual(extract_result_text(SimpleNamespace(base_response="A")), "A")

    def test_result_attr(self) -> None:
        self.assertEqual(extract_result_text(SimpleNamespace(result="B")), "B")

    def test_output_attr(self) -> None:
        self.assertEqual(extract_result_text(SimpleNamespace(output="C")), "C")

    def test_empty_base_response_falls_through_to_result(self) -> None:
        obj = SimpleNamespace(base_response="", result="R")
        self.assertEqual(extract_result_text(obj), "R")

    def test_single_element_tuple_recurses(self) -> None:
        self.assertEqual(extract_result_text(("solo",)), "solo")

    def test_multi_worker_tuple_serializes_all(self) -> None:
        self.assertEqual(
            extract_result_text(("A", "B")),
            "### Worker 1\n\nA\n\n### Worker 2\n\nB",
        )

    def test_tuple_drops_none_then_recurses(self) -> None:
        self.assertEqual(extract_result_text(("A", None)), "A")

    def test_empty_tuple_is_empty(self) -> None:
        self.assertEqual(extract_result_text(()), "")


if __name__ == "__main__":
    unittest.main()
