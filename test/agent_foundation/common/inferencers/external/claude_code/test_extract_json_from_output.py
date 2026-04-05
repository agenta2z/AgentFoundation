#!/usr/bin/env python3
"""Unit tests for ClaudeCodeCliInferencer._extract_json_from_output().

Validates the dict guard: json.loads() can return int, str, list, bool, or
None for valid JSON scalars/arrays. Only dict results should be returned;
everything else must fall through to None so parse_output() uses the raw
text fallback path.

Usage:
    buck2 run fbcode//rankevolve/test/agentic_foundation:test_extract_json_from_output
"""

import unittest

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
    ClaudeCodeCliInferencer,
)


class ExtractJsonFromOutputTest(unittest.TestCase):
    """Tests for _extract_json_from_output dict guard."""

    def setUp(self) -> None:
        self.inferencer = ClaudeCodeCliInferencer(target_path="/tmp")

    # === Valid dict inputs (should return dict) ===

    def test_clean_json_object(self) -> None:
        stdout = '{"result": "hello", "session_id": "abc123"}'
        result = self.inferencer._extract_json_from_output(stdout)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["result"], "hello")
        self.assertEqual(result["session_id"], "abc123")

    def test_json_object_with_whitespace(self) -> None:
        stdout = '  \n {"result": "world"} \n  '
        result = self.inferencer._extract_json_from_output(stdout)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["result"], "world")

    def test_json_object_with_prefix_noise(self) -> None:
        stdout = 'Some banner text\n{"result": "extracted"}'
        result = self.inferencer._extract_json_from_output(stdout)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["result"], "extracted")

    def test_json_object_with_suffix_noise(self) -> None:
        stdout = '{"result": "ok"}\nSome trailing text'
        result = self.inferencer._extract_json_from_output(stdout)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["result"], "ok")

    def test_json_object_surrounded_by_noise(self) -> None:
        stdout = 'prefix\n{"key": "value"}\nsuffix'
        result = self.inferencer._extract_json_from_output(stdout)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["key"], "value")

    def test_nested_json_object(self) -> None:
        stdout = '{"result": "ok", "usage": {"input": 100, "output": 50}}'
        result = self.inferencer._extract_json_from_output(stdout)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["usage"]["input"], 100)

    def test_empty_json_object(self) -> None:
        stdout = "{}"
        result = self.inferencer._extract_json_from_output(stdout)
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {})

    # === Non-dict JSON scalars (should return None — the dict guard) ===

    def test_integer_returns_none(self) -> None:
        """json.loads('4') returns int(4). Must not crash parse_output()."""
        self.assertIsNone(self.inferencer._extract_json_from_output("4"))

    def test_negative_integer_returns_none(self) -> None:
        self.assertIsNone(self.inferencer._extract_json_from_output("-42"))

    def test_float_returns_none(self) -> None:
        self.assertIsNone(self.inferencer._extract_json_from_output("3.14"))

    def test_string_returns_none(self) -> None:
        """json.loads('"hello"') returns str('hello')."""
        self.assertIsNone(self.inferencer._extract_json_from_output('"hello"'))

    def test_boolean_true_returns_none(self) -> None:
        self.assertIsNone(self.inferencer._extract_json_from_output("true"))

    def test_boolean_false_returns_none(self) -> None:
        self.assertIsNone(self.inferencer._extract_json_from_output("false"))

    def test_null_returns_none(self) -> None:
        self.assertIsNone(self.inferencer._extract_json_from_output("null"))

    def test_json_array_returns_none(self) -> None:
        """json.loads('[1,2,3]') returns list. Not a dict."""
        self.assertIsNone(self.inferencer._extract_json_from_output("[1, 2, 3]"))

    def test_json_array_of_objects_returns_none(self) -> None:
        self.assertIsNone(
            self.inferencer._extract_json_from_output('[{"a": 1}, {"b": 2}]')
        )

    # === Non-dict with embedded braces (fallback extraction should still guard) ===

    def test_integer_with_braces_in_surrounding_text(self) -> None:
        """Ensure fallback extraction with {/} doesn't return a non-dict."""
        stdout = "The answer is 4\nSome text with { and } chars"
        result = self.inferencer._extract_json_from_output(stdout)
        self.assertIsNone(result)

    def test_array_with_embedded_dict_extracts_inner_dict(self) -> None:
        """Fallback finds { and } inside array and extracts inner dict."""
        stdout = '[{"nested": true}]'
        result = self.inferencer._extract_json_from_output(stdout)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["nested"])

    # === Empty / whitespace / garbage inputs (should return None) ===

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(self.inferencer._extract_json_from_output(""))

    def test_whitespace_only_returns_none(self) -> None:
        self.assertIsNone(self.inferencer._extract_json_from_output("   \n\t  "))

    def test_none_input_returns_none(self) -> None:
        self.assertIsNone(self.inferencer._extract_json_from_output(None))

    def test_plain_text_returns_none(self) -> None:
        self.assertIsNone(
            self.inferencer._extract_json_from_output("Hello, world!")
        )

    def test_invalid_json_returns_none(self) -> None:
        self.assertIsNone(
            self.inferencer._extract_json_from_output("{invalid json}")
        )

    def test_multiline_plain_text_returns_none(self) -> None:
        stdout = "Line 1\nLine 2\nLine 3\n"
        self.assertIsNone(self.inferencer._extract_json_from_output(stdout))

    # === Real-world streaming outputs that triggered the original bug ===

    def test_numeric_answer_from_streaming(self) -> None:
        """Regression: 'What is 2+2?' in streaming mode outputs '4\\n'."""
        self.assertIsNone(self.inferencer._extract_json_from_output("4\n"))

    def test_word_answer_from_streaming(self) -> None:
        """Streaming text like 'Four\\n' is not valid JSON at all."""
        self.assertIsNone(self.inferencer._extract_json_from_output("Four\n"))

    def test_multiline_streaming_response(self) -> None:
        stdout = "Python is a programming language.\nIt is widely used.\n"
        self.assertIsNone(self.inferencer._extract_json_from_output(stdout))

    # === Real-world JSON mode outputs (should return dict) ===

    def test_real_claude_json_output(self) -> None:
        """Typical --output-format json response from Claude CLI."""
        stdout = (
            '{"result": "Four", "session_id": "abc-123", '
            '"is_error": false, "total_cost_usd": 0.01, '
            '"num_turns": 1, "duration_ms": 5000}'
        )
        result = self.inferencer._extract_json_from_output(stdout)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["result"], "Four")
        self.assertEqual(result["session_id"], "abc-123")
        self.assertFalse(result["is_error"])

    def test_real_claude_json_error_output(self) -> None:
        stdout = '{"result": "", "is_error": true, "session_id": "xyz-789"}'
        result = self.inferencer._extract_json_from_output(stdout)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["is_error"])


if __name__ == "__main__":
    unittest.main()
