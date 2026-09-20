# pyre-strict

"""``AgenticOutput`` decoding + the escalation carriers (``Agentic`` / ``escalate``)."""

from __future__ import annotations

import unittest
from typing import Any

from agent_foundation.common.inferencers.agentic_functions import (
    Agentic,
    AgenticOutput,
    escalate,
    ESCALATE,
    FromInference,
    ParseError,
)
from agent_foundation.common.inferencers.agentic_functions.output import _FromInference


class AgenticOutputTest(unittest.TestCase):
    def test_text_str_and_raw(self) -> None:
        out = AgenticOutput({"native": 1}, "the text")
        self.assertEqual(out.text, "the text")
        self.assertEqual(str(out), "the text")
        self.assertEqual(out.raw, {"native": 1})

    def test_parse_block_plain_absent_is_none(self) -> None:
        out = AgenticOutput(None, 'x\n```json foo\n{"a": 1}\n```\ny')
        self.assertEqual(out.parse_block("foo"), {"a": 1})
        self.assertIsNone(out.parse_block("bar"))

    def test_json_label_present(self) -> None:
        out = AgenticOutput(None, '```json foo\n{"a": 1}\n```')
        self.assertEqual(out.json("foo"), {"a": 1})

    def test_json_missing_label_raises(self) -> None:
        out = AgenticOutput(None, "no fence here")
        with self.assertRaises(ParseError):
            out.json("foo")

    def test_json_rejects_duplicate_keys(self) -> None:
        out = AgenticOutput(None, '```json foo\n{"a": 1, "a": 2}\n```')
        with self.assertRaises(ParseError):
            out.json("foo")

    def test_json_rejects_nonfinite(self) -> None:
        out = AgenticOutput(None, '```json foo\n{"a": NaN}\n```')
        with self.assertRaises(ParseError):
            out.json("foo")

    def test_json_no_label_whole_text(self) -> None:
        out = AgenticOutput(None, '{"a": 1}')
        self.assertEqual(out.json(), {"a": 1})


class EscalationCarrierTest(unittest.TestCase):
    def test_escalate_singleton_is_value_equal_but_distinct(self) -> None:
        self.assertIsInstance(ESCALATE, Agentic)
        self.assertEqual(ESCALATE, Agentic())  # frozen attrs value equality
        self.assertIsNot(ESCALATE, Agentic())  # not interned

    def test_escalate_factory_carries_feed_and_parser(self) -> None:
        def _p(out: AgenticOutput) -> Any:
            return out

        a = escalate(feed={"k": "v"}, parser=_p)
        self.assertIsInstance(a, Agentic)
        self.assertEqual(dict(a.feed), {"k": "v"})
        self.assertIs(a.parser, _p)

    def test_escalate_defaults(self) -> None:
        a = escalate()
        self.assertEqual(dict(a.feed), {})
        self.assertIsNone(a.parser)

    def test_from_inference_is_singleton(self) -> None:
        self.assertIs(_FromInference(), FromInference)


if __name__ == "__main__":
    unittest.main()
