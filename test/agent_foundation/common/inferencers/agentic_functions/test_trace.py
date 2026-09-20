# pyre-strict

"""``AgenticFunctionTrace`` record + the per-function observability ``ContextVar``.

The trace has no argument fields at all, so a redacted argument can never appear
in it (redaction is structurally guaranteed, not merely filtered).
"""

from __future__ import annotations

import json
import unittest

import attr
from agent_foundation.common.inferencers.agentic_functions import AgenticFunctionTrace
from agent_foundation.common.inferencers.agentic_functions.trace import new_trace_var


class TraceTest(unittest.TestCase):
    def test_defaults(self) -> None:
        t = AgenticFunctionTrace()
        self.assertEqual(t.function, "")
        self.assertEqual(t.path, "")
        self.assertIsNone(t.prompt)
        self.assertIsNone(t.raw_text)
        self.assertEqual(t.stages_used, ())
        self.assertEqual(t.attempts, 0)
        self.assertEqual(t.errors, ())
        self.assertFalse(t.parsed)

    def test_trace_has_no_argument_fields(self) -> None:
        # The redaction guarantee: no field can carry a call argument's value.
        fields = set(vars(AgenticFunctionTrace()))
        self.assertNotIn("args", fields)
        self.assertNotIn("kwargs", fields)
        self.assertNotIn("arguments", fields)

    def test_new_trace_var_defaults_none_and_is_isolated(self) -> None:
        v1 = new_trace_var("agentic::a")
        v2 = new_trace_var("agentic::b")
        self.assertIsNone(v1.get())
        self.assertIsNone(v2.get())
        v1.set(AgenticFunctionTrace(function="a"))
        got = v1.get()
        assert got is not None
        self.assertEqual(got.function, "a")
        self.assertIsNone(v2.get())  # independent ContextVar


class TraceSerializationTest(unittest.TestCase):
    def test_result_field_defaults_to_none(self) -> None:
        self.assertIsNone(AgenticFunctionTrace().result)

    def test_to_dict_expands_nested_attrs_and_is_json_serializable(self) -> None:
        @attr.s(auto_attribs=True, frozen=True)
        class _Decision:
            repo: str = "fbsource"
            corpora: tuple[str, ...] = ("fbcode",)

        t = AgenticFunctionTrace(function="judge_code_scope", result=_Decision())
        d = t.to_dict()
        # A nested attrs result expands recursively; attrs turns tuples into lists.
        self.assertEqual(d["result"], {"repo": "fbsource", "corpora": ["fbcode"]})
        # The whole record round-trips through JSON (the persisted-artifact contract).
        round_tripped = json.loads(json.dumps(d, default=str))
        self.assertEqual(round_tripped["result"]["repo"], "fbsource")


if __name__ == "__main__":
    unittest.main()
