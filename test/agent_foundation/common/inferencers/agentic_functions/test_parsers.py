# pyre-strict

"""Stage-1 resolution, return-annotation defaults, body-as-parser, and the
explicit parse-retry loop (sync AND async) with ``fallback``."""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import attr
from agent_foundation.common.inferencers.agentic_functions import (
    agentic_function,
    AgenticFunctionConfigurationError,
    AgenticOutput,
    escalate,
    ParseError,
)
from agent_foundation.common.inferencers.agentic_functions.parsers import (
    annotation_kind,
    default_parser,
    resolve_parser,
)

from ._helpers import FakeInferencer


# --- Struct return types for the FF2 auto-construction tests ---------------
# Defined at module level so `typing.get_type_hints(fn)` (which resolves against
# the function's module globals) can evaluate a `-> Point` string annotation
# back to the real class under `from __future__ import annotations`.


@dataclass
class Point:
    x: int
    y: int


@attr.s(auto_attribs=True, frozen=True)
class AttrPoint:
    x: int
    y: int


class PlainClass:
    """A non-struct class — no attrs/dataclass/pydantic markers → ``"other"``."""


class FakePydanticV2:
    """Duck-typed pydantic v2 model (``model_validate`` + ``model_fields``).

    Avoids a real pydantic dependency while proving ``_struct_constructor``
    selects the v2 branch: ``model_validate`` stamps ``via`` so a test can assert
    that entry point (not a plain ``cls(**data)``) was used.
    """

    model_fields: Dict[str, Any] = {"x": None}

    def __init__(self, x: int, via: str = "init") -> None:
        self.x = x
        self.via = via

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "FakePydanticV2":
        return cls(via="model_validate", **data)


class FakePydanticV1:
    """Duck-typed pydantic v1 model (``parse_obj`` + ``__fields__``)."""

    __fields__: Dict[str, Any] = {"x": None}

    def __init__(self, x: int, via: str = "init") -> None:
        self.x = x
        self.via = via

    @classmethod
    def parse_obj(cls, data: Dict[str, Any]) -> "FakePydanticV1":
        return cls(via="parse_obj", **data)


class Stage1ResolutionTest(unittest.TestCase):
    def test_callable_passthrough(self) -> None:
        p = resolve_parser(lambda out: out.text.upper())
        self.assertEqual(p(AgenticOutput(None, "ok")), "OK")

    def test_sequence_composes_right_to_left(self) -> None:
        # [outer, inner] -> inner receives the AgenticOutput first, then outer.
        p = resolve_parser([lambda s: s + "!", lambda out: out.text])
        self.assertEqual(p(AgenticOutput(None, "ok")), "ok!")

    def test_single_element_sequence(self) -> None:
        p = resolve_parser([lambda out: out.text])
        self.assertEqual(p(AgenticOutput(None, "solo")), "solo")

    def test_empty_sequence_raises(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError):
            resolve_parser([])

    def test_non_parser_spec_raises(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError):
            resolve_parser(123)


class AnnotationDefaultTest(unittest.TestCase):
    def test_int_strict(self) -> None:
        def f() -> int: ...

        self.assertEqual(default_parser(f)(AgenticOutput(None, " 7 ")), 7)
        with self.assertRaises(ParseError):
            default_parser(f)(AgenticOutput(None, "The answer is 7"))

    def test_float(self) -> None:
        def f() -> float: ...

        self.assertEqual(default_parser(f)(AgenticOutput(None, "7.5")), 7.5)

    def test_bool(self) -> None:
        def f() -> bool: ...

        self.assertTrue(default_parser(f)(AgenticOutput(None, "yes")))
        self.assertFalse(default_parser(f)(AgenticOutput(None, "no")))
        with self.assertRaises(ParseError):
            default_parser(f)(AgenticOutput(None, "maybe"))

    def test_dict(self) -> None:
        def f() -> dict: ...

        self.assertEqual(default_parser(f)(AgenticOutput(None, '{"a": 1}')), {"a": 1})

    def test_str_and_unannotated(self) -> None:
        def f() -> str: ...

        def g(): ...

        self.assertEqual(default_parser(f)(AgenticOutput(None, "hi")), "hi")
        self.assertEqual(default_parser(g)(AgenticOutput(None, "hi")), "hi")

    def test_agentic_output_identity(self) -> None:
        def f() -> AgenticOutput: ...

        out = AgenticOutput(None, "hi")
        self.assertIs(default_parser(f)(out), out)


class BodyAsParserTest(unittest.TestCase):
    def test_post_parser_receives_agentic_output_when_no_stage1(self) -> None:
        # Regresses the _bind fix: a required keyword-only `response` (no default).
        fake = FakeInferencer("ok")

        @agentic_function(inferencer=lambda: fake, template_string="{{ task }}")
        def echo(task: str, *, response: Any) -> str:
            return response.text.upper()

        self.assertEqual(echo("hi"), "OK")
        self.assertEqual(fake.calls, 1)

    def test_stage1_then_body_compose(self) -> None:
        fake = FakeInferencer("  ok  ")

        @agentic_function(
            inferencer=lambda: fake,
            template_string="{{ task }}",
            parser=lambda out: out.text.strip(),
        )
        def f(task: str, *, response: Any) -> str:
            return response.upper()

        self.assertEqual(f("hi"), "OK")


class ParseRetryLoopTest(unittest.TestCase):
    def test_sync_exactly_max_retries_plus_one_attempts(self) -> None:
        fake = FakeInferencer("nope")

        @agentic_function(
            inferencer=lambda: fake, template_string="{{ x }}", parse_max_retries=2
        )
        def f(x: Any) -> int:
            return escalate()

        with self.assertRaises(ParseError):
            f("a")
        self.assertEqual(fake.calls, 3)

    def test_async_exactly_max_retries_plus_one_attempts(self) -> None:
        fake = FakeInferencer("nope")

        @agentic_function(
            inferencer=lambda: fake, template_string="{{ x }}", parse_max_retries=2
        )
        async def f(x: Any) -> int:
            return escalate()

        with self.assertRaises(ParseError):
            asyncio.run(f("a"))
        self.assertEqual(fake.calls, 3)

    def test_retry_on_widens_the_retry_set(self) -> None:
        class Boom(Exception):
            pass

        fake = FakeInferencer("ok")

        @agentic_function(
            inferencer=lambda: fake,
            template_string="{{ x }}",
            parse_max_retries=1,
            retry_on=(Boom,),
            fallback="FB",
        )
        def f(x: Any, *, response: Any) -> str:
            raise Boom("bad")

        self.assertEqual(f("a"), "FB")
        self.assertEqual(fake.calls, 2)

    def test_body_exception_outside_retry_on_propagates(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(
            inferencer=lambda: fake, template_string="{{ x }}", parse_max_retries=1
        )
        def f(x: Any, *, response: Any) -> str:
            raise TypeError("a real bug")

        with self.assertRaises(TypeError):
            f("a")
        self.assertEqual(fake.calls, 1)


class FallbackTest(unittest.TestCase):
    def test_value(self) -> None:
        fake = FakeInferencer("nope")

        @agentic_function(
            inferencer=lambda: fake, template_string="{{ x }}", fallback=99
        )
        def f(x: Any) -> int:
            return escalate()

        self.assertEqual(f("a"), 99)

    def test_callable_zero_arg(self) -> None:
        fake = FakeInferencer("nope")

        @agentic_function(
            inferencer=lambda: fake, template_string="{{ x }}", fallback=lambda: 42
        )
        def f(x: Any) -> int:
            return escalate()

        self.assertEqual(f("a"), 42)

    def test_callable_arg_aware(self) -> None:
        fake = FakeInferencer("nope")

        @agentic_function(
            inferencer=lambda: fake,
            template_string="{{ a }}{{ b }}",
            fallback=lambda a, b: a * 10 + b,
        )
        def f(a: int, b: int) -> int:
            return escalate()

        self.assertEqual(f(2, 3), 23)

    def test_raise_reraises_last_error(self) -> None:
        fake = FakeInferencer("nope")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> int:
            return escalate()

        with self.assertRaises(ParseError):
            f("a")


class StructClassificationTest(unittest.TestCase):
    """``annotation_kind`` recognizes attrs / dataclass / pydantic as ``struct``."""

    def test_dataclass_is_struct(self) -> None:
        self.assertEqual(annotation_kind(Point), "struct")

    def test_attrs_is_struct(self) -> None:
        self.assertEqual(annotation_kind(AttrPoint), "struct")

    def test_pydantic_v2_is_struct(self) -> None:
        self.assertEqual(annotation_kind(FakePydanticV2), "struct")

    def test_pydantic_v1_is_struct(self) -> None:
        self.assertEqual(annotation_kind(FakePydanticV1), "struct")

    def test_plain_class_is_other(self) -> None:
        self.assertEqual(annotation_kind(PlainClass), "other")

    def test_optional_struct_is_other(self) -> None:
        # Optional[...] is intentionally left to raw text in this fast-follow
        # (it interacts with escalate_on_none); supply a parser= for it.
        self.assertEqual(annotation_kind(Optional[Point]), "other")

    def test_list_of_struct_is_other(self) -> None:
        self.assertEqual(annotation_kind(List[Point]), "other")

    def test_stringized_struct_is_other(self) -> None:
        # A bare string annotation (degraded get_type_hints fallback) cannot be
        # introspected, so it falls through to raw text rather than misconstructing.
        self.assertEqual(annotation_kind("Point"), "other")


class StructConstructionTest(unittest.TestCase):
    """``default_parser`` builds the declared struct from a JSON object."""

    def test_dataclass_constructed_from_json(self) -> None:
        def f() -> Point: ...

        p = default_parser(f)(AgenticOutput(None, '{"x": 1, "y": 2}'))
        self.assertEqual((p.x, p.y), (1, 2))

    def test_attrs_constructed_from_json(self) -> None:
        def f() -> AttrPoint: ...

        p = default_parser(f)(AgenticOutput(None, '{"x": 3, "y": 4}'))
        self.assertEqual((p.x, p.y), (3, 4))

    def test_pydantic_v2_uses_model_validate(self) -> None:
        def f() -> FakePydanticV2: ...

        p = default_parser(f)(AgenticOutput(None, '{"x": 5}'))
        self.assertEqual(p.x, 5)
        self.assertEqual(p.via, "model_validate")

    def test_pydantic_v1_uses_parse_obj(self) -> None:
        def f() -> FakePydanticV1: ...

        p = default_parser(f)(AgenticOutput(None, '{"x": 6}'))
        self.assertEqual(p.x, 6)
        self.assertEqual(p.via, "parse_obj")

    def test_missing_field_raises_parse_error(self) -> None:
        def f() -> Point: ...

        with self.assertRaises(ParseError):
            default_parser(f)(AgenticOutput(None, '{"x": 1}'))

    def test_extra_field_raises_parse_error(self) -> None:
        def f() -> Point: ...

        with self.assertRaises(ParseError):
            default_parser(f)(AgenticOutput(None, '{"x": 1, "y": 2, "z": 3}'))

    def test_non_object_json_raises_parse_error(self) -> None:
        def f() -> Point: ...

        with self.assertRaises(ParseError):
            default_parser(f)(AgenticOutput(None, "[1, 2]"))

    def test_invalid_json_raises_parse_error(self) -> None:
        def f() -> Point: ...

        with self.assertRaises(ParseError):
            default_parser(f)(AgenticOutput(None, "not json at all"))

    def test_end_to_end_pre_attempt_constructs_struct(self) -> None:
        # A pure-agentic body (no parser=, no response slot) escalates, infers,
        # and the return-annotation default constructs the struct end to end.
        fake = FakeInferencer('{"x": 7, "y": 8}')

        @agentic_function(inferencer=lambda: fake, template_string="{{ spec }}")
        def make(spec: str) -> Point:
            return escalate()

        p = make("origin")
        self.assertEqual((p.x, p.y), (7, 8))
        self.assertEqual(fake.calls, 1)

    def test_end_to_end_bad_reply_falls_back(self) -> None:
        # A malformed reply → ParseError → fallback (no partial object leaks).
        fake = FakeInferencer("not json")
        sentinel = Point(0, 0)

        @agentic_function(
            inferencer=lambda: fake, template_string="{{ spec }}", fallback=sentinel
        )
        def make(spec: str) -> Point:
            return escalate()

        self.assertIs(make("origin"), sentinel)


if __name__ == "__main__":
    unittest.main()
