# pyre-strict

"""Tests for :class:`FunctionInferencer` — a plain callable as an inferencer."""

import asyncio
import unittest
from typing import Any

# Populate the alias registry so ``Function`` resolves.
import agent_foundation.common.configs.registered_targets  # noqa: F401
from agent_foundation.common.inferencers.agentic_functions.config import _import_symbol
from agent_foundation.common.inferencers.function_inferencer import FunctionInferencer
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from rich_python_utils.config_utils._registry import resolve_target


def _shout(text: str) -> str:
    return text.upper()


async def _ashout(text: str) -> str:
    return text.upper()


class FunctionInferencerUnitTest(unittest.TestCase):
    """The ``_infer`` / ``_ainfer`` primitives in isolation."""

    def test_infer_returns_func_result_unchanged(self) -> None:
        payload = {"a": 1}
        fi = FunctionInferencer(func=lambda x: x)
        self.assertIs(fi._infer(payload), payload)

    def test_infer_raises_on_coroutine_func(self) -> None:
        fi = FunctionInferencer(func=_ashout)
        with self.assertRaises(TypeError):
            fi._infer("hi")

    def test_ainfer_awaits_coroutine_func(self) -> None:
        fi = FunctionInferencer(func=_ashout)
        self.assertEqual(asyncio.run(fi._ainfer("hi")), "HI")

    def test_ainfer_runs_sync_func(self) -> None:
        fi = FunctionInferencer(func=_shout)
        self.assertEqual(asyncio.run(fi._ainfer("hi")), "HI")


class FunctionInferencerPipelineTest(unittest.TestCase):
    """The wrapper is a real ``InferencerBase`` — it runs through public infer."""

    def test_is_inferencer_base(self) -> None:
        self.assertIsInstance(FunctionInferencer(func=_shout), InferencerBase)

    def test_public_infer_roundtrip(self) -> None:
        self.assertEqual(FunctionInferencer(func=_shout).infer("hi"), "HI")

    def test_public_ainfer_roundtrip_sync_func(self) -> None:
        fi = FunctionInferencer(func=_shout)
        self.assertEqual(asyncio.run(fi.ainfer("hi")), "HI")

    def test_public_ainfer_roundtrip_async_func(self) -> None:
        fi = FunctionInferencer(func=_ashout)
        self.assertEqual(asyncio.run(fi.ainfer("hi")), "HI")

    def test_public_infer_propagates_func_exception(self) -> None:
        # A raising func propagates its ORIGINAL exception out of the public
        # infer() path unwrapped (no InferenceError wrapping, no swallow-to-None).
        # This is the contract a caller relies on to fail closed on its own error.
        def boom(_: Any) -> Any:
            raise ValueError("kaboom")

        fi = FunctionInferencer(func=boom)
        with self.assertRaises(ValueError) as ctx:
            fi.infer("x")
        self.assertIn("kaboom", str(ctx.exception))

    def test_public_ainfer_propagates_func_exception(self) -> None:
        async def aboom(_: Any) -> Any:
            raise ValueError("akaboom")

        fi = FunctionInferencer(func=aboom)
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(fi.ainfer("x"))
        self.assertIn("akaboom", str(ctx.exception))


class FunctionInferencerConstructionTest(unittest.TestCase):
    """``func`` is required and must be callable."""

    def test_missing_func_raises(self) -> None:
        with self.assertRaises(TypeError):
            FunctionInferencer()  # type: ignore[call-arg]

    def test_non_callable_func_raises(self) -> None:
        with self.assertRaises(TypeError):
            FunctionInferencer(func=42)  # type: ignore[arg-type]


class FunctionInferencerAliasTest(unittest.TestCase):
    """The ``Function`` registry alias resolves to the class."""

    def test_alias_resolves_to_the_class(self) -> None:
        resolved: Any = _import_symbol(resolve_target("Function"))
        self.assertIs(resolved, FunctionInferencer)


if __name__ == "__main__":
    unittest.main()
