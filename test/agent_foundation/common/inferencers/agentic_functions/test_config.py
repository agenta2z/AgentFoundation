# pyre-strict

"""Inferencer resolution + config safety: the polymorphic provider dispatch,
``inferencer_kwargs`` validation, the ``OmegaConf.missing_keys`` MISSING guard,
trusted-root enforcement, and cached vs. ``fresh_per_call`` construction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import attr
from agent_foundation.common.inferencers.agentic_functions import (
    AgenticFunctionConfigurationError,
)
from agent_foundation.common.inferencers.agentic_functions.config import (
    _DEFAULT_INFERENCER_PATH,
    _import_symbol,
    _is_relative_to,
    _trusted_roots,
    _validate_config_path,
    add_trusted_config_root,
    assert_no_missing_keys,
    InferencerConfig,
    InferencerProvider,
    validate_inferencer_kwargs,
)
from omegaconf import OmegaConf


class ProviderDispatchTest(unittest.TestCase):
    def test_mapping_spec_instantiates_and_caches(self) -> None:
        p = InferencerProvider({"_target_": "builtins.list"})
        a = p.get({})
        b = p.get({})
        self.assertEqual(a, [])
        self.assertIs(a, b)  # cacheable spec built once

    def test_callable_factory_is_lazy_then_cached(self) -> None:
        calls = {"n": 0}

        def factory() -> Any:
            calls["n"] += 1
            return object()

        p = InferencerProvider(factory)
        self.assertEqual(calls["n"], 0)  # nothing built at construction
        a = p.get({})
        self.assertEqual(calls["n"], 1)
        b = p.get({})
        self.assertEqual(calls["n"], 1)  # reused, not rebuilt
        self.assertIs(a, b)

    def test_none_spec_with_unknown_kwargs_raises_before_construction(self) -> None:
        # The default ClaudeApiInferencer is never constructed: validation runs in
        # _builder and rejects the bogus key first.
        p = InferencerProvider(None, inferencer_kwargs={"definitely_not_a_field": 1})
        with self.assertRaises(AgenticFunctionConfigurationError):
            p.get({})

    def test_spec_and_kwargs_conflict_raises_at_construction(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError):
            InferencerProvider(lambda: object(), inferencer_kwargs={"a": 1})

    def test_unsupported_spec_type_raises(self) -> None:
        # A bare non-callable object is not a valid spec (this is exactly why the
        # test fakes are injected via inferencer=lambda: fake).
        with self.assertRaises(AgenticFunctionConfigurationError):
            InferencerProvider(object()).get({})


class ValidateInferencerKwargsTest(unittest.TestCase):
    def test_toy_attrs_class_accepts_known_rejects_unknown(self) -> None:
        @attr.s(auto_attribs=True)
        class _Toy:
            a: int = 1
            b: str = "x"

        validate_inferencer_kwargs(_Toy, {})  # empty is always fine
        validate_inferencer_kwargs(_Toy, {"a": 5, "b": "y"})
        with self.assertRaises(AgenticFunctionConfigurationError):
            validate_inferencer_kwargs(_Toy, {"zzz": 1})

    def test_var_keyword_init_accepts_anything(self) -> None:
        class _Var:
            def __init__(self, **kwargs: Any) -> None: ...

        validate_inferencer_kwargs(_Var, {"anything": 1, "goes": 2})  # no raise

    def test_real_default_inferencer_kwargs_are_valid(self) -> None:
        # Regression for the Metamate judge: model_id / max_retry /
        # total_timeout_seconds must all be real ClaudeApiInferencer fields.
        cls = _import_symbol(_DEFAULT_INFERENCER_PATH)
        validate_inferencer_kwargs(
            cls, {"model_id": "x", "max_retry": 2, "total_timeout_seconds": 60}
        )
        with self.assertRaises(AgenticFunctionConfigurationError):
            validate_inferencer_kwargs(cls, {"definitely_not_a_field": 1})


class MissingGuardTest(unittest.TestCase):
    def test_clean_config_passes(self) -> None:
        assert_no_missing_keys(OmegaConf.create({"a": 1}), "src")  # no raise

    def test_missing_node_raises(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError):
            assert_no_missing_keys(OmegaConf.create({"a": "???"}), "src")

    def test_missing_keys_is_the_authoritative_check(self) -> None:
        # Why the guard uses missing_keys (not a post-hoc "???" scan): it walks
        # the config and returns the offending dotted key(s).
        self.assertEqual(
            set(OmegaConf.missing_keys(OmegaConf.create({"a": "???", "b": 2}))), {"a"}
        )

    def test_config_file_with_missing_key_raises_via_provider(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "leaf.yaml"
            path.write_text("_target_: builtins.dict\nfoo: ???\n")
            add_trusted_config_root(d)
            provider = InferencerProvider(InferencerConfig(path=str(path)))
            with self.assertRaises(AgenticFunctionConfigurationError):
                provider.get({})


class TrustedRootsTest(unittest.TestCase):
    def test_path_outside_roots_rejected_then_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.yaml"
            path.write_text("_target_: builtins.list\n")
            if any(_is_relative_to(path.resolve(), r) for r in _trusted_roots()):
                self.skipTest("tempdir already under a trusted root")
            with self.assertRaises(AgenticFunctionConfigurationError):
                _validate_config_path(str(path))
            add_trusted_config_root(d)
            self.assertEqual(_validate_config_path(str(path)), path.resolve())

    def test_missing_file_rejected(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError):
            _validate_config_path("/no/such/file/__nope__.yaml")


class ConfigResolutionTest(unittest.TestCase):
    def test_plain_config_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "leaf.yaml"
            path.write_text("_target_: builtins.list\n")
            add_trusted_config_root(d)
            provider = InferencerProvider(InferencerConfig(path=str(path)))
            self.assertIs(provider.get({}), provider.get({}))

    def test_fresh_per_call_yields_distinct_instances(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "leaf.yaml"
            path.write_text("_target_: builtins.list\n")
            add_trusted_config_root(d)
            provider = InferencerProvider(
                InferencerConfig(path=str(path), fresh_per_call=True)
            )
            a = provider.get({})
            b = provider.get({})
            self.assertEqual(a, [])
            self.assertIsNot(a, b)


class InferencerConfigDefaultsTest(unittest.TestCase):
    def test_defaults(self) -> None:
        c = InferencerConfig(path="/x")
        self.assertIsNone(c.overrides)
        self.assertIsNone(c.env_prefix)
        self.assertIsNone(c.config_defaults)
        self.assertFalse(c.fresh_per_call)


if __name__ == "__main__":
    unittest.main()
