# pyre-strict

"""Inferencer resolution + config safety for the agentic-function framework.

The decorator hands the user an inferencer (it does not re-expose the ~35
framework knobs). This module turns the polymorphic ``inferencer=`` argument
into a lazily-built, cached ``InferencerBase`` while enforcing two safety
invariants the framework promises: a topology config with an unresolved
``MISSING`` node fails loud (never lets the literal ``"???"`` reach a leaf as a
path), and a config file path must resolve under a trusted root.
"""

from __future__ import annotations

import inspect
import threading
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple

import attr
from agent_foundation.common.inferencers.agentic_functions.errors import (
    AgenticFunctionConfigurationError,
)

# Dotted path (not a top-level import) so importing this module never triggers
# the inferencer import cascade — construction stays lazy (fbcode import safety).
_DEFAULT_INFERENCER_PATH: str = (
    "agent_foundation.common.inferencers.api_inferencers."
    "claude_api_inferencer.ClaudeApiInferencer"
)


@attr.s(auto_attribs=True, frozen=True)
class InferencerConfig:
    """Points the decorator at a YAML config (a leaf inferencer OR a topology).

    ``overrides`` are dotted-key, pre-interpolation values merged at highest
    precedence (``overrides > env > config_defaults > YAML``); a callable is
    invoked per call with the render feed as keyword arguments, so a config can
    depend on the call arguments. ``fresh_per_call`` re-instantiates a fresh
    subtree each call (via ``LazyConfigFactory``) so a topology's inner nodes are
    never shared across concurrent calls.
    """

    path: str
    overrides: Optional[Any] = None
    env_prefix: Optional[str] = None
    config_defaults: Optional[Mapping[str, Any]] = None
    fresh_per_call: bool = False


# ---------------------------------------------------------------------------
# Trusted config roots
# ---------------------------------------------------------------------------

_EXTRA_TRUSTED_ROOTS: List[Path] = []


def add_trusted_config_root(path: Any) -> None:
    """Allow config files under ``path`` (canonicalized). For advanced callers.

    A config path is decorator-authored, never a runtime argument or model
    output; this allowlist is defense-in-depth against a traversal that would
    load an arbitrary YAML as an inferencer.
    """
    _EXTRA_TRUSTED_ROOTS.append(Path(path).resolve())


def _trusted_roots() -> List[Path]:
    import agent_foundation

    af_pkg = Path(inspect.getfile(agent_foundation)).resolve().parent
    # agent_foundation/ (configs live under it) and its parent src/ dir.
    return [af_pkg, af_pkg.parent, *_EXTRA_TRUSTED_ROOTS]


def _is_relative_to(child: Path, root: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_config_path(path: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise AgenticFunctionConfigurationError(
            f"config path does not exist or is not a file: {resolved}"
        )
    roots = _trusted_roots()
    if not any(_is_relative_to(resolved, root) for root in roots):
        raise AgenticFunctionConfigurationError(
            f"config path {resolved} is outside the trusted roots "
            f"{[str(r) for r in roots]}; call add_trusted_config_root(...) to "
            f"allow it (config paths must be code-authored, never runtime input)"
        )
    return resolved


# ---------------------------------------------------------------------------
# MISSING guard
# ---------------------------------------------------------------------------


def assert_no_missing_keys(cfg: Any, source: str) -> None:
    """Raise if a resolved OmegaConf config still has ``MISSING`` node(s).

    ``load_config`` eagerly ``OmegaConf.resolve``s and returns the ``DictConfig``
    *without* ``throw_on_missing``; a MISSING node (and any node interpolating it)
    therefore survives as the MISSING sentinel and would stringify to the literal
    ``"???"`` at ``to_container`` time, silently reaching a leaf (e.g. a
    ``workspace.root`` path). ``OmegaConf.missing_keys`` enumerates the offending
    dotted keys and — unlike a post-hoc ``"???"`` scan — does not false-positive
    on a user-supplied literal ``"???"`` string.
    """
    from omegaconf import OmegaConf

    missing = OmegaConf.missing_keys(cfg)
    if missing:
        raise AgenticFunctionConfigurationError(
            f"config {source} has unresolved MISSING key(s): {sorted(missing)}; "
            f"supply them via InferencerConfig.overrides"
        )


def _load_and_guard(
    path: str,
    overrides: Optional[Mapping[str, Any]],
    env_prefix: Optional[str],
    config_defaults: Optional[Mapping[str, Any]],
) -> Any:
    from rich_python_utils.config_utils._instantiate import load_config

    cfg = load_config(
        path,
        overrides=dict(overrides) if overrides else None,
        env_prefix=env_prefix,
        config_defaults=dict(config_defaults) if config_defaults else None,
    )
    assert_no_missing_keys(cfg, path)
    return cfg


# ---------------------------------------------------------------------------
# inferencer_kwargs validation
# ---------------------------------------------------------------------------


def validate_inferencer_kwargs(cls: type, kwargs: Mapping[str, Any]) -> None:
    """Raise on an ``inferencer_kwargs`` key the class ``__init__`` won't accept.

    Silently dropping an unknown key is exactly the bug that once shipped
    un-templated Metamate prompts (a dropped ``template_variables``), so this
    fails closed. ``inspect.signature`` on an attrs class reports the generated
    ``__init__`` parameters (with attrs' underscore-stripped names), so it is more
    accurate than reconstructing them from ``attr.fields``.
    """
    if not kwargs:
        return
    params = inspect.signature(cls).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return  # accepts **kwargs — cannot validate
    valid = {
        name
        for name, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = set(kwargs) - valid
    if unknown:
        raise AgenticFunctionConfigurationError(
            f"unknown inferencer_kwargs {sorted(unknown)} for {cls.__name__}; "
            f"accepted keys: {sorted(valid)}"
        )


def _runs_one_host_call_at_a_time(inferencer: Any, run_context: Any) -> bool:
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase
    from agent_foundation.common.inferencers.run_context import host_pure_certified

    return (
        run_context is not None
        and not run_context.legacy_mint
        and isinstance(inferencer, InferencerBase)
        and not host_pure_certified(type(inferencer))
    )


def _import_symbol(dotted: str) -> Any:
    import importlib

    module_path, _, name = dotted.rpartition(".")
    if not module_path:
        raise AgenticFunctionConfigurationError(f"not a dotted import path: {dotted!r}")
    return getattr(importlib.import_module(module_path), name)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class InferencerProvider:
    """Lazily resolves + caches the inferencer behind the polymorphic spec.

    Nothing is constructed at decoration/import: ``__init__`` only stores the
    spec. The first ``get`` builds the inferencer under a double-checked lock and
    caches it (a prebuilt instance or a zero-arg factory's result is shared). A
    non-cacheable spec (``fresh_per_call`` or callable ``overrides``) is rebuilt
    on every ``get``.

    ``for_call`` is what a call runs on: calls of one agentic function can overlap
    (concurrent fan-out workers each judging their own task), and under a host run
    context an inferencer whose class is not host-pure certified runs one
    invocation at a time (``UncertifiedConcurrentUseError``), so such a call gets
    a ``fresh_instance()`` of the shared inferencer.
    """

    def __init__(
        self,
        spec: Any,
        inferencer_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._spec = spec
        self._kwargs: Mapping[str, Any] = dict(inferencer_kwargs or {})
        self._lock = threading.Lock()
        self._cached: Any = None
        self._has_cached: bool = False
        if spec is not None and self._kwargs:
            # inferencer_kwargs constructs the DEFAULT inferencer; for an explicit
            # spec, root-node attrs belong in the config/overrides — refuse the
            # ambiguity rather than silently ignore the kwargs.
            raise AgenticFunctionConfigurationError(
                "inferencer_kwargs is only valid with the default inferencer "
                "(inferencer=None); configure an explicit inferencer via its "
                "config/overrides instead"
            )

    def get(self, feed: Mapping[str, Any]) -> Any:
        builder, cacheable = self._builder(feed)
        if not cacheable:
            return builder()
        return self._shared(builder)

    def for_call(self, feed: Mapping[str, Any], run_context: Any) -> Any:
        """The inferencer one call at ``run_context`` runs on (see the class
        docstring)."""
        builder, cacheable = self._builder(feed)
        if not cacheable:
            return builder()
        shared = self._shared(builder)
        if not _runs_one_host_call_at_a_time(shared, run_context):
            return shared
        try:
            return shared.fresh_instance()
        except TypeError:
            # No construction recipe to copy from: the guard keeps refusing an
            # overlapping call on the shared instance.
            return shared

    def _shared(self, builder: Callable[[], Any]) -> Any:
        if not self._has_cached:
            with self._lock:
                if not self._has_cached:
                    self._cached = builder()
                    self._has_cached = True
        return self._cached

    def _builder(self, feed: Mapping[str, Any]) -> Tuple[Callable[[], Any], bool]:
        from agent_foundation.common.inferencers.inferencer_base import InferencerBase

        spec = self._spec
        if spec is None:
            cls = _import_symbol(_DEFAULT_INFERENCER_PATH)
            validate_inferencer_kwargs(cls, self._kwargs)
            kwargs = dict(self._kwargs)
            return (lambda: cls(**kwargs)), True
        if isinstance(spec, InferencerConfig):
            return self._config_builder(spec, feed)
        if isinstance(spec, str):
            from agent_foundation.common.configs.factories import load_inferencer

            name = spec
            return (lambda: load_inferencer(name)), True
        if isinstance(spec, Mapping):
            return self._mapping_builder(dict(spec))
        if isinstance(spec, InferencerBase):
            return (lambda: spec), True
        if callable(spec):
            # A zero-arg factory. Cacheable unless explicitly per-call.
            return (lambda: spec()), True
        raise AgenticFunctionConfigurationError(
            f"unsupported inferencer spec of type {type(spec).__name__}; expected "
            f"None | InferencerBase | str | InferencerConfig | Mapping | Callable"
        )

    def _config_builder(
        self, cfg_spec: InferencerConfig, feed: Mapping[str, Any]
    ) -> Tuple[Callable[[], Any], bool]:
        from rich_python_utils.config_utils._instantiate import instantiate

        path = str(_validate_config_path(cfg_spec.path))
        overrides = cfg_spec.overrides
        callable_overrides = callable(overrides) and not isinstance(overrides, Mapping)

        if cfg_spec.fresh_per_call or callable_overrides:
            resolved_overrides = (
                overrides(**feed) if callable_overrides else overrides  # type: ignore[operator]
            )
            cfg = _load_and_guard(
                path, resolved_overrides, cfg_spec.env_prefix, cfg_spec.config_defaults
            )
            if cfg_spec.fresh_per_call:
                factory = self._make_lazy_factory(cfg)
                return (lambda: factory()), False
            return (lambda: instantiate(cfg)), False

        cfg = _load_and_guard(
            path, overrides, cfg_spec.env_prefix, cfg_spec.config_defaults
        )
        return (lambda: instantiate(cfg)), True

    def _mapping_builder(
        self, mapping: Mapping[str, Any]
    ) -> Tuple[Callable[[], Any], bool]:
        from omegaconf import OmegaConf
        from rich_python_utils.config_utils._instantiate import instantiate

        cfg = OmegaConf.create(dict(mapping))
        assert_no_missing_keys(cfg, "<inline inferencer mapping>")
        return (lambda: instantiate(cfg)), True

    @staticmethod
    def _make_lazy_factory(cfg: Any) -> Callable[[], Any]:
        from omegaconf import OmegaConf
        from rich_python_utils.config_utils._lazy_config_factory import (
            LazyConfigFactory,
        )

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(config_dict, dict):
            raise AgenticFunctionConfigurationError(
                "fresh_per_call config did not resolve to a mapping"
            )
        return LazyConfigFactory(config_dict)
