# pyre-strict

"""Co-located template resolution, rendering, and variable validation.

A prompt comes from exactly one of three sources, selected by the *type* of the
``template`` argument — never by inspecting a string's contents:

* ``template_string=<str>`` — inline Jinja2 source.
* ``template=<str | os.PathLike>`` — a file next to the decorated function.
* ``template=<TemplateManager>`` — a shared registry, addressed by
  ``template_key`` / ``template_root_space`` / ``template_master_version``.

Each is wrapped in a :class:`TemplateSource` so the decorator renders through a
single call whatever the origin. A manager renders through its *own* formatter
rather than being read as raw text: a ``TemplateManager`` may be configured for
Handlebars, and running that text through Jinja2 would mis-render it silently.
"""

from __future__ import annotations

import abc
import copy
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from agent_foundation.common.inferencers.agentic_functions.errors import (
    AgenticFunctionConfigurationError,
)
from rich_python_utils.string_utils.formatting.jinja2_format import (
    extract_variables,
    format_template,
)
from rich_python_utils.string_utils.formatting.template_manager import TemplateManager

# Jinja2 globals that format_template injects itself (not caller-supplied args).
_HELPER_GLOBALS: frozenset[str] = frozenset(
    {"currentDate", "currentTime", "currentDateTime"}
)

# Planted as the fallback template on a throwaway copy of a manager so that a
# key which misses is distinguishable from one that resolves. Nothing weaker
# works: ``get_raw_template`` never raises and silently returns
# ``default_template`` on a miss, and the manager's own ``strict_lookup`` only
# fires when that default is *itself empty* — so a manager carrying a non-empty
# default would quietly render the wrong prompt.
_UNRESOLVED: str = "\x00__agentic_function_unresolved_template__\x00"

_text_cache: dict[Path, str] = {}


class TemplateSource(abc.ABC):
    """Where one decorated function's prompt text comes from."""

    @abc.abstractmethod
    def validate(self, allowed: "frozenset[str]") -> None:
        """Raise if this template cannot serve a function with ``allowed`` params."""

    @abc.abstractmethod
    def render(self, feed: Mapping[str, Any]) -> str:
        """Render the prompt for a single call."""


class TextTemplateSource(TemplateSource):
    """Inline source, or a file read once from disk — always Jinja2."""

    def __init__(self, text: str, source: Optional[Path] = None) -> None:
        self.text = text
        self.source = source

    def validate(self, allowed: "frozenset[str]") -> None:
        validate_template_variables(self.text, set(allowed), source=self.source)

    def render(self, feed: Mapping[str, Any]) -> str:
        return format_template(self.text, feed=dict(feed))


class ManagerTemplateSource(TemplateSource):
    """A shared ``TemplateManager`` registry entry addressed by key."""

    def __init__(
        self,
        manager: TemplateManager,
        *,
        template_key: Optional[str] = None,
        template_root_space: Optional[str] = None,
        template_master_version: Optional[str] = None,
    ) -> None:
        self.manager = manager
        self.template_key = template_key
        self.template_root_space = template_root_space
        self.template_master_version = template_master_version

    def validate(self, allowed: "frozenset[str]") -> None:
        """Raise if the key does not resolve to a real template.

        The template's declared variables are deliberately *not* checked against
        ``allowed``: a registry template may legitimately reference predefined
        variables and components that the manager resolves itself, which are not
        call arguments of the decorated function.
        """
        probe = copy.copy(self.manager)
        probe.default_template = _UNRESOLVED
        raw = probe.get_raw_template(
            self.template_key,
            active_template_root_space=self.template_root_space,
            master_version=self.template_master_version,
        )
        if raw is None or raw == _UNRESOLVED:
            raise AgenticFunctionConfigurationError(
                f"template_key={self.template_key!r} "
                f"(template_root_space={self.template_root_space!r}) does not "
                f"resolve in the supplied TemplateManager; rendering it would "
                f"silently fall back to the manager's default template"
            )

    def render(self, feed: Mapping[str, Any]) -> str:
        # ``feed=`` and never ``**feed``: the keys here are the decorated
        # function's own parameter names, so splatting would let a parameter
        # called e.g. ``master_version`` or ``formatter`` bind to the manager's
        # like-named keyword instead of becoming a template variable.
        rendered = self.manager(
            self.template_key,
            feed=dict(feed),
            active_template_root_space=self.template_root_space,
            master_version=self.template_master_version,
        )
        if not isinstance(rendered, str):
            raise AgenticFunctionConfigurationError(
                f"TemplateManager returned {type(rendered).__name__}, not str; a "
                f"multi-response template cannot back an @agentic_function"
            )
        return rendered


def resolve_template_source(
    fn: Callable[..., Any],
    *,
    template: Any = None,
    template_string: Optional[str] = None,
    template_key: Optional[str] = None,
    template_root_space: Optional[str] = None,
    template_master_version: Optional[str] = None,
) -> TemplateSource:
    """Select the prompt source by type; reject anything else loudly."""
    if (template is None) == (template_string is None):
        raise AgenticFunctionConfigurationError(
            "exactly one of template= or template_string= must be provided"
        )

    if isinstance(template, TemplateManager):
        if not template_key and not template_root_space:
            raise AgenticFunctionConfigurationError(
                "template=<TemplateManager> needs template_key= (or "
                "template_root_space=) to address a specific template"
            )
        return ManagerTemplateSource(
            template,
            template_key=template_key,
            template_root_space=template_root_space,
            template_master_version=template_master_version,
        )

    manager_only = {
        "template_key": template_key,
        "template_root_space": template_root_space,
        "template_master_version": template_master_version,
    }
    supplied = sorted(name for name, value in manager_only.items() if value is not None)
    if supplied:
        raise AgenticFunctionConfigurationError(
            f"{supplied} are only valid with template=<TemplateManager>"
        )

    if template_string is not None:
        return TextTemplateSource(template_string)

    if isinstance(template, (str, os.PathLike)):
        text, source = resolve_template_text(fn, template=str(template))
        return TextTemplateSource(text, source)

    raise AgenticFunctionConfigurationError(
        f"unsupported template spec of type {type(template).__name__}; expected "
        f"TemplateManager | str | os.PathLike"
    )


def resolve_template_text(
    fn: Callable[..., Any],
    *,
    template: Optional[str] = None,
    template_string: Optional[str] = None,
) -> tuple[str, Optional[Path]]:
    """Return ``(raw_template_text, source_path_or_None)`` for a text template.

    ``template_string`` is inline text (no path). ``template`` is resolved
    module-relative to the *defining* function's file. We deliberately do NOT
    call ``.resolve()``: keeping the Buck runfiles link-tree path means a missing
    resource fails loud under ``buck run`` instead of silently resolving against
    some other tree.
    """
    if (template is None) == (template_string is None):
        raise AgenticFunctionConfigurationError(
            "exactly one of template= or template_string= must be provided"
        )
    if template_string is not None:
        return template_string, None
    path = Path(inspect.getfile(inspect.unwrap(fn))).parent / template
    return _read_cached(path), path


def _read_cached(path: Path) -> str:
    cached = _text_cache.get(path)
    if cached is None:
        try:
            cached = path.read_text()
        except OSError as e:
            raise AgenticFunctionConfigurationError(
                f"template file not found or unreadable: {path} ({e})"
            ) from e
        _text_cache[path] = cached
    return cached


def render(raw_text: str, feed: dict[str, Any]) -> str:
    """Render with the same Jinja2 engine ``TemplateManager`` uses under the hood."""
    return format_template(raw_text, feed=feed)


def validate_template_variables(
    raw_text: str, allowed: set[str], *, source: Optional[Path] = None
) -> None:
    """Raise if the template declares a variable outside ``allowed`` plus helpers.

    ``format_template`` renders a missing variable as empty (its
    ``_FalsyChainableUndefined``), so a typo would silently produce an empty
    substitution. Validating the declared variables at first call closes that
    hole.
    """
    declared = set(extract_variables(raw_text))
    unknown = declared - allowed - _HELPER_GLOBALS
    if unknown:
        where = f" in {source}" if source is not None else ""
        raise AgenticFunctionConfigurationError(
            f"template{where} references undeclared variable(s) "
            f"{sorted(unknown)}; known parameters are {sorted(allowed)}"
        )
