# pyre-strict

"""Exceptions for the agentic-function framework."""

from __future__ import annotations


class AgenticFunctionError(Exception):
    """Base class for all agentic-function errors."""


class ParseError(AgenticFunctionError):
    """A stage-1 / stage-2 parse (or strict return-annotation coercion) failed.

    Raised by the parse pipeline and by :meth:`AgenticOutput.json` when a labeled
    block is absent or a value cannot be coerced to the declared return type. It
    is the default member of a decorator's ``retry_on`` set, so a parse failure
    drives the explicit parse-retry loop rather than propagating immediately.
    """


class AgenticFunctionConfigurationError(AgenticFunctionError):
    """A decorated function is misconfigured (raised at decoration or first call).

    Covers an unknown ``inferencer_kwargs`` key, an ambiguous ``escalate_on_none``
    on an ``Optional`` return, an undeclared template variable, an unresolved
    OmegaConf ``MISSING`` in a topology config, an untrusted config path, and a
    non-empty ``preflight_all()`` report.
    """
