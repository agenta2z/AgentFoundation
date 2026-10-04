"""The invocation contract errors (plan v8 §5.1).

A leaf module, so the store can raise them for path claims without importing the
invocation module.
"""

from __future__ import annotations

from typing import Any


class InvocationContractError(RuntimeError):
    """A violated invocation contract. Never retried."""


class ConcurrentInvocationError(InvocationContractError):
    """Two overlapping public invocations at one ``(store, path)``."""


class UncertifiedConcurrentUseError(InvocationContractError):
    """Overlapping host invocations of one instance whose class is not host-pure."""


class StageOwnershipError(InvocationContractError):
    """A stage factory returned an object this call already owns."""


class InvocationCleanupError(InvocationContractError):
    """The call succeeded but closing its owned resources failed.

    ``result`` is the completed result, so it is never lost; ``errors`` holds one
    ``"<resource>: <exception type>: <message>"`` string per failed close.
    """

    def __init__(self, result: Any, errors: tuple[str, ...]) -> None:
        self.result = result
        self.errors = tuple(errors)
        super().__init__(result, self.errors)

    def __str__(self) -> str:
        return (
            f"{len(self.errors)} owned resource(s) failed to close after a "
            f"successful call: {'; '.join(self.errors)}"
        )


class NoInvocationError(RuntimeError):
    """An in-call read ran outside its owner's invocation."""
