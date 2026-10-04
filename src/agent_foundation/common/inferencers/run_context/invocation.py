"""Invocation frames: the home of per-call state (plan v8 §4.2 home 2, §5.1).

Every public entry of an inferencer opens one :class:`InvocationFrame` for the
lifetime of the call (retries, guardrail, finalize). The frame holds:

* typed **components**, keyed by :class:`RuntimeKey` (compared by identity), so a
  class keeps per-call values off its own instance;
* a :class:`ResourceLedger` of resources the call owns, closed in reverse
  registration order when the invocation ends;
* the ``cleanup_errors`` recorded by every ledger closed during the invocation.

Under a ctx, the node's typed outcome is cleared when the invocation opens and
published once, from the owner's ``_outcome_for(frame)``, when it succeeds.

Under a ctx, an invocation holds the path claim of ``(ctx.store, ctx.path)`` while
it is open (``ActivePathClaims`` in ``store.py``): one path hosts one invocation at
a time, and nested work runs at a child path.

Frames form a chain through the ``_current_invocation`` ContextVar, so they follow
the same copy semantics as ``_active_ctx`` (``asyncio`` tasks and
``copy_context`` threads see the frame of the call that created them).
:func:`frame_for` walks that chain by owner identity, which lets an owner reach its
own frame while a child's call is running. A frame is marked ``closed`` when its
invocation ends; closed frames are skipped by :func:`frame_for`, and their ledger
refuses new registrations, so a task that outlives the call can't write into it.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar

import attrs
from rich_python_utils.common_utils.async_utils import run_async_joined

from .bridge import _active_ctx, active_run_context, resolve_run
from .errors import (
    InvocationCleanupError,
    InvocationContractError,
    NoInvocationError,
    UncertifiedConcurrentUseError,
)

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

InvocationMode = Literal["host", "legacy", "no_ctx"]

_CLOSE_METHODS = ("adisconnect", "aclose", "close")


def _to_read_only_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(value))


@attrs.define(frozen=True, eq=False)
class RuntimeKey(Generic[T]):
    """Names one typed component of an invocation frame.

    Keys hash by identity: declare each one once, as a ClassVar beside its owner
    class. ``compat`` maps each documented bare-getter field the value backs to the
    attribute of the value that feeds it (``""`` is the value itself).
    """

    name: str
    factory: Callable[[], T] | None = None
    compat: Mapping[str, str] = attrs.field(
        factory=dict, converter=_to_read_only_mapping
    )


class ResourceLedger:
    """Resources owned by one invocation (or one BTA attempt).

    A resource is anything with ``adisconnect``, ``aclose`` or ``close``; the first
    one present is used. Closing attempts every resource, in reverse registration
    order, and seals the ledger: a later ``register`` raises.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[tuple[Any, str]] = []
        self._sealed = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def register(self, resource: Any, label: str) -> None:
        if not any(callable(getattr(resource, m, None)) for m in _CLOSE_METHODS):
            raise InvocationContractError(
                f"Cannot own {label!r}: {type(resource).__name__} has none of "
                f"{', '.join(_CLOSE_METHODS)}."
            )
        with self._lock:
            if self._sealed:
                raise InvocationContractError(
                    f"Cannot own {label!r}: the ledger is already closed (its "
                    f"invocation ended), so the resource would never be closed."
                )
            self._entries.append((resource, label))

    def _seal(self) -> list[tuple[Any, str]]:
        with self._lock:
            self._sealed = True
            entries, self._entries = self._entries, []
        entries.reverse()
        return entries

    async def aclose(self) -> list[str]:
        """Close every resource in the entry's loop; return the failure strings."""
        return await _aclose_entries(self._seal())

    def close_joined(self) -> list[str]:
        """Sync twin of :meth:`aclose`, run through ``run_async_joined``.

        Loop-bound resources must already be closed inside the loop that created
        them; what reaches a sync ledger is loop-independent teardown.
        """
        entries = self._seal()
        if not entries:
            return []
        return run_async_joined(_aclose_entries(entries))


async def _aclose_one(resource: Any) -> None:
    for name in _CLOSE_METHODS:
        method = getattr(resource, name, None)
        if callable(method):
            outcome = method()
            if inspect.isawaitable(outcome):
                await outcome
            return


async def _aclose_entries(entries: list[tuple[Any, str]]) -> list[str]:
    errors: list[str] = []
    for index, (resource, label) in enumerate(entries):
        try:
            await _aclose_one(resource)
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
        except BaseException as interrupt:
            # Cancellation and interpreter exits still let every other resource
            # close before the interrupt propagates.
            errors.extend(await _aclose_entries(entries[index + 1 :]))
            for error in errors:
                interrupt.add_note(f"owned resource failed to close: {error}")
            raise
    return errors


@attrs.define(eq=False, repr=False)
class InvocationFrame:
    """Per-call state of one public invocation of ``owner``."""

    owner: Any
    ctx: Any
    mode: InvocationMode
    parent: InvocationFrame | None
    entry: str
    opened_at: float = attrs.field(factory=time.monotonic)
    invocation_id: str = attrs.field(factory=lambda: uuid.uuid4().hex)
    components: dict[RuntimeKey[Any], Any] = attrs.field(factory=dict)
    pending_compat: dict[str, Any] = attrs.field(factory=dict)
    ledger: ResourceLedger = attrs.field(factory=ResourceLedger)
    cleanup_errors: list[str] = attrs.field(factory=list)
    closed: bool = False
    result: Any = None

    def __repr__(self) -> str:
        return (
            f"InvocationFrame(owner={type(self.owner).__name__}, "
            f"entry={self.entry!r}, mode={self.mode!r}, "
            f"invocation_id={self.invocation_id!r}, closed={self.closed})"
        )

    def get(self, key: RuntimeKey[T]) -> T | None:
        return self.components.get(key)

    def has(self, key: RuntimeKey[T]) -> bool:
        """Whether ``key`` was put in this invocation (also with ``None``)."""
        return key in self.components

    def require(self, key: RuntimeKey[T]) -> T:
        try:
            return self.components[key]
        except KeyError:
            raise KeyError(
                f"{key.name} is not set in the current invocation of "
                f"{type(self.owner).__name__}"
            ) from None

    def get_or_create(self, key: RuntimeKey[T]) -> T:
        if key in self.components:
            return self.components[key]
        if key.factory is None:
            raise InvocationContractError(
                f"{key.name} has no factory; put it before reading it with "
                f"get_or_create."
            )
        value = key.factory()
        self.components[key] = value
        return value

    def put(self, key: RuntimeKey[T], value: T) -> None:
        self.components[key] = value

    def discard(self, key: RuntimeKey[T]) -> None:
        self.components.pop(key, None)


_current_invocation: contextvars.ContextVar[InvocationFrame | None] = (
    contextvars.ContextVar("_af_current_invocation", default=None)
)


def _mode_of(ctx: Any) -> InvocationMode:
    if ctx is None:
        return "no_ctx"
    return "legacy" if ctx.legacy_mint else "host"


def _new_frame(owner: Any, entry: str, ctx: Any = None) -> InvocationFrame:
    if ctx is None:
        ctx = active_run_context()
    return InvocationFrame(
        owner=owner,
        ctx=ctx,
        mode=_mode_of(ctx),
        parent=_current_invocation.get(),
        entry=entry,
    )


def frame_for(owner: Any) -> InvocationFrame | None:
    """The innermost open frame of ``owner`` in the current chain, else ``None``."""
    frame = _current_invocation.get()
    while frame is not None:
        if frame.owner is owner and not frame.closed:
            return frame
        frame = frame.parent
    return None


def invocation_of(owner: Any) -> InvocationFrame:
    """:func:`frame_for`, raising :class:`NoInvocationError` when there is none."""
    frame = frame_for(owner)
    if frame is None:
        raise NoInvocationError(
            f"{type(owner).__name__} has no open invocation. Public entries open "
            f"an invocation; tests that call a private hook directly wrap it in "
            f"open_invocation(inst) / aopen_invocation(inst)."
        )
    return frame


def _compat_values(key: RuntimeKey[T], value: T | None) -> dict[str, Any]:
    return {
        field: value if attr == "" or value is None else getattr(value, attr)
        for field, attr in key.compat.items()
    }


def publish_result(owner: Any, key: RuntimeKey[T], value: T) -> None:
    """Publish a result-bearing component of ``owner``'s current invocation (§5.4).

    Inside the owner's invocation the value becomes the frame component; in a
    non-host invocation the fields ``key.compat`` declares are also queued and
    written onto the owner when the invocation closes, whatever its outcome.
    Without an invocation (a private hook called directly) it is a no-op under a
    host ctx, and otherwise writes the declared compat fields at once: the
    documented getter is all a bare caller can observe.
    """
    frame = frame_for(owner)
    if frame is not None:
        frame.put(key, value)
        if frame.mode != "host":
            frame.pending_compat.update(_compat_values(key, value))
        return
    ctx = active_run_context()
    if ctx is not None and not ctx.legacy_mint:
        return
    for field, field_value in _compat_values(key, value).items():
        setattr(owner, field, field_value)


def discard_result(owner: Any, key: RuntimeKey[T]) -> None:
    """Withdraw ``key`` from ``owner``'s current invocation: the component and the
    compat fields it queued, so the invocation's close never flushes them."""
    frame = invocation_of(owner)
    frame.discard(key)
    for field in key.compat:
        frame.pending_compat.pop(field, None)


def read_result(owner: Any, key: RuntimeKey[T]) -> T | None:
    """The value of ``key`` in ``owner``'s current invocation (``None`` if unset).

    Raises :class:`NoInvocationError` outside one: compat fields are a lossy
    projection, and an in-call read outside the owner's invocation is a broken
    precondition.
    """
    frame = frame_for(owner)
    if frame is None:
        raise NoInvocationError(
            f"{key.name} read outside an invocation of {type(owner).__name__}. "
            f"Public entries open an invocation; tests that call a private hook "
            f"directly wrap it in open_invocation(inst) / aopen_invocation(inst)."
        )
    return frame.get(key)


def declared_runtime_keys(cls: type) -> tuple[RuntimeKey[Any], ...]:
    """Every ``RuntimeKey`` declared as a class attribute along ``cls``'s MRO."""
    seen: dict[int, RuntimeKey[Any]] = {}
    for klass in cls.__mro__:
        for value in vars(klass).values():
            if isinstance(value, RuntimeKey):
                seen.setdefault(id(value), value)
    return tuple(seen.values())


def declared_compat_fields(cls: type) -> frozenset[str]:
    """The instance fields ``cls``'s declared keys feed as bare compat getters."""
    return frozenset(
        field for key in declared_runtime_keys(cls) for field in key.compat
    )


def _record_failure_cleanup(
    frame: InvocationFrame, exc: BaseException, errors: list[str]
) -> None:
    frame.cleanup_errors.extend(errors)
    if not frame.cleanup_errors:
        return
    logger.error(
        "%s.%s raised %s; its owned resources also failed to close: %s",
        type(frame.owner).__name__,
        frame.entry,
        type(exc).__name__,
        "; ".join(frame.cleanup_errors),
    )
    for error in frame.cleanup_errors:
        exc.add_note(f"owned resource failed to close: {error}")


def _raise_if_cleanup_failed(frame: InvocationFrame) -> None:
    if frame.cleanup_errors:
        raise InvocationCleanupError(frame.result, tuple(frame.cleanup_errors))


def _claim_path(frame: InvocationFrame) -> None:
    if frame.ctx is not None:
        frame.ctx.store.claims.acquire(frame.ctx.path, frame)


# The single-flight guard: the live host frame of every owner whose class is not
# host-pure certified, by ``id(owner)``. A frame holds its owner strongly, so the id
# can't be reused while registered.
_single_flight: dict[int, InvocationFrame] = {}
_single_flight_lock = threading.Lock()


def host_pure_certified(cls: type) -> bool:
    """Whether ``cls`` itself declares ``_HOST_PURE_CERTIFIED``: certification is
    never inherited by a subclass, which may add state."""
    return bool(vars(cls).get("_HOST_PURE_CERTIFIED", False))


def _hold_single_flight(frame: InvocationFrame) -> None:
    """In host mode, an uncertified owner runs one invocation at a time in the
    process, whatever its path, stage role or depth; an overlapping one raises."""
    if frame.mode != "host" or host_pure_certified(type(frame.owner)):
        return
    with _single_flight_lock:
        holder = _single_flight.setdefault(id(frame.owner), frame)
    if holder is frame:
        return
    raise UncertifiedConcurrentUseError(
        f"{type(frame.owner).__name__} is not host-pure certified, so one instance "
        f"runs one host invocation at a time: {frame.entry!r} at "
        f"{frame.ctx.path!r} overlaps {holder.entry!r} at "
        f"{holder.ctx.path if holder.ctx is not None else None!r}, running for "
        f"{time.monotonic() - holder.opened_at:.1f}s. Use a factory slot (one "
        f"instance per call) for stages that run concurrently."
    )


def _release_single_flight(frame: InvocationFrame) -> None:
    with _single_flight_lock:
        if _single_flight.get(id(frame.owner)) is frame:
            del _single_flight[id(frame.owner)]


def _acquire(frame: InvocationFrame) -> None:
    """Claim the frame's path, then hold the single-flight guard; a refused guard
    gives the claim back, so a rejected call holds nothing."""
    _claim_path(frame)
    try:
        _hold_single_flight(frame)
    except BaseException:
        if frame.ctx is not None:
            frame.ctx.store.claims.release(frame.ctx.path, frame)
        raise


def _clear_outcome(frame: InvocationFrame) -> None:
    if frame.ctx is not None:
        frame.ctx.store.clear_outcome(frame.ctx.path)


def _publish_outcome(frame: InvocationFrame) -> None:
    """Publish the owner's ``_outcome_for(frame)`` at its node, stamped with the
    invocation id and every cleanup failure recorded so far."""
    if frame.ctx is None:
        return
    hook = getattr(frame.owner, "_outcome_for", None)
    outcome = hook(frame) if callable(hook) else None
    if outcome is None:
        return
    frame.ctx.store.publish_outcome(
        frame.ctx.path,
        (type(frame.owner).__qualname__, frame.ctx.path),
        attrs.evolve(
            outcome,
            invocation_id=frame.invocation_id,
            cleanup_errors=tuple(frame.cleanup_errors),
        ),
    )


def _flush_compat(frame: InvocationFrame) -> None:
    """Write the declared compat fields this invocation published onto its owner;
    non-host invocations only, at every close (success, failure, cancellation)."""
    for field, value in frame.pending_compat.items():
        setattr(frame.owner, field, value)


def _release(frame: InvocationFrame) -> None:
    try:
        _flush_compat(frame)
    finally:
        frame.closed = True
        _release_single_flight(frame)
        if frame.ctx is not None:
            frame.ctx.store.claims.release(frame.ctx.path, frame)


def _end_invocation(
    frame: InvocationFrame, token: contextvars.Token[InvocationFrame | None]
) -> None:
    try:
        _release(frame)
    finally:
        _current_invocation.reset(token)


@contextlib.asynccontextmanager
async def aopen_invocation(
    owner: Any, entry: str = "ainfer"
) -> AsyncIterator[InvocationFrame]:
    """Open an invocation of ``owner`` for an async entry (§5.1 steps 2–11).

    Under a ctx the invocation claims ``(ctx.store, ctx.path)`` first; an overlapping
    claim raises :class:`ConcurrentInvocationError` before the frame is bound, so a
    rejected call runs nothing. In host mode the single-flight guard then refuses an
    overlapping invocation of an uncertified owner the same way
    (:class:`UncertifiedConcurrentUseError`). Once claimed, the node's outcome is cleared, so a
    failed invocation never exposes an earlier one's.

    The body sets ``frame.result`` on success. An exception from the body always
    propagates unchanged, with any cleanup failures attached as notes, and publishes
    nothing. After a success, the owner's ``_outcome_for(frame)`` (when not ``None``)
    is published at its node, stamped with the invocation id and cleanup failures;
    only then do cleanup failures raise :class:`InvocationCleanupError` carrying the
    result. The claim is released on every exit.
    """
    frame = _new_frame(owner, entry)
    _acquire(frame)
    token = _current_invocation.set(frame)
    try:
        _clear_outcome(frame)
        try:
            yield frame
        except BaseException as exc:
            _record_failure_cleanup(frame, exc, await frame.ledger.aclose())
            raise
        frame.cleanup_errors.extend(await frame.ledger.aclose())
        _publish_outcome(frame)
        _raise_if_cleanup_failed(frame)
    finally:
        _end_invocation(frame, token)


@contextlib.contextmanager
def open_invocation(owner: Any, entry: str = "infer") -> Iterator[InvocationFrame]:
    """Sync twin of :func:`aopen_invocation`.

    The ledger closes through ``run_async_joined``, so a sync entry called from
    inside a running loop still closes it.
    """
    frame = _new_frame(owner, entry)
    _acquire(frame)
    token = _current_invocation.set(frame)
    try:
        _clear_outcome(frame)
        try:
            yield frame
        except BaseException as exc:
            _record_failure_cleanup(frame, exc, frame.ledger.close_joined())
            raise
        frame.cleanup_errors.extend(frame.ledger.close_joined())
        _publish_outcome(frame)
        _raise_if_cleanup_failed(frame)
    finally:
        _end_invocation(frame, token)


# -- streaming entries --------------------------------------------------------


@contextlib.contextmanager
def _bound(frame: InvocationFrame) -> Iterator[None]:
    """Bind the frame's ctx and the frame itself for one resumption of a stream."""
    ctx_token = _active_ctx.set(frame.ctx)
    frame_token = _current_invocation.set(frame)
    try:
        yield
    finally:
        _current_invocation.reset(frame_token)
        _active_ctx.reset(ctx_token)


def _cleanup_note(label: str, exc: BaseException) -> str:
    return f"{label}: {type(exc).__name__}: {exc}"


async def _aclose_stream(inner: AsyncIterator[Any] | None) -> list[str]:
    aclose = getattr(inner, "aclose", None)
    if aclose is None:
        return []
    try:
        await aclose()
    except Exception as exc:
        return [_cleanup_note("stream", exc)]
    return []


def _close_stream(inner: Iterator[Any] | None) -> list[str]:
    close = getattr(inner, "close", None)
    if close is None:
        return []
    try:
        close()
    except Exception as exc:
        return [_cleanup_note("stream", exc)]
    return []


async def framed_agen(
    owner: Any,
    entry: str,
    run_context: Any,
    start: Callable[[], AsyncIterator[T]],
    *,
    default_workspace: Any = None,
) -> AsyncIterator[T]:
    """The invocation of an async streaming entry (§5.1, streaming binding rule).

    Nothing runs until the first resumption. Then the ctx is resolved once by the
    bridge's mint policy (explicit, then active, then a legacy root from
    ``default_workspace``), the path is claimed (a rejected stream binds nothing),
    the node's outcome is cleared and ``start()`` builds the inner stream. The ctx
    and the frame are bound only while the inner stream runs, around each
    resumption, and reset before every yield, so the consumer only ever sees its
    own context, and every token is reset in the Context that set it.

    The stream succeeds only when the inner stream is exhausted: the ledger closes
    and the owner's ``_outcome_for(frame)`` is published, as in
    :func:`aopen_invocation`. Closing it early (``aclose``, cancellation, garbage
    collection) closes the inner stream and the ledger under the same binding and
    publishes nothing. The claim is released on every exit; an abandoned stream
    keeps it until it is closed or finalized.
    """
    frame = _new_frame(
        owner, entry, resolve_run(run_context, default_workspace=default_workspace)
    )
    _acquire(frame)
    inner: AsyncIterator[T] | None = None
    try:
        try:
            with _bound(frame):
                _clear_outcome(frame)
                inner = start()
            while True:
                with _bound(frame):
                    try:
                        item = await inner.__anext__()
                    except StopAsyncIteration:
                        break
                yield item
        except BaseException as exc:
            with _bound(frame):
                errors = await _aclose_stream(inner)
                errors.extend(await frame.ledger.aclose())
            _record_failure_cleanup(frame, exc, errors)
            raise
        with _bound(frame):
            frame.cleanup_errors.extend(await frame.ledger.aclose())
            _publish_outcome(frame)
        _raise_if_cleanup_failed(frame)
    finally:
        _release(frame)


def framed_gen(
    owner: Any,
    entry: str,
    run_context: Any,
    start: Callable[[], Iterator[T]],
    *,
    default_workspace: Any = None,
) -> Iterator[T]:
    """Sync twin of :func:`framed_agen`; the ledger closes through
    ``run_async_joined``."""
    frame = _new_frame(
        owner, entry, resolve_run(run_context, default_workspace=default_workspace)
    )
    _acquire(frame)
    inner: Iterator[T] | None = None
    try:
        try:
            with _bound(frame):
                _clear_outcome(frame)
                inner = start()
            while True:
                with _bound(frame):
                    try:
                        item = next(inner)
                    except StopIteration:
                        break
                yield item
        except BaseException as exc:
            with _bound(frame):
                errors = _close_stream(inner)
                errors.extend(frame.ledger.close_joined())
            _record_failure_cleanup(frame, exc, errors)
            raise
        with _bound(frame):
            frame.cleanup_errors.extend(frame.ledger.close_joined())
            _publish_outcome(frame)
        _raise_if_cleanup_failed(frame)
    finally:
        _release(frame)


def ctx_bound_gen(ctx: Any, inner: Iterator[T]) -> Iterator[T]:
    """A lazy iterator whose every step runs under ``ctx``: ``_active_ctx`` is bound
    around each ``next()`` of ``inner`` (and its ``close()``) and reset before
    yielding, so the consumer keeps its own context between items and an abandoned
    iterator leaves nothing behind. It opens no frame: each item's own entry does."""
    try:
        while True:
            token = _active_ctx.set(ctx)
            try:
                item = next(inner)
            except StopIteration:
                return
            finally:
                _active_ctx.reset(token)
            yield item
    finally:
        token = _active_ctx.set(ctx)
        try:
            close = getattr(inner, "close", None)
            if close is not None:
                close()
        finally:
            _active_ctx.reset(token)
