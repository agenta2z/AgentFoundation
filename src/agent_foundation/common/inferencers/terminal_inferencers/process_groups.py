"""The CLI process groups this process started, and the last-resort reaper that
ends them when the interpreter exits.

The terminal CLI leaves (through the spawn helpers of
``TerminalInferencerBase`` / ``TerminalSessionInferencerBase``) and the native
per-turn ``CliProcess`` start each CLI in its own session, so that a
cancellation ends the CLI's whole tree with one ``killpg``. Such a group no
longer receives the signals of the host's terminal or process group, and
nothing ends it when the host exits without running its own cleanup (e.g. a
server whose forced exit skips the shutdown that cancels its runs): the CLI and
everything it spawned keep running, reparented to init.

So a leaf registers each group once spawned and unregisters it once it ended
the group. Whatever is still registered when the interpreter exits is ended by
``reap_all`` (SIGTERM, up to ``TERM_GRACE_S`` for the group to empty, then
SIGKILL): from ``atexit``, and, once ``install_exit_signal_reaper`` ran, from
the handler of a terminating signal (``atexit`` does not run when a signal's
default action ends the process).

``PR_SET_PDEATHSIG`` is not used: it reaches only the direct child (the ``sh``
wrapper or the CLI launcher) while the agent runs further down the tree, it
fires when the spawning *thread* exits rather than the process, and setting it
takes a ``preexec_fn``, which is not safe in a threaded host.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
import time
from types import FrameType
from typing import Iterable, Optional

# How long the groups get to exit on SIGTERM before SIGKILL.
TERM_GRACE_S = 5.0
_POLL_S = 0.05

# Reentrant: a signal handler running ``reap_all`` interrupts the main thread,
# possibly inside ``register``.
_lock = threading.RLock()
# pgid -> pid of the process that registered it: a forked child inherits the
# registry, and must not end its parent's groups.
_groups: dict[int, int] = {}
_atexit_registered = False


def register(pgid: int) -> None:
    """Record a CLI process group this process started (``start_new_session``:
    the pgid is the CLI's pid)."""
    global _atexit_registered
    with _lock:
        _groups[pgid] = os.getpid()
        if not _atexit_registered:
            atexit.register(reap_all)
            _atexit_registered = True


def unregister(pgid: int) -> None:
    """Forget a group once its leaf ended it: its pgid may then name another
    group."""
    with _lock:
        _groups.pop(pgid, None)


def registered() -> frozenset[int]:
    """The groups this process started and has not ended yet."""
    pid = os.getpid()
    with _lock:
        return frozenset(g for g, owner in _groups.items() if owner == pid)


def reap_all(grace_s: float = TERM_GRACE_S) -> None:
    """End every registered group: SIGTERM, then SIGKILL to the groups not
    empty after ``grace_s``. Synchronous: it runs where no event loop does."""
    groups = registered()
    alive = {g for g in groups if _signal(g, signal.SIGTERM)}
    deadline = time.monotonic() + grace_s
    while alive and time.monotonic() < deadline:
        time.sleep(_POLL_S)
        alive = {g for g in alive if _occupied(g)}
    for pgid in alive:
        _signal(pgid, signal.SIGKILL)
    with _lock:
        for pgid in groups:
            _groups.pop(pgid, None)


def install_exit_signal_reaper(signals: Optional[Iterable[int]] = None) -> None:
    """Run ``reap_all`` before SIGTERM or SIGHUP (or ``signals``) ends the
    process.

    Only a signal left to its default action gets the handler (a handled one
    belongs to its handler), and the action is kept: the handler reaps,
    restores the default and re-raises the signal. A handler installed later
    that restores the one it found when it is done (as uvicorn does) restores
    this one. Main thread only (``signal.signal``).
    """
    if threading.current_thread() is not threading.main_thread():
        return
    if signals is None:
        signals = [
            sig
            for sig in (
                getattr(signal, "SIGTERM", None),
                getattr(signal, "SIGHUP", None),
            )
            if sig is not None
        ]
    for sig in signals:
        if signal.getsignal(sig) is signal.SIG_DFL:
            signal.signal(sig, _reap_then_die)


def _reap_then_die(sig: int, frame: Optional[FrameType]) -> None:
    reap_all()
    signal.signal(sig, signal.SIG_DFL)
    signal.raise_signal(sig)


def _occupied(pgid: int) -> bool:
    """Whether a process is left in the group. The leader is this process's
    child: once it exited it stays a member until reaped, so reap it here."""
    try:
        os.waitpid(pgid, os.WNOHANG)
    except ChildProcessError:
        pass
    return _signal(pgid, 0)


def _signal(pgid: int, sig: int) -> bool:
    """Signal a process group; False once no process is left in it."""
    if not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:  # a member we may not signal (setuid)
        pass
    return True
