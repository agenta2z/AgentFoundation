"""M3 / §9.5 completeness gate: every child-inference call site is threaded OR exempt.

This is the executable invariant that replaces hand-enumeration (the "whack-a-mole"
diagnosis): it scans every inferencer module and fails if any child-inference call
site neither threads ``run_context=`` nor is on the documented exempt-list. A newly
added (or newly un-threaded) site becomes a red test, not a future audit finding.
"""

import glob
import os

import pytest

from agent_foundation.common.inferencers.run_context.lint import find_child_call_sites

_INFERENCERS_DIR = os.path.join(
    os.path.dirname(
        __import__("agent_foundation.common.inferencers.inferencer_base", fromlist=["x"]).__file__
    ),
)

# Documented exempt sites: (file_basename, enclosing_suffix, method) -> reason.
# Each is exempt BY DESIGN (does not thread run_context):
#  * nested self-delegation -> the §2.2 mint policy reuses the active ctx;
#  * host-mint boundaries -> the app mints the root there (not a child thread);
#  * a generic helper whose role-slot isn't determinable at the call site.
# (The ``_inference_api`` provider splat is NOT in CHILD_CALL_METHODS (option C), so it is
#  never flagged here and needs no exemption; its kwarg-leak defense is the
#  ``get_relevant_named_args`` filter, covered by ``test_m2_kwarg_leak_filter.py``.)
EXEMPT = {
    # NOTE: conversational_inferencer.py::run::run_agentic_loop is now THREADED
    # (the SOP host passes a per-turn child ctx) — removed from the exempt-list.
    ("flow_node_adapter.py", "_ainfer", "run_agentic_loop"): "host boundary — run_agentic_loop run_context param is a separate host step",
    ("rovodev_anthropic_proxy.py", "messages", "infer"): "server route host-mint boundary",
    # super() delegation: run_context rides **kwargs to the base ainfer_streaming (which
    # installs the bridge); but the pre-super() session block reads/writes active_session_id
    # BEFORE that install, so under a ctx it targets the active (parent) branch, not this
    # leaf's. Real fix = the N-Major2 "install the bridge FIRST" leaf conversion (+ tests).
    ("rovodev_cli_inferencer.py", "ainfer_streaming", "ainfer_streaming"): "super() delegation forwards run_context via **kwargs; session block reads active_session_id before the base installs the bridge — N-Major2 'install bridge first' leaf conversion pending",
    ("tool_as_inferencer.py", "_ainfer", "ainfer_streaming"): "nested self (reuses active ctx)",
    # NOTE: the EXTERNAL fallback wrappers in __infer_single_impl / __ainfer_single_impl
    # are now THREADED (run_context=ctx.child("fallback").child(f"external_{i}"), §9.3
    # E3/I3) — removed from the exempt-list (they pass by fix, not exemption). The
    # separate _recovery_wrapper re-runs self and correctly reuses the active ctx.
    ("inferencer_base.py", "iter_infer", "infer"): "nested self (reuses active ctx)",
    ("inferencer_base.py", "__call__", "infer"): "nested self (reuses active ctx)",
    ("inferencer_base.py", "aiter_infer", "ainfer"): "nested self (reuses active ctx)",
    ("streaming_inferencer_base.py", "_ainfer", "ainfer_streaming"): "nested self (reuses active ctx)",
    ("streaming_inferencer_base.py", "_run_async_streaming", "ainfer_streaming"): "nested self via copy_context (reuses active ctx)",
    # Session helpers install the bridge (enter_run) FIRST (N-Major2) so the session
    # reset/read targets the branch; the inner self-call then REUSES that active ctx.
    ("streaming_inferencer_base.py", "new_session", "infer"): "nested self — helper installs the bridge first (N-Major2)",
    ("streaming_inferencer_base.py", "anew_session", "ainfer"): "nested self — helper installs the bridge first (N-Major2)",
    ("streaming_inferencer_base.py", "resume_session", "infer"): "nested self — helper installs the bridge first (N-Major2)",
    ("streaming_inferencer_base.py", "aresume_session", "ainfer"): "nested self — helper installs the bridge first (N-Major2)",
}


def _all_unthreaded_sites():
    out = []
    for f in glob.glob(os.path.join(_INFERENCERS_DIR, "**", "*.py"), recursive=True):
        if "run_context" + os.sep in f:
            continue
        try:
            sites = find_child_call_sites(open(f).read())
        except SyntaxError:
            continue
        base = os.path.basename(f)
        for s in sites:
            if not s.has_run_context:
                enclosing_suffix = s.enclosing.split(".")[-1]
                out.append((base, enclosing_suffix, s.method, s.lineno))
    return out


def test_every_child_call_site_is_threaded_or_exempt():
    """The §9.5 completeness invariant — no UNDOCUMENTED un-threaded child-call site."""
    undocumented = [
        (base, enc, method, lineno)
        for (base, enc, method, lineno) in _all_unthreaded_sites()
        if (base, enc, method) not in EXEMPT
    ]
    assert not undocumented, (
        "Un-threaded child-inference call site(s) not on the exempt-list — thread "
        "run_context= or add a documented exemption:\n"
        + "\n".join(f"  {b}:{ln} {m} in {e}" for (b, e, m, ln) in undocumented)
    )


def test_exempt_entries_are_all_still_present():
    """Guard against a stale exempt-list: every EXEMPT key must match a real site."""
    live = {(b, e, m) for (b, e, m, _ln) in _all_unthreaded_sites()}
    stale = [k for k in EXEMPT if k not in live]
    assert not stale, f"Exempt-list has stale entries (site threaded or removed): {stale}"


def test_with_child_ctx_wrapped_lifecycle_call_is_recognized_as_threaded():
    """The §9.5 lint must NOT false-positive on the slot-binding pattern: a
    lifecycle call inside `with self._with_child_ctx(slot):` binds the child's ctx
    without a run_context= kwarg, so it is threaded-equivalent."""
    from agent_foundation.common.inferencers.run_context.lint import (
        LIFECYCLE_CALL_METHODS,
        find_child_call_sites,
        lint_source,
    )

    src = (
        "class O:\n"
        "    async def _areset(self):\n"
        "        for slot, inf in self._iter_child_slots():\n"
        "            with self._with_child_ctx(slot):\n"
        "                inf.reset_session()\n"
        "        inf.reset_session()  # NOT wrapped -> flagged\n"
    )
    sites = find_child_call_sites(src, methods=LIFECYCLE_CALL_METHODS)
    bound = [s for s in sites if s.ctx_bound]
    unbound = [s for s in sites if not s.ctx_bound]
    assert len(bound) == 1 and bound[0].method == "reset_session"
    assert len(unbound) == 1  # the bare trailing call
    # lint_source flags ONLY the unwrapped one
    viol = lint_source(src, methods=LIFECYCLE_CALL_METHODS)
    assert len(viol) == 1 and viol[0].lineno == unbound[0].lineno


def test_aconnect_adisconnect_excluded_from_lifecycle_lint():
    """aconnect/adisconnect run at no-context lifecycle boundaries (their correctness
    is the leaf drain-all teardown, not call-site threading), so they are deliberately
    NOT in the lifecycle-call set — flagging the correct bare child.adisconnect() calls
    would be a false-positive."""
    from agent_foundation.common.inferencers.run_context.lint import (
        LIFECYCLE_CALL_METHODS,
    )

    assert "aconnect" not in LIFECYCLE_CALL_METHODS
    assert "adisconnect" not in LIFECYCLE_CALL_METHODS
    assert LIFECYCLE_CALL_METHODS == frozenset({"reset_session", "switch_role"})
