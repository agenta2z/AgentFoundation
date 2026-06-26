"""M0 — the child-call completeness lint (§9.5 / J5, made concrete).

The recurring lesson of the design rounds: a hand-enumerated ``file:line`` list
is **necessary but insufficient** — every round surfaced a new threading site.
Completeness is enforced **programmatically**: an **AST** pass finds *every*
child-inference call site and **fails** unless each is either threaded
(``run_context=`` passed, or on a documented exempt-list).

This is the mechanism (importable + self-tested).  Wiring it as a CI gate over
the real tree — seeded with the §9.3 list as the initial exempt-set and
tightened per M-step — is the M0 deliverable; until M2+ thread the sites, the
real-tree run is a *report*, not a hard gate.
"""

from __future__ import annotations

import ast
from typing import Iterable, NamedTuple

# Attribute-call method names that denote a child inference (the §9.3 surface).
# NOTE (option C): the provider-callable splat ``_inference_api`` is deliberately NOT in
# this set. It is the LLM-backend boundary, not a child inferencer — ``run_context`` must
# stop there, so there is nothing to "thread". Its kwarg-leak defense (run_context must
# never reach the provider) is the ``get_relevant_named_args`` filter in
# ``ApiInferencerBase._infer``, covered by ``test_m2_kwarg_leak_filter.py`` — not this
# child-call lint. Keeping it out means the child-call lint contains only genuine
# child-inference methods and needs no provider-splat exemption.
CHILD_CALL_METHODS = frozenset(
    {
        "infer",
        "ainfer",
        "infer_streaming",
        "ainfer_streaming",
        "iter_infer",
        "run_agentic_loop",
    }
)

# Lifecycle calls that RELOCATE/RESET per-branch Tier-3 state from INSIDE _ainfer
# (under the orchestrator's active ctx) — these must run under each child's
# `ctx.child(slot)`, so the lifecycle-call lint (N-S3/S4) checks them.
#   - reset_session / switch_role: run during _ainfer (active ctx present) → the
#     slot binding via `_with_child_ctx(slot)` targets the child's OWN handle.
#   - `aconnect`/`adisconnect` are DELIBERATELY EXCLUDED: they run at lifecycle
#     boundaries (`__aenter__`/`__aexit__`/host cleanup) where NO context is active,
#     so a call-site `run_context=`/slot-binding would no-op. Their correctness is a
#     LEAF concern (drain every connection-scoped branch at teardown, see
#     StreamingInferencerBase._iter_live_handle_sets), NOT a call-site one — flagging
#     the (correct) bare `child.adisconnect()` calls would only add false-positives.
LIFECYCLE_CALL_METHODS = frozenset({"reset_session", "switch_role"})


class CallSite(NamedTuple):
    lineno: int
    col: int
    method: str
    has_run_context: bool
    enclosing: str  # qualified-ish name of the enclosing function/method
    ctx_bound: bool = False  # call is inside a `with self._with_child_ctx(slot):` body


class Violation(NamedTuple):
    lineno: int
    method: str
    enclosing: str
    reason: str


def _call_method_name(node: ast.Call) -> str | None:
    """Return the attribute method name for ``x.method(...)`` / bare ``method(...)``."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _has_run_context_kwarg(node: ast.Call) -> bool:
    if any(kw.arg == "run_context" for kw in node.keywords if kw.arg):
        return True
    # ``**something`` forwarding is treated as NOT explicitly threaded (conservative).
    return False


def _is_with_child_ctx_item(item: ast.withitem) -> bool:
    """True if a ``with`` item is ``…._with_child_ctx(slot)`` — the helper that binds
    ``_active_ctx = active.child(slot)`` for its body (inferencer_base.py). A lifecycle
    call inside such a block targets the child's OWN Tier-3 handle WITHOUT passing a
    ``run_context=`` kwarg, so it is threaded-equivalent and must not be flagged."""
    ctx = item.context_expr
    return (
        isinstance(ctx, ast.Call)
        and isinstance(ctx.func, ast.Attribute)
        and ctx.func.attr == "_with_child_ctx"
    )


def find_child_call_sites(source: str, methods: Iterable[str] = CHILD_CALL_METHODS) -> list[CallSite]:
    """Parse ``source`` and return every child-inference call site it contains."""
    method_set = frozenset(methods)
    tree = ast.parse(source)
    sites: list[CallSite] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._stack: list[str] = []
            self._ctx_depth = 0  # nesting depth inside `with self._with_child_ctx(...)`

        def _visit_scope(self, node: ast.AST, name: str) -> None:
            self._stack.append(name)
            self.generic_visit(node)
            self._stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_scope(node, node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_scope(node, node.name)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._visit_scope(node, node.name)

        def _visit_with(self, node) -> None:
            bound = any(_is_with_child_ctx_item(it) for it in node.items)
            if bound:
                self._ctx_depth += 1
            self.generic_visit(node)
            if bound:
                self._ctx_depth -= 1

        def visit_With(self, node: ast.With) -> None:
            self._visit_with(node)

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            self._visit_with(node)

        def visit_Call(self, node: ast.Call) -> None:
            method = _call_method_name(node)
            if method in method_set:
                sites.append(
                    CallSite(
                        lineno=node.lineno,
                        col=node.col_offset,
                        method=method,
                        has_run_context=_has_run_context_kwarg(node),
                        enclosing=".".join(self._stack) or "<module>",
                        ctx_bound=self._ctx_depth > 0,
                    )
                )
            self.generic_visit(node)

    _Visitor().visit(tree)
    return sites


def lint_source(
    source: str,
    *,
    exempt: Iterable[tuple[str, int]] = (),
    methods: Iterable[str] = CHILD_CALL_METHODS,
) -> list[Violation]:
    """Flag every child-call site that is neither threaded nor exempt.

    ``exempt`` is a set of ``(method, lineno)`` pairs (the documented exempt-list).
    A site is OK iff it passes ``run_context=`` or is exempt.
    """
    exempt_set = {(m, ln) for (m, ln) in exempt}
    violations: list[Violation] = []
    for site in find_child_call_sites(source, methods):
        if site.has_run_context or site.ctx_bound:
            # threaded explicitly (run_context= kwarg) or via `_with_child_ctx(slot)`
            continue
        if (site.method, site.lineno) in exempt_set:
            continue
        violations.append(
            Violation(
                lineno=site.lineno,
                method=site.method,
                enclosing=site.enclosing,
                reason=(
                    f"child-inference call {site.method!r} in {site.enclosing} does not "
                    f"thread run_context= and is not on the exempt-list"
                ),
            )
        )
    return violations
