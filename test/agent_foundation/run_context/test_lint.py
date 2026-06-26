"""M0 §9.5 child-call completeness lint — self-tested against synthetic fixtures."""

from agent_foundation.common.inferencers.run_context.lint import (
    CHILD_CALL_METHODS,
    find_child_call_sites,
    lint_source,
)

# A synthetic "inferencer module": a threaded call, an un-threaded call, and an
# exempt one — exactly the three cases the lint must distinguish.
FIXTURE = '''
class Orchestrator:
    async def _ainfer(self, x):
        # threaded: passes run_context -> OK
        a = await self.child_a.ainfer(x, run_context=ctx.child("a"))
        # un-threaded: no run_context -> VIOLATION
        b = await self.child_b.ainfer(x)
        # sync child call, un-threaded -> VIOLATION
        c = self.child_c.infer(x)
        return a, b, c

    def helper(self):
        # exempt-example (documented): a nested-self child call we choose to exempt
        return self.iter_infer(x)
'''


def test_find_child_call_sites_detects_all_methods():
    sites = find_child_call_sites(FIXTURE)
    methods = sorted({s.method for s in sites})
    assert "ainfer" in methods and "infer" in methods and "iter_infer" in methods
    # enclosing scope is tracked (class.method)
    ainfer_sites = [s for s in sites if s.method == "ainfer"]
    assert all("Orchestrator._ainfer" == s.enclosing for s in ainfer_sites)


def test_threaded_call_is_not_a_violation():
    sites = find_child_call_sites(FIXTURE)
    threaded = [s for s in sites if s.method == "ainfer" and s.has_run_context]
    assert len(threaded) == 1


def test_lint_flags_unthreaded_and_respects_exempt_list():
    # Exempt the iter_infer line; the two un-threaded ainfer/infer remain violations.
    sites = find_child_call_sites(FIXTURE)
    exempt_line = next(s.lineno for s in sites if s.method == "iter_infer")
    violations = lint_source(FIXTURE, exempt=[("iter_infer", exempt_line)])
    flagged = sorted({v.method for v in violations})
    assert flagged == ["ainfer", "infer"]
    assert all(v.method != "iter_infer" for v in violations)


def test_fully_threaded_or_exempt_source_has_no_violations():
    clean = '''
class C:
    async def _ainfer(self, x):
        return await self.k.ainfer(x, run_context=ctx.child("k"))
'''
    assert lint_source(clean) == []


def test_child_call_method_set_covers_the_documented_surface():
    for m in ("infer", "ainfer", "ainfer_streaming", "iter_infer", "run_agentic_loop"):
        assert m in CHILD_CALL_METHODS


def test_inference_api_is_not_a_child_call_method():
    """Option C: the provider-callable splat ``_inference_api`` is the LLM-backend
    boundary, NOT a child-inference call site. ``run_context`` stops there, so there is
    nothing to thread; its kwarg-leak defense is the ``get_relevant_named_args`` filter
    (``test_m2_kwarg_leak_filter.py``). It is therefore deliberately absent from
    CHILD_CALL_METHODS and needs no exemption in the completeness gate."""
    assert "_inference_api" not in CHILD_CALL_METHODS
