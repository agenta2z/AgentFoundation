"""Regression tests for ``ConversationalInferencer`` SOP discovery filters.

Locks in the principled behavior of the ``allowed_sops`` (whitelist) and
``disallowed_sops`` (denylist) knobs added in this commit. Both are
purely cosmetic — the SOPs remain loadable via ``/sop <name>`` regardless
— but they control which SOPs appear in the rendered ``available_sops``
prompt section so the LLM doesn't surface irrelevant ones.

Precedence (same as iptables / AWS IAM / k8s NetworkPolicy):
  1. ``allowed_sops`` (whitelist) — if non-empty, only those names pass.
  2. ``disallowed_sops`` (denylist) — filters the survivors of step 1.

Naming convention mirrors the existing AF ``claude_code_*.allowed_tools``
attribs (single verb root with prefix variation for the antonym).
"""

from __future__ import annotations

import unittest

from attr import attrs

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase


@attrs(slots=False)
class _MockBase(InferencerBase):
    def _infer(self, inp, cfg=None, **kw):  # type: ignore[override]
        return "ok"

    async def _ainfer(self, inp, cfg=None, **kw):  # type: ignore[override]
        return "ok"


class TestSopDiscoveryFilters(unittest.TestCase):
    """Lock in the ``allowed_sops`` + ``disallowed_sops`` filter behavior."""

    # ─── Default state ─────────────────────────────────────────────

    def test_defaults_are_empty_lists(self) -> None:
        """Both filters default to empty list = backward-compatible."""
        ci = ConversationalInferencer(base_inferencer=_MockBase())
        self.assertEqual(ci.allowed_sops, [])
        self.assertEqual(ci.disallowed_sops, [])

    def test_explicit_filters_are_preserved(self) -> None:
        ci = ConversationalInferencer(
            base_inferencer=_MockBase(),
            allowed_sops=["sop_creation"],
            disallowed_sops=["model_optimization", "code_optimization"],
        )
        self.assertEqual(ci.allowed_sops, ["sop_creation"])
        self.assertEqual(
            ci.disallowed_sops,
            ["model_optimization", "code_optimization"],
        )

    # ─── Whitelist (allowed_sops) ─────────────────────────────────

    def test_allowed_sops_keeps_only_whitelisted(self) -> None:
        """When allowed_sops is non-empty, ONLY those SOPs appear."""
        ci_baseline = ConversationalInferencer(base_inferencer=_MockBase())
        rendered_baseline = ci_baseline._render_prompt("hello")
        # Baseline includes multiple framework SOPs — vacuity guard.
        self.assertIn("(`code_optimization`)", rendered_baseline)
        self.assertIn("(`model_optimization`)", rendered_baseline)

        ci_allowed = ConversationalInferencer(
            base_inferencer=_MockBase(),
            allowed_sops=["model_optimization"],
        )
        rendered = ci_allowed._render_prompt("hello")
        # Only the whitelisted one passes.
        self.assertIn("(`model_optimization`)", rendered)
        self.assertNotIn("(`code_optimization`)", rendered)
        self.assertNotIn("(`sop_creation`)", rendered)

    # ─── Denylist (disallowed_sops) ───────────────────────────────

    def test_disallowed_sops_hides_only_blacklisted(self) -> None:
        """disallowed_sops names are hidden; others remain visible."""
        ci_denied = ConversationalInferencer(
            base_inferencer=_MockBase(),
            disallowed_sops=["code_optimization"],
        )
        rendered = ci_denied._render_prompt("hello")
        # Blacklisted name MUST be gone from the Available SOPs section.
        self.assertNotIn(
            "(`code_optimization`)", rendered,
            "Denylist must filter out the named SOP.",
        )
        # Non-blacklisted MUST still appear (targeted, not nuclear).
        self.assertIn("(`model_optimization`)", rendered)

    # ─── Precedence (allow-then-deny) ─────────────────────────────

    def test_precedence_allow_then_deny(self) -> None:
        """allowed_sops is applied FIRST; disallowed_sops filters survivors.

        Matches iptables / AWS IAM / k8s NetworkPolicy precedence:
        the whitelist defines the pool; the denylist subtracts from it.
        """
        ci = ConversationalInferencer(
            base_inferencer=_MockBase(),
            allowed_sops=["model_optimization", "code_optimization"],
            disallowed_sops=["code_optimization"],
        )
        rendered = ci._render_prompt("hello")
        # Survives both filters.
        self.assertIn("(`model_optimization`)", rendered)
        # Filtered by denylist after passing whitelist.
        self.assertNotIn("(`code_optimization`)", rendered)
        # Not in whitelist → filtered.
        self.assertNotIn("(`sop_creation`)", rendered)

    def test_empty_filters_are_no_ops(self) -> None:
        """Empty allowed_sops + empty disallowed_sops = baseline behavior."""
        ci_baseline = ConversationalInferencer(base_inferencer=_MockBase())
        ci_explicit = ConversationalInferencer(
            base_inferencer=_MockBase(),
            allowed_sops=[],
            disallowed_sops=[],
        )
        self.assertEqual(
            ci_baseline._render_prompt("hello"),
            ci_explicit._render_prompt("hello"),
            "Empty filter lists must be byte-identical to no filters at all.",
        )

    # ─── Semantic lock-in: empty allow ≠ "no SOPs allowed" ────────
    #
    # This is the SINGLE most consequential semantic of the allow/deny
    # design and the easiest one to invert by accident in a future
    # refactor (e.g. someone changing ``if sops and self.allowed_sops:``
    # to ``if sops:`` would silently flip empty-list from "unconstrained"
    # to "deny all"). These tests make such a regression impossible to
    # land silently — they will fail loudly with an explicit message.

    def test_empty_allowed_sops_means_unconstrained_not_deny_all(self) -> None:
        """Empty allowed_sops must mean 'no whitelist restriction', NOT
        'allow nothing'.

        This locks in the conventional allow-list semantic (AWS IAM,
        kubernetes NetworkPolicy, iptables, AF's ``allowed_tools``):
        an empty whitelist skips the whitelist filter entirely; every
        discovered SOP passes through. The alternative interpretation
        (empty = deny all) would be a footgun because the default state
        (factory=list → []) would silently hide every SOP.
        """
        # Construct CI with EXPLICITLY empty allowed_sops (the very state
        # we're protecting against being interpreted as "deny all").
        ci = ConversationalInferencer(
            base_inferencer=_MockBase(),
            allowed_sops=[],  # ← MUST mean "no restriction", not "deny all"
        )
        rendered = ci._render_prompt("hello")

        # Multiple framework SOPs MUST still be visible. If a regression
        # flipped the semantic, the rendered prompt would lose every
        # SOP bullet — caught by these assertions.
        sops_that_must_remain_visible = [
            "code_optimization",
            "model_optimization",
            "sop_creation",
        ]
        missing = [
            name for name in sops_that_must_remain_visible
            if f"(`{name}`)" not in rendered
        ]
        self.assertEqual(
            missing, [],
            f"REGRESSION: empty allowed_sops was interpreted as 'deny all' "
            f"and silently hid framework SOPs {missing}. The conventional "
            f"semantic is: empty allow-list = NO whitelist restriction "
            f"(every SOP passes). See ConversationalInferencer attrib "
            f"docstring 'CRITICAL SEMANTIC' block for full rationale.",
        )

    def test_empty_disallowed_sops_means_no_drops(self) -> None:
        """Symmetric guard: empty disallowed_sops must mean 'no drops',
        not 'drop everything'."""
        ci = ConversationalInferencer(
            base_inferencer=_MockBase(),
            disallowed_sops=[],  # ← MUST mean "no drops"
        )
        rendered = ci._render_prompt("hello")
        # Same framework SOPs MUST all be visible.
        for name in ("code_optimization", "model_optimization"):
            self.assertIn(
                f"(`{name}`)", rendered,
                f"REGRESSION: empty disallowed_sops was interpreted as "
                f"'drop everything' and silently hid {name}.",
            )

    def test_unset_filters_match_explicitly_empty_filters(self) -> None:
        """The implicit default (factory=list → []) must behave identically
        to the explicit `allowed_sops=[], disallowed_sops=[]`. If they ever
        diverge, the documented semantic is broken."""
        ci_default = ConversationalInferencer(base_inferencer=_MockBase())
        ci_explicit_empty = ConversationalInferencer(
            base_inferencer=_MockBase(),
            allowed_sops=[],
            disallowed_sops=[],
        )
        self.assertEqual(
            ci_default._render_prompt("hello"),
            ci_explicit_empty._render_prompt("hello"),
            "Default (unset) filters must behave byte-identically to "
            "explicitly-empty filters — otherwise the 'empty = unconstrained' "
            "contract is broken at the default-construction boundary.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
