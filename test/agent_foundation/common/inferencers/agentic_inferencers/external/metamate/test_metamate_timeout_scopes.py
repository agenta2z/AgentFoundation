# pyre-strict

"""MetaMate SDK timeout scopes — one budget per scope, never one shared number.

``InferencerBase.total_timeout_seconds`` bounds the WHOLE retry loop (every
attempt plus every guardrail evaluation); ``stream_total_timeout_seconds``
bounds ONE streaming poll loop. MetaMate used to override the former with the
latter's value, so a single number governed both scopes. Two consequences, both
observed in production:

* ``_compute_effective_timeout`` takes ``min(attempt_timeout, remaining)`` and
  ``remaining <= total``, so a finite total silently defeated the flow config's
  ``attempt_timeout_seconds: 7200`` hang guard — it could never bind.
* One attempt was allowed the entire budget, so any retry was guaranteed to
  breach it. Nodes that were still making progress were killed mid-flight and
  their output discarded.

These tests pin the split so the two scopes cannot be collapsed again.
"""

from __future__ import annotations

import unittest

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
    MetamateSDKInferencer,
)


class MetamateTimeoutScopesTest(unittest.TestCase):
    def test_retry_loop_has_no_cumulative_cap_by_default(self) -> None:
        """The inherited retry-loop budget must stay disabled (framework default)."""
        inf = MetamateSDKInferencer(api_key=None)

        self.assertEqual(
            inf.total_timeout_seconds,
            0,
            "A finite total_timeout_seconds is spent on attempts AND guardrail "
            "evaluations and is min()'d with the per-attempt cap, so it disables "
            "the attempt_timeout_seconds hang guard and kills progressing nodes.",
        )

    def test_streaming_call_keeps_its_own_budget(self) -> None:
        """Disabling the retry-loop cap must not leave one poll loop unbounded."""
        inf = MetamateSDKInferencer(api_key=None)

        self.assertEqual(inf.stream_total_timeout_seconds, 1800)

    def test_the_two_scopes_are_distinct_attributes(self) -> None:
        """Setting one scope must never move the other."""
        inf = MetamateSDKInferencer(
            api_key=None,
            total_timeout_seconds=900,
            stream_total_timeout_seconds=120,
        )

        self.assertEqual(inf.total_timeout_seconds, 900)
        self.assertEqual(inf.stream_total_timeout_seconds, 120)

    def test_attempt_cap_can_bind_under_the_default(self) -> None:
        """The invariant the flow config depends on.

        ``attempt_timeout_seconds`` only binds while it is smaller than the
        remaining total budget. With the total disabled there is no ceiling to
        min() against, so the configured per-attempt hang cap governs.
        """
        inf = MetamateSDKInferencer(api_key=None, attempt_timeout_seconds=7200)

        self.assertEqual(inf.attempt_timeout_seconds, 7200)
        self.assertFalse(
            0 < inf.total_timeout_seconds <= inf.attempt_timeout_seconds,
            "total_timeout_seconds must not be a positive value <= the "
            "per-attempt cap — that combination makes the cap unreachable.",
        )
