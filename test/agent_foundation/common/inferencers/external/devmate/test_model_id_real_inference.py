#!/usr/bin/env python3
"""Real-inference verification for every ``ClaudeModels`` → Devmate mapping.

For each ``ClaudeModels`` enum value, we:
1. Create a ``DevmateCliInferencer`` with that value as ``model_id``.
2. Verify the translation produced a Devmate-valid ``model_name`` (sanity).
3. Issue a tiny real Devmate inference (``"Reply with one word: pong"``).
4. Assert the call succeeded AND returned a non-empty response AND
   surfaced a session_id (proves devmate accepted the model name and
   the model actually ran end-to-end).

These tests are SLOW (~15-30s per model) and require an authenticated
devmate environment, so they are gated behind ``DEVMATE_RUN_REAL=1`` via
``conftest.py``. They run in CI only when explicitly opted in.

Run locally with::

    DEVMATE_RUN_REAL=1 python3.12 -m pytest \
      test/agent_foundation/common/inferencers/external/devmate/\
test_model_id_real_inference.py -v
"""

import unittest

from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
    DevmateCliInferencer,
)

# ClaudeModels copied as raw strings (avoids the optional anthropic SDK import).
_CLAUDE_MODELS: dict[str, str] = {
    "CLAUDE_45_HAIKU": "claude-haiku-4-5-20251001",
    "CLAUDE_45_SONNET": "claude-sonnet-4-5-20250929",
    "CLAUDE_45_OPUS": "claude-opus-4-5-20251101",
    "CLAUDE_46_SONNET": "claude-sonnet-4-6",
    "CLAUDE_46_OPUS": "claude-opus-4-6",
    "CLAUDE_47_OPUS": "claude-opus-4-7",
    "CLAUDE_46_OPUS_1M": "claude-opus-4-6[1m]",
    "CLAUDE_47_OPUS_1M": "claude-opus-4-7[1m]",
    "CLAUDE_47_OPUS_LATEST": "opus",
    "CLAUDE_47_OPUS_LATEST_1M": "opus[1m]",
}

_REPO_PATH = "/data/users/zgchen/fbsource"
_PROMPT = "Reply with one word: pong"
_MAX_TOKENS = 128


def _run_real_inference(model_id: str) -> tuple[bool, str, str | None]:
    """Run a real devmate inference for the given ``model_id``.

    Returns ``(success, output, session_id)``.
    """
    inferencer = DevmateCliInferencer(
        target_path=_REPO_PATH,
        model_id=model_id,
        max_tokens=_MAX_TOKENS,
        no_create_commit=True,
    )
    response = inferencer(_PROMPT)
    success = bool(getattr(response, "success", False))
    output = getattr(response, "output", "") or ""
    session_id = getattr(response, "session_id", None)
    return success, output, session_id


def _assert_real_inference_works(test: unittest.TestCase, model_id: str) -> None:
    success, output, session_id = _run_real_inference(model_id)

    # The translated model_name must reach the CLI as a recognized
    # Devmate ModelName — devmate rejects unknown names with a fast
    # validation error, so any return_code==0 + non-empty output here
    # proves the translation produced a name devmate accepted.
    test.assertTrue(
        success,
        f"Devmate inference failed for model_id={model_id!r}. "
        f"This usually means resolve_model_tag produced a name that "
        f"Devmate's ModelName enum doesn't accept. Output: {output[:300]!r}",
    )
    test.assertGreater(
        len(output), 0,
        f"Empty response for model_id={model_id!r}; expected the model "
        f"to actually run and emit output. Got: {output!r}",
    )
    test.assertIsNotNone(
        session_id,
        f"No session_id extracted for model_id={model_id!r}; either "
        f"devmate didn't actually run a session, or session-id parsing "
        f"broke. Output snippet: {output[:300]!r}",
    )


# Generate one test method per model so failures are clearly named in
# the pytest report (rather than collapsed under a single parameterized test).
class ClaudeModelsRealInferenceTest(unittest.TestCase):
    """Real devmate inference for every ``ClaudeModels`` enum value."""

    pass


def _make_test_method(model_name_label: str, model_id_value: str):
    def _test(self: ClaudeModelsRealInferenceTest) -> None:
        _assert_real_inference_works(self, model_id_value)
    _test.__name__ = f"test_real_inference_{model_name_label.lower()}"
    _test.__doc__ = (
        f"Verify devmate accepts ClaudeModels.{model_name_label} "
        f"(raw={model_id_value!r}) via the model_id translation path."
    )
    return _test


for _label, _value in _CLAUDE_MODELS.items():
    setattr(
        ClaudeModelsRealInferenceTest,
        f"test_real_inference_{_label.lower()}",
        _make_test_method(_label, _value),
    )


# Also verify the new default (no model_id, no model_name override) actually
# runs end-to-end. Catches regressions in the default value.
class DefaultModelRealInferenceTest(unittest.TestCase):
    """Default-only inferencer (no model_id, no model_name override)."""

    def test_default_model_runs_end_to_end(self) -> None:
        inferencer = DevmateCliInferencer(
            target_path=_REPO_PATH,
            max_tokens=_MAX_TOKENS,
            no_create_commit=True,
        )
        # The default should be ``claude-opus-4.7-1m``; verify before sending.
        self.assertEqual(
            inferencer.model_name, "claude-opus-4.7-1m",
            f"Default model_name drifted: {inferencer.model_name!r}",
        )
        response = inferencer(_PROMPT)
        self.assertTrue(getattr(response, "success", False))
        self.assertGreater(len(getattr(response, "output", "") or ""), 0)
        self.assertIsNotNone(getattr(response, "session_id", None))


if __name__ == "__main__":
    unittest.main()
