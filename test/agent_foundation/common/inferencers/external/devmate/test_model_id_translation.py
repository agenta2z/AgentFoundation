"""Tests for ``model_id`` → ``model_name`` translation in Devmate inferencers.

Verifies that:
1. Every ``ClaudeModels`` enum value (the canonical single source of truth)
   translates to a value that exists in Devmate's server-side ``ModelName``
   enum (``fbcode/devai/config/llm_config.py``).
2. The translation precedence rule holds: ``model_id`` (when non-empty)
   wins over ``model_name``.
3. Both ``DevmateCliInferencer`` and ``DevmateSDKInferencer`` apply
   identical translation behavior.

The ``ClaudeModels`` enum is read indirectly (as raw strings) to avoid
pulling the optional ``anthropic`` SDK into the unit-test path.
"""

import unittest

from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
    DevmateCliInferencer,
    DevmateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.common import (
    resolve_model_tag,
)


# ---------------------------------------------------------------------------
# Source-of-truth mappings (kept in lockstep with the upstream enums).
# ---------------------------------------------------------------------------

# Copied from ``apis/claude_llm.py`` ``ClaudeModels``. Reproduced here so the
# unit test doesn't import ``apis/claude_llm.py`` (which transitively imports
# the optional ``anthropic`` SDK package).
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

# Expected ``ClaudeModels.<name>`` → Devmate ``ModelName`` value mapping.
# Verified against ``fbcode/devai/config/llm_config.py`` ``ModelName`` enum.
_EXPECTED_DEVMATE_NAME: dict[str, str] = {
    "CLAUDE_45_HAIKU": "claude-haiku-4.5",
    "CLAUDE_45_SONNET": "claude-sonnet-4.5",
    "CLAUDE_45_OPUS": "claude-opus-4.5",
    "CLAUDE_46_SONNET": "claude-sonnet-4.6",
    "CLAUDE_46_OPUS": "claude-opus-4.6",
    "CLAUDE_47_OPUS": "claude-opus-4.7",
    "CLAUDE_46_OPUS_1M": "claude-opus-4.6-1m",
    "CLAUDE_47_OPUS_1M": "claude-opus-4.7-1m",
    "CLAUDE_47_OPUS_LATEST": "claude-opus-4.7",
    "CLAUDE_47_OPUS_LATEST_1M": "claude-opus-4.7-1m",
}

# Subset of Devmate's ``ModelName`` enum values that we expect ``ClaudeModels``
# to resolve to. The full enum has ~130 entries (Llama, GPT, Gemini, etc.) —
# we only enumerate the Claude-family Plugboard values here.
_DEVMATE_CLAUDE_NAMES: frozenset[str] = frozenset({
    "claude3-haiku",
    "claude3.5-haiku",
    "claude-haiku-4.5",
    "claude3.5-sonnet",
    "claude3.7-sonnet",
    "claude4-sonnet",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "claude-sonnet-4.6-1m",
    "claude-opus-4.5",
    "claude-opus-4.6",
    "claude-opus-4.7",
    "claude-opus-4.6-1m",
    "claude-opus-4.7-1m",
    "gcp-claude-4-opus",  # CLAUDE_4_OPUS_PLUGBOARD
})


# ---------------------------------------------------------------------------
# Translation correctness
# ---------------------------------------------------------------------------

class ResolveModelTagClaudeModelsTest(unittest.TestCase):
    """Every ``ClaudeModels`` value resolves to a Devmate-valid name."""

    def test_all_claude_models_translate_to_valid_devmate_name(self) -> None:
        for name, value in _CLAUDE_MODELS.items():
            with self.subTest(claude_model=name, raw=value):
                resolved = resolve_model_tag(value)
                self.assertIn(
                    resolved,
                    _DEVMATE_CLAUDE_NAMES,
                    f"{name}={value!r} resolved to {resolved!r}, "
                    f"which is NOT in Devmate's ModelName Claude subset. "
                    f"Either add it to Devmate's enum or update "
                    f"_KNOWN_ALIASES in devmate/common.py.",
                )

    def test_claude_models_translate_to_expected_value(self) -> None:
        for name, raw in _CLAUDE_MODELS.items():
            with self.subTest(claude_model=name):
                expected = _EXPECTED_DEVMATE_NAME[name]
                actual = resolve_model_tag(raw)
                self.assertEqual(
                    actual, expected,
                    f"{name}: resolve_model_tag({raw!r}) returned {actual!r}, "
                    f"expected {expected!r}",
                )

    def test_already_devmate_format_is_idempotent(self) -> None:
        """Devmate-native values pass through unchanged."""
        for name in _DEVMATE_CLAUDE_NAMES:
            with self.subTest(devmate_name=name):
                self.assertEqual(resolve_model_tag(name), name)

    def test_unknown_value_passes_through_unchanged(self) -> None:
        """Unmapped values pass through (caller-responsible to validate)."""
        self.assertEqual(resolve_model_tag("not-a-real-model"), "not-a-real-model")
        self.assertEqual(resolve_model_tag(""), "")

    def test_bracket_form_normalized_after_dash_to_dot(self) -> None:
        """``claude-opus-4-7[1m]`` → ``claude-opus-4.7-1m`` (not the bracketed form)."""
        # Both Anthropic-form and already-dotted form should reach the same target.
        self.assertEqual(resolve_model_tag("claude-opus-4-7[1m]"), "claude-opus-4.7-1m")
        self.assertEqual(resolve_model_tag("claude-opus-4.7[1m]"), "claude-opus-4.7-1m")

    def test_opus_short_alias_resolves_to_latest(self) -> None:
        """Anthropic's ``opus`` alias maps to the latest devmate Opus."""
        self.assertEqual(resolve_model_tag("opus"), "claude-opus-4.7")
        self.assertEqual(resolve_model_tag("opus[1m]"), "claude-opus-4.7-1m")


# ---------------------------------------------------------------------------
# Precedence rule + integration with inferencer __attrs_post_init__
# ---------------------------------------------------------------------------

class DevmateCliInferencerModelIdTest(unittest.TestCase):
    """``model_id`` translation is applied in ``DevmateCliInferencer``."""

    def test_default_model_id_empty_preserves_model_name_default(self) -> None:
        """No ``model_id`` set → default ``model_name`` is used unchanged."""
        inf = DevmateCliInferencer()
        self.assertEqual(inf.model_id, "")
        self.assertEqual(inf.model_name, "claude-opus-4.7-1m")

    def test_model_id_set_overrides_model_name(self) -> None:
        """``model_id`` wins when both are set explicitly."""
        inf = DevmateCliInferencer(
            model_id="claude-opus-4-7",
            model_name="claude-sonnet-4.5",  # should be overwritten
        )
        self.assertEqual(inf.model_id, "claude-opus-4-7")
        self.assertEqual(inf.model_name, "claude-opus-4.7")

    def test_model_id_translates_each_claude_model(self) -> None:
        """Every ``ClaudeModels`` value sets a valid ``model_name``."""
        for name, value in _CLAUDE_MODELS.items():
            with self.subTest(claude_model=name):
                inf = DevmateCliInferencer(model_id=value)
                self.assertEqual(inf.model_name, _EXPECTED_DEVMATE_NAME[name])
                self.assertIn(inf.model_name, _DEVMATE_CLAUDE_NAMES)

    def test_model_id_translation_visible_in_construct_command(self) -> None:
        """The translated ``model_name`` appears in the generated CLI command."""
        inf = DevmateCliInferencer(model_id="claude-opus-4-7[1m]")
        command = inf.construct_command("hello")
        self.assertIn('"model_name=claude-opus-4.7-1m"', command)

    def test_explicit_model_name_used_when_model_id_empty(self) -> None:
        """Explicit ``model_name`` (no ``model_id``) is honored unchanged."""
        inf = DevmateCliInferencer(model_name="claude-sonnet-4.6")
        self.assertEqual(inf.model_name, "claude-sonnet-4.6")


class DevmateSDKInferencerModelIdTest(unittest.TestCase):
    """``model_id`` translation is applied in ``DevmateSDKInferencer``."""

    def test_default_model_id_empty_preserves_model_name_default(self) -> None:
        inf = DevmateSDKInferencer()
        self.assertEqual(inf.model_id, "")
        self.assertEqual(inf.model_name, "claude-opus-4.7-1m")

    def test_model_id_set_overrides_model_name(self) -> None:
        inf = DevmateSDKInferencer(
            model_id="claude-opus-4-7",
            model_name="claude-sonnet-4.5",
        )
        self.assertEqual(inf.model_name, "claude-opus-4.7")

    def test_model_id_translates_each_claude_model(self) -> None:
        for name, value in _CLAUDE_MODELS.items():
            with self.subTest(claude_model=name):
                inf = DevmateSDKInferencer(model_id=value)
                self.assertEqual(inf.model_name, _EXPECTED_DEVMATE_NAME[name])
                self.assertIn(inf.model_name, _DEVMATE_CLAUDE_NAMES)


# ---------------------------------------------------------------------------
# Symmetric behavior between CLI and SDK
# ---------------------------------------------------------------------------

class CliAndSdkProduceSameModelNameTest(unittest.TestCase):
    """For every ``ClaudeModels`` value, CLI and SDK resolve to the same name."""

    def test_cli_and_sdk_resolve_identically(self) -> None:
        for name, value in _CLAUDE_MODELS.items():
            with self.subTest(claude_model=name):
                cli = DevmateCliInferencer(model_id=value)
                sdk = DevmateSDKInferencer(model_id=value)
                self.assertEqual(
                    cli.model_name, sdk.model_name,
                    f"CLI/SDK divergence for {name}: "
                    f"CLI={cli.model_name!r} SDK={sdk.model_name!r}",
                )


if __name__ == "__main__":
    unittest.main()
