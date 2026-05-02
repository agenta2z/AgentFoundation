"""Unit tests: ClaudeCodeSdkInferencer ``effort`` and ``permission_mode`` routing.

The SDK wrapper splits values between ``ClaudeAgentOptions``'s typed fields
(``permission_mode`` / ``effort``) and ``extra_args`` so that values outside
the SDK's narrower Literal still reach the CLI subprocess. These tests cover
two complementary surfaces:

1. ``_build_permission_effort_kwargs()`` — the pure routing helper. Verified
   directly without any SDK / network involvement.
2. ``aconnect()`` — verified by patching ``ClaudeAgentOptions`` and
   ``ClaudeSDKClient`` so we can capture exactly what the inferencer
   constructs, without spawning the SDK subprocess.

End-to-end behavior against the real SDK subprocess is covered in
``test_effort_and_permission_mode_real.py``.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer import (
    ClaudeCodeSdkInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.common import (
    SDK_NATIVE_EFFORT_LEVELS,
    SDK_NATIVE_PERMISSION_MODES,
)


# ---------------------------------------------------------------------------
# Layer 1: the pure routing helper
# ---------------------------------------------------------------------------


class RoutingHelperDefaultsTest(unittest.TestCase):
    """Default-constructed inferencer routes effort=max via the typed field."""

    def test_default_effort_is_max(self):
        self.assertEqual(ClaudeCodeSdkInferencer().effort, "max")

    def test_default_permission_mode_is_none(self):
        self.assertIsNone(ClaudeCodeSdkInferencer().permission_mode)

    def test_default_helper_routes_effort_max_via_typed_field(self):
        sdk_kwargs, extra_args = (
            ClaudeCodeSdkInferencer()._build_permission_effort_kwargs()
        )
        self.assertEqual(sdk_kwargs, {"effort": "max"})
        self.assertEqual(extra_args, {})


class EffortRoutingTest(unittest.TestCase):
    """Effort levels split correctly between typed field and extra_args."""

    def test_native_effort_levels_use_typed_field(self):
        for level in SDK_NATIVE_EFFORT_LEVELS:
            with self.subTest(level=level):
                inf = ClaudeCodeSdkInferencer(effort=level)
                sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
                self.assertEqual(sdk_kwargs.get("effort"), level)
                self.assertNotIn("effort", extra_args)

    def test_xhigh_routes_via_extra_args(self):
        # `xhigh` is a CLI value not in the SDK's typed Literal — must route
        # through extra_args so the SDK transport still emits --effort xhigh.
        inf = ClaudeCodeSdkInferencer(effort="xhigh")
        sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
        self.assertNotIn("effort", sdk_kwargs)
        self.assertEqual(extra_args.get("effort"), "xhigh")

    def test_effort_none_routes_nothing(self):
        inf = ClaudeCodeSdkInferencer(effort=None)
        sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
        self.assertNotIn("effort", sdk_kwargs)
        self.assertNotIn("effort", extra_args)


class PermissionModeRoutingTest(unittest.TestCase):
    """Permission modes split correctly between typed field and extra_args."""

    def test_native_modes_use_typed_field(self):
        for mode in SDK_NATIVE_PERMISSION_MODES:
            with self.subTest(mode=mode):
                inf = ClaudeCodeSdkInferencer(permission_mode=mode)
                sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
                self.assertEqual(sdk_kwargs.get("permission_mode"), mode)
                self.assertNotIn("permission-mode", extra_args)

    def test_auto_routes_via_extra_args(self):
        inf = ClaudeCodeSdkInferencer(permission_mode="auto")
        sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
        self.assertNotIn("permission_mode", sdk_kwargs)
        self.assertEqual(extra_args.get("permission-mode"), "auto")

    def test_dont_ask_routes_via_extra_args(self):
        inf = ClaudeCodeSdkInferencer(permission_mode="dontAsk")
        sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
        self.assertNotIn("permission_mode", sdk_kwargs)
        self.assertEqual(extra_args.get("permission-mode"), "dontAsk")

    def test_permission_mode_none_routes_nothing(self):
        inf = ClaudeCodeSdkInferencer(permission_mode=None)
        sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
        self.assertNotIn("permission_mode", sdk_kwargs)
        self.assertNotIn("permission-mode", extra_args)


class CombinedRoutingTest(unittest.TestCase):
    """Both attributes can be set together and route independently."""

    def test_native_mode_and_native_effort_both_typed(self):
        inf = ClaudeCodeSdkInferencer(permission_mode="plan", effort="high")
        sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
        self.assertEqual(sdk_kwargs, {"permission_mode": "plan", "effort": "high"})
        self.assertEqual(extra_args, {})

    def test_native_mode_and_cli_only_effort_split(self):
        inf = ClaudeCodeSdkInferencer(permission_mode="plan", effort="xhigh")
        sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
        self.assertEqual(sdk_kwargs, {"permission_mode": "plan"})
        self.assertEqual(extra_args, {"effort": "xhigh"})

    def test_cli_only_mode_and_native_effort_split(self):
        inf = ClaudeCodeSdkInferencer(permission_mode="auto", effort="max")
        sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
        self.assertEqual(sdk_kwargs, {"effort": "max"})
        self.assertEqual(extra_args, {"permission-mode": "auto"})

    def test_both_cli_only(self):
        inf = ClaudeCodeSdkInferencer(permission_mode="dontAsk", effort="xhigh")
        sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
        self.assertEqual(sdk_kwargs, {})
        self.assertEqual(
            extra_args, {"permission-mode": "dontAsk", "effort": "xhigh"}
        )


# ---------------------------------------------------------------------------
# Layer 2: aconnect() actually feeds those kwargs into ClaudeAgentOptions
# ---------------------------------------------------------------------------


class _CapturedOptionsHelper:
    """Patch ``ClaudeAgentOptions`` + ``ClaudeSDKClient`` and capture what
    ``aconnect()`` passes to ``ClaudeAgentOptions(...)``."""

    def __init__(self):
        self.captured_options_kwargs = None

    async def run(self, inferencer):
        captured = {}

        # The real ClaudeAgentOptions is a dataclass; we just need to capture
        # what the inferencer passes. We return a MagicMock so subsequent
        # ClaudeSDKClient(options=...) doesn't error.
        def _fake_options(**kwargs):
            captured.update(kwargs)
            return MagicMock(name="ClaudeAgentOptions")

        fake_client = MagicMock(name="ClaudeSDKClient")
        fake_client.connect = AsyncMock()
        fake_client.disconnect = AsyncMock()

        # Patch the SDK module that aconnect() imports lazily inside the function.
        # Because the import is inside aconnect(), we patch the attribute on the
        # already-imported module so the local import resolves to our fakes.
        import claude_agent_sdk
        with patch.object(claude_agent_sdk, "ClaudeAgentOptions", _fake_options), \
             patch.object(
                 claude_agent_sdk, "ClaudeSDKClient", lambda options: fake_client,
             ):
            await inferencer.aconnect()
        # Tear down the background task aconnect() left running so the loop
        # closes cleanly.
        await inferencer.adisconnect()
        self.captured_options_kwargs = captured


def _capture_aconnect_options(inferencer):
    """Run aconnect() on a fresh event loop and return the captured kwargs."""
    helper = _CapturedOptionsHelper()
    asyncio.run(helper.run(inferencer))
    return helper.captured_options_kwargs


class AconnectIntegrationTest(unittest.TestCase):
    """``aconnect()`` constructs ``ClaudeAgentOptions`` using the helper output."""

    def test_default_inferencer_passes_effort_max_typed(self):
        kwargs = _capture_aconnect_options(ClaudeCodeSdkInferencer())
        self.assertEqual(kwargs.get("effort"), "max")
        # Default permission_mode is None — must NOT be set
        self.assertNotIn("permission_mode", kwargs)
        # extra_args is always passed (factory default {} or our dict)
        self.assertEqual(kwargs.get("extra_args"), {})

    def test_plan_and_high_pass_via_typed_fields(self):
        inf = ClaudeCodeSdkInferencer(permission_mode="plan", effort="high")
        kwargs = _capture_aconnect_options(inf)
        self.assertEqual(kwargs.get("permission_mode"), "plan")
        self.assertEqual(kwargs.get("effort"), "high")
        self.assertEqual(kwargs.get("extra_args"), {})

    def test_auto_and_xhigh_pass_via_extra_args(self):
        inf = ClaudeCodeSdkInferencer(permission_mode="auto", effort="xhigh")
        kwargs = _capture_aconnect_options(inf)
        self.assertNotIn("permission_mode", kwargs)
        self.assertNotIn("effort", kwargs)
        self.assertEqual(
            kwargs.get("extra_args"),
            {"permission-mode": "auto", "effort": "xhigh"},
        )

    def test_effort_none_omits_both_typed_and_extra(self):
        inf = ClaudeCodeSdkInferencer(effort=None)
        kwargs = _capture_aconnect_options(inf)
        self.assertNotIn("effort", kwargs)
        # extra_args may exist as {} but must not contain "effort"
        self.assertNotIn("effort", kwargs.get("extra_args", {}))


# ---------------------------------------------------------------------------
# Layer 3: cross-check against the real SDK transport's command-building logic
# ---------------------------------------------------------------------------


class SdkTransportCrossCheckTest(unittest.TestCase):
    """Sanity-check: the real SDK transport actually forwards what we set.

    This doesn't spawn a subprocess — it constructs a real ``ClaudeAgentOptions``
    with our routing applied, then inspects the command list the SDK transport
    would build. Confirms the typed fields and ``extra_args`` both end up as
    real CLI flags.
    """

    def _build_cmd(self, **inf_kwargs):
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk._internal.transport.subprocess_cli import (
            SubprocessCLITransport,
        )

        inf = ClaudeCodeSdkInferencer(**inf_kwargs)
        sdk_kwargs, extra_args = inf._build_permission_effort_kwargs()
        options = ClaudeAgentOptions(extra_args=extra_args, **sdk_kwargs)
        # SubprocessCLITransport._build_command is the actual code path that
        # turns options into a CLI argv list. We don't execute, just inspect.
        transport = SubprocessCLITransport(prompt="ignored", options=options)
        return transport._build_command()

    def test_native_effort_appears_as_effort_flag(self):
        cmd = self._build_cmd(effort="max")
        self.assertIn("--effort", cmd)
        self.assertIn("max", cmd)

    def test_xhigh_via_extra_args_appears_as_effort_flag(self):
        cmd = self._build_cmd(effort="xhigh")
        self.assertIn("--effort", cmd)
        self.assertIn("xhigh", cmd)

    def test_native_permission_mode_appears_as_permission_mode_flag(self):
        cmd = self._build_cmd(permission_mode="plan")
        self.assertIn("--permission-mode", cmd)
        self.assertIn("plan", cmd)

    def test_auto_via_extra_args_appears_as_permission_mode_flag(self):
        cmd = self._build_cmd(permission_mode="auto")
        self.assertIn("--permission-mode", cmd)
        self.assertIn("auto", cmd)

    def test_combined_settings_emit_both_flags(self):
        cmd = self._build_cmd(permission_mode="dontAsk", effort="xhigh")
        joined = " ".join(cmd)
        self.assertIn("--permission-mode dontAsk", joined)
        self.assertIn("--effort xhigh", joined)


if __name__ == "__main__":
    unittest.main()
