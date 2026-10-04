"""Opt-in real qualification of the native Claude Code SDK backend (plan §12.2).

Runs the qualification sequence of ``_helpers/native_qualification.py`` against
the real vendor and checks transport and session evidence.

Opt in with ``AF_REAL_NATIVE=1`` (from the AgentFoundation root)::

    AF_REAL_NATIVE=1 python3 -m pytest -s -p no:cacheprovider \\
        test/agent_foundation/common/inferencers/test_real_integration/test_real_native_claude_sdk.py
"""

import unittest

import pytest
from later.unittest import TestCase

from ._helpers.native_qualification import run_qualification, skip_reason

pytestmark = pytest.mark.integration
_SKIP = skip_reason("claude_sdk")


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class RealNativeClaudeSdkTest(TestCase):
    async def test_qualification_sequence(self) -> None:
        checks = await run_qualification("claude_sdk")
        self.assertEqual(checks.failures, [], checks.summary())
