"""Opt-in real qualification of the native Devmate dm backend (plan §12.2).

Runs the qualification sequence of ``_helpers/native_qualification.py`` against
the real vendor and checks transport and session evidence.
It is skipped unless CAT credentials for dm's workflow mode exist (see
``skip_reason``): without them dm-core's MCP warmup stalls AF tool turns.

Opt in with ``AF_REAL_NATIVE=1`` (from the AgentFoundation root)::

    AF_REAL_NATIVE=1 python3 -m pytest -s -p no:cacheprovider \\
        test/agent_foundation/common/inferencers/test_real_integration/test_real_native_devmate_dm.py
"""

import unittest

import pytest
from later.unittest import TestCase

from ._helpers.native_qualification import run_qualification, skip_reason

pytestmark = pytest.mark.integration
_SKIP = skip_reason("devmate_dm")


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class RealNativeDevmateDmTest(TestCase):
    async def test_qualification_sequence(self) -> None:
        checks = await run_qualification("devmate_dm")
        self.assertEqual(checks.failures, [], checks.summary())
