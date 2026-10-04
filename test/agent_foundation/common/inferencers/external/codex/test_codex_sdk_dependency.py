"""CodexSdkInferencer without its SDK: the official ``openai-codex`` package
(module ``openai_codex``) is not vendored in fbsource, so a missing SDK must be
a typed, non-retryable dependency error."""

import asyncio
import tempfile
import unittest
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_sdk_inferencer import (
    CodexSdkInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import MissingDependencyError


class MissingSdkTest(unittest.TestCase):
    def test_connect_without_the_sdk_is_a_missing_dependency(self) -> None:
        inferencer = CodexSdkInferencer(target_path=tempfile.mkdtemp())
        with mock.patch.dict("sys.modules", {"openai_codex": None}):
            with self.assertRaises(MissingDependencyError) as caught:
                asyncio.run(inferencer.aconnect())
        self.assertIn("pip install openai-codex", str(caught.exception))
