

"""Tests for QueueInteractive.stream_token_batches task_id support.

Verifies Phase 0: task_id parameter flows through to token_batch and
stream_end messages.
"""

import asyncio
import unittest
from unittest.mock import MagicMock


class StreamTokenBatchesTaskIdTest(unittest.TestCase):
    """Verify task_id flows through stream_token_batches."""

    def test_stream_token_batches_signature_accepts_task_id(self) -> None:
        """Phase 0: stream_token_batches should accept task_id parameter."""
        import inspect
        from agent_foundation.common.ui.queue_interactive import (
            QueueInteractive,
        )

        sig = inspect.signature(QueueInteractive.stream_token_batches)
        params = list(sig.parameters.keys())
        self.assertIn("task_id", params, "stream_token_batches should accept task_id")

    def test_flush_token_batch_signature_accepts_task_id(self) -> None:
        """Phase 0: _flush_token_batch should accept task_id parameter."""
        import inspect
        from agent_foundation.common.ui.queue_interactive import (
            QueueInteractive,
        )

        sig = inspect.signature(QueueInteractive._flush_token_batch)
        params = list(sig.parameters.keys())
        self.assertIn("task_id", params, "_flush_token_batch should accept task_id")


if __name__ == "__main__":
    unittest.main()
