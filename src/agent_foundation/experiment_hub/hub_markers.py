# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Stream completion markers shared between the hub runner and the stream tailer.

Reimplemented locally (no ``rankevolve`` import) but with the EXACT marker
strings the original ``rankevolve.src.common.streaming.markers`` used so any
producer/consumer that already wrote these literals stays compatible.
"""

from __future__ import annotations

STREAM_DONE_MARKER: str = "--- STREAM COMPLETED SUCCESSFULLY ---"
STREAM_FAIL_MARKER: str = "--- STREAM FAILED:"
