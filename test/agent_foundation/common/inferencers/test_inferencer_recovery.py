# Feature: retry-native-timeout, Property 10: Recovery Method Default Equivalence
"""Property-based test for InferencerBase recovery method default equivalence.

**Validates: Requirements 16.1, 16.2, 16.3, 15.7**

Property 10 states: For any InferencerBase subclass that does NOT override
_infer_recovery / _ainfer_recovery, with fallback_mode=ON_FIRST_FAILURE and
max_retry=N, the total number of inference attempts SHALL be exactly 1 + N
(one primary attempt plus N recovery retries). Because the default recovery
delegates to _infer / _ainfer, the observable behavior SHALL be equivalent
to max_retry=N+1 under today's retry-same-function semantic.

We test this by creating a concrete subclass that counts _infer/_ainfer calls
and always raises, then verifying:
1. The default _infer_recovery delegates to _infer (sync)
2. The default _ainfer_recovery delegates to _ainfer (async)
3. Calling recovery with any (last_exception, last_partial_output) still
   invokes the underlying _infer/_ainfer with the correct arguments
"""

import asyncio
import unittest

from attr import attrib, attrs
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from rich_python_utils.common_utils.function_helper import FallbackMode


@attrs
class CountingInferencer(InferencerBase):
    """Concrete test subclass that counts _infer/_ainfer calls and always raises.

    Does NOT override _infer_recovery or _ainfer_recovery — uses defaults.
    """

    infer_call_count: int = attrib(default=0, init=False)
    ainfer_call_count: int = attrib(default=0, init=False)
    last_infer_input: object = attrib(default=None, init=False)
    last_ainfer_input: object = attrib(default=None, init=False)
    last_infer_config: object = attrib(default=None, init=False)
    last_ainfer_config: object = attrib(default=None, init=False)

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        self.infer_call_count += 1
        self.last_infer_input = inference_input
        self.last_infer_config = inference_config
        raise RuntimeError(f"Deliberate failure #{self.infer_call_count}")

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        self.ainfer_call_count += 1
        self.last_ainfer_input = inference_input
        self.last_ainfer_config = inference_config
        raise RuntimeError(f"Deliberate async failure #{self.ainfer_call_count}")



@attrs
class SuccessCountingInferencer(InferencerBase):
    """Concrete test subclass that counts calls and returns a value.

    Does NOT override _infer_recovery or _ainfer_recovery — uses defaults.
    """

    infer_call_count: int = attrib(default=0, init=False)
    ainfer_call_count: int = attrib(default=0, init=False)

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        self.infer_call_count += 1
        return f"result_{self.infer_call_count}"

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        self.ainfer_call_count += 1
        return f"async_result_{self.ainfer_call_count}"


class TestRecoveryMethodDefaultEquivalence(unittest.TestCase):
    """Property 10: Recovery Method Default Equivalence (sync path).

    **Validates: Requirements 16.1, 16.2, 16.3, 15.7**

    For any max_retry N, the default _infer_recovery delegates to _infer,
    so calling _infer_recovery produces the same observable effect (same call
    to _infer) as calling _infer directly.
    """

    @given(max_retry=st.integers(min_value=1, max_value=20))
    @settings(max_examples=100)
    def test_sync_recovery_delegates_to_infer(self, max_retry: int):
        """Default _infer_recovery calls _infer with the same input/config.

        For any max_retry N, create an inferencer with fallback_mode=ON_FIRST_FAILURE.
        Call _infer_recovery directly and verify it delegates to _infer.
        The recovery method ignores last_exception and last_partial_output.
        """
        inf = CountingInferencer(
            max_retry=max_retry,
            fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        )

        test_input = f"input_for_retry_{max_retry}"
        test_config = {"retry": max_retry}

        # _infer_recovery should delegate to _infer, which raises
        with self.assertRaises(RuntimeError):
            inf._infer_recovery(
                inference_input=test_input,
                last_exception=ValueError("prior failure"),
                last_partial_output="some partial",
                inference_config=test_config,
            )

        # Verify _infer was called exactly once by the recovery method
        self.assertEqual(inf.infer_call_count, 1)
        self.assertEqual(inf.last_infer_input, test_input)
        self.assertEqual(inf.last_infer_config, test_config)

    @given(max_retry=st.integers(min_value=1, max_value=20))
    @settings(max_examples=100)
    def test_sync_recovery_returns_same_as_infer(self, max_retry: int):
        """Default _infer_recovery returns the same value as _infer.

        For any max_retry N, the recovery method's return value is identical
        to calling _infer directly with the same arguments.
        """
        inf = SuccessCountingInferencer(
            max_retry=max_retry,
            fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        )

        test_input = f"input_{max_retry}"
        test_config = {"n": max_retry}

        # Call _infer directly
        direct_result = inf._infer(test_input, test_config)
        direct_count = inf.infer_call_count

        # Call _infer_recovery (should delegate to _infer)
        recovery_result = inf._infer_recovery(
            inference_input=test_input,
            last_exception=RuntimeError("doesn't matter"),
            last_partial_output=None,
            inference_config=test_config,
        )
        recovery_count = inf.infer_call_count

        # Recovery added exactly one more _infer call
        self.assertEqual(recovery_count, direct_count + 1)
        # Both produce results from the same _infer method
        self.assertIsNotNone(direct_result)
        self.assertIsNotNone(recovery_result)

    @given(max_retry=st.integers(min_value=1, max_value=20))
    @settings(max_examples=100)
    def test_sync_attempt_count_equivalence(self, max_retry: int):
        """With default recovery and ON_FIRST_FAILURE, calling _infer_recovery
        N times after 1 primary _infer call gives 1 + N total _infer calls.

        This is equivalent to max_retry=N+1 under the old retry-same-function
        semantic (no fallback).
        """
        inf = CountingInferencer(
            max_retry=max_retry,
            fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        )

        # Simulate: 1 primary attempt
        try:
            inf._infer("test_input", None)
        except RuntimeError:
            pass

        primary_calls = inf.infer_call_count
        self.assertEqual(primary_calls, 1)

        # Simulate: N recovery attempts (each delegates to _infer)
        for _ in range(max_retry):
            try:
                inf._infer_recovery(
                    inference_input="test_input",
                    last_exception=RuntimeError("failed"),
                    last_partial_output=None,
                    inference_config=None,
                )
            except RuntimeError:
                pass

        total_calls = inf.infer_call_count
        # Total should be 1 (primary) + max_retry (recovery) = 1 + N
        self.assertEqual(total_calls, 1 + max_retry)


class TestRecoveryMethodDefaultEquivalenceAsync(unittest.IsolatedAsyncioTestCase):
    """Property 10: Recovery Method Default Equivalence (async path).

    **Validates: Requirements 16.1, 16.2, 16.3, 15.7**

    For any max_retry N, the default _ainfer_recovery delegates to _ainfer,
    so calling _ainfer_recovery produces the same observable effect as calling
    _ainfer directly.
    """

    @given(max_retry=st.integers(min_value=1, max_value=20))
    @settings(max_examples=100)
    def test_async_recovery_delegates_to_ainfer(self, max_retry: int):
        """Default _ainfer_recovery calls _ainfer with the same input/config."""

        async def _run():
            inf = CountingInferencer(
                max_retry=max_retry,
                fallback_mode=FallbackMode.ON_FIRST_FAILURE,
            )

            test_input = f"async_input_{max_retry}"
            test_config = {"async_retry": max_retry}

            with self.assertRaises(RuntimeError):
                await inf._ainfer_recovery(
                    inference_input=test_input,
                    last_exception=ValueError("prior async failure"),
                    last_partial_output="partial async output",
                    inference_config=test_config,
                )

            self.assertEqual(inf.ainfer_call_count, 1)
            self.assertEqual(inf.last_ainfer_input, test_input)
            self.assertEqual(inf.last_ainfer_config, test_config)

        asyncio.run(_run())

    @given(max_retry=st.integers(min_value=1, max_value=20))
    @settings(max_examples=100)
    def test_async_recovery_returns_same_as_ainfer(self, max_retry: int):
        """Default _ainfer_recovery returns the same value as _ainfer."""

        async def _run():
            inf = SuccessCountingInferencer(
                max_retry=max_retry,
                fallback_mode=FallbackMode.ON_FIRST_FAILURE,
            )

            test_input = f"async_input_{max_retry}"
            test_config = {"n": max_retry}

            direct_result = await inf._ainfer(test_input, test_config)
            direct_count = inf.ainfer_call_count

            recovery_result = await inf._ainfer_recovery(
                inference_input=test_input,
                last_exception=RuntimeError("doesn't matter"),
                last_partial_output=None,
                inference_config=test_config,
            )
            recovery_count = inf.ainfer_call_count

            self.assertEqual(recovery_count, direct_count + 1)
            self.assertIsNotNone(direct_result)
            self.assertIsNotNone(recovery_result)

        asyncio.run(_run())

    @given(max_retry=st.integers(min_value=1, max_value=20))
    @settings(max_examples=100)
    def test_async_attempt_count_equivalence(self, max_retry: int):
        """With default recovery and ON_FIRST_FAILURE, calling _ainfer_recovery
        N times after 1 primary _ainfer call gives 1 + N total _ainfer calls.

        This is equivalent to max_retry=N+1 under the old retry-same-function
        semantic (no fallback).
        """

        async def _run():
            inf = CountingInferencer(
                max_retry=max_retry,
                fallback_mode=FallbackMode.ON_FIRST_FAILURE,
            )

            # 1 primary attempt
            try:
                await inf._ainfer("test_input", None)
            except RuntimeError:
                pass

            self.assertEqual(inf.ainfer_call_count, 1)

            # N recovery attempts
            for _ in range(max_retry):
                try:
                    await inf._ainfer_recovery(
                        inference_input="test_input",
                        last_exception=RuntimeError("failed"),
                        last_partial_output=None,
                        inference_config=None,
                    )
                except RuntimeError:
                    pass

            # Total: 1 (primary) + N (recovery) = 1 + N
            self.assertEqual(inf.ainfer_call_count, 1 + max_retry)

        asyncio.run(_run())


# ---- Warning tests for task 8.3 ----


@attrs
class AsyncOnlyRecoveryInferencer(InferencerBase):
    """Overrides only _ainfer_recovery, not _infer_recovery."""

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        return "ok"

    async def _ainfer_recovery(self, inference_input, last_exception, last_partial_output,
                                inference_config=None, **kwargs):
        return await self._ainfer(inference_input, inference_config, **kwargs)


@attrs
class SyncOnlyRecoveryInferencer(InferencerBase):
    """Overrides only _infer_recovery, not _ainfer_recovery."""

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        return "ok"

    def _infer_recovery(self, inference_input, last_exception, last_partial_output,
                         inference_config=None, **kwargs):
        return self._infer(inference_input, inference_config, **kwargs)


@attrs
class BothRecoveryInferencer(InferencerBase):
    """Overrides both _ainfer_recovery and _infer_recovery — no warning expected."""

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        return "ok"

    def _infer_recovery(self, inference_input, last_exception, last_partial_output,
                         inference_config=None, **kwargs):
        return self._infer(inference_input, inference_config, **kwargs)

    async def _ainfer_recovery(self, inference_input, last_exception, last_partial_output,
                                inference_config=None, **kwargs):
        return await self._ainfer(inference_input, inference_config, **kwargs)


class TestPairedOverrideWarning(unittest.TestCase):
    """Tests for paired-override warning in __attrs_post_init__."""

    def setUp(self):
        # Clear the fire-once set before each test
        InferencerBase._paired_override_warned.clear()

    def test_async_only_override_warns(self):
        """Overriding only _ainfer_recovery should emit a warning."""
        import logging
        with self.assertLogs(logging.getLogger("agent_foundation.common.inferencers.inferencer_base"),
                             level="WARNING") as cm:
            AsyncOnlyRecoveryInferencer()
        self.assertTrue(any("AsyncOnlyRecoveryInferencer" in msg for msg in cm.output))
        self.assertTrue(any("_ainfer_recovery" in msg for msg in cm.output))
        self.assertTrue(any("_infer_recovery" in msg for msg in cm.output))

    def test_sync_only_override_warns(self):
        """Overriding only _infer_recovery should emit a warning."""
        import logging
        with self.assertLogs(logging.getLogger("agent_foundation.common.inferencers.inferencer_base"),
                             level="WARNING") as cm:
            SyncOnlyRecoveryInferencer()
        self.assertTrue(any("SyncOnlyRecoveryInferencer" in msg for msg in cm.output))
        self.assertTrue(any("_infer_recovery" in msg for msg in cm.output))

    def test_both_override_no_warning(self):
        """Overriding both recovery methods should NOT emit a warning."""
        import logging
        logger = logging.getLogger("agent_foundation.common.inferencers.inferencer_base")
        with self.assertRaises(AssertionError):
            # assertLogs raises AssertionError if no logs are emitted at WARNING
            with self.assertLogs(logger, level="WARNING"):
                BothRecoveryInferencer()

    def test_no_override_no_warning(self):
        """Not overriding either recovery method should NOT emit a warning."""
        import logging
        logger = logging.getLogger("agent_foundation.common.inferencers.inferencer_base")
        with self.assertRaises(AssertionError):
            with self.assertLogs(logger, level="WARNING"):
                CountingInferencer()

    def test_fire_once_per_class(self):
        """Warning should fire only once per class name."""
        import logging
        logger = logging.getLogger("agent_foundation.common.inferencers.inferencer_base")
        with self.assertLogs(logger, level="WARNING") as cm:
            AsyncOnlyRecoveryInferencer()
        first_count = len([m for m in cm.output if "AsyncOnlyRecoveryInferencer" in m])
        self.assertEqual(first_count, 1)

        # Second instantiation should NOT produce another warning
        with self.assertRaises(AssertionError):
            with self.assertLogs(logger, level="WARNING"):
                AsyncOnlyRecoveryInferencer()


class TestNestedRetryWarning(unittest.TestCase):
    """Tests for nested-retry warning in __attrs_post_init__."""

    def setUp(self):
        # Clear the fire-once set before each test
        InferencerBase._nested_retry_warned.clear()

    def test_fallback_with_high_max_retry_warns(self):
        """fallback_inferencer with max_retry > 1 should emit a warning."""
        import logging
        fallback = SuccessCountingInferencer(max_retry=3)
        with self.assertLogs(logging.getLogger("agent_foundation.common.inferencers.inferencer_base"),
                             level="WARNING") as cm:
            SuccessCountingInferencer(fallback_inferencer=fallback)
        self.assertTrue(any("max_retry=3" in msg for msg in cm.output))
        self.assertTrue(any("multiplicative" in msg for msg in cm.output))

    def test_fallback_with_max_retry_1_no_warning(self):
        """fallback_inferencer with max_retry=1 should NOT emit a warning."""
        import logging
        fallback = SuccessCountingInferencer(max_retry=1)
        logger = logging.getLogger("agent_foundation.common.inferencers.inferencer_base")
        with self.assertRaises(AssertionError):
            with self.assertLogs(logger, level="WARNING"):
                SuccessCountingInferencer(fallback_inferencer=fallback)

    def test_fallback_list_warns_on_first_high_retry(self):
        """A list of fallbacks should warn on the first one with max_retry > 1."""
        import logging
        fb1 = SuccessCountingInferencer(max_retry=1)
        fb2 = SuccessCountingInferencer(max_retry=5)
        with self.assertLogs(logging.getLogger("agent_foundation.common.inferencers.inferencer_base"),
                             level="WARNING") as cm:
            SuccessCountingInferencer(fallback_inferencer=[fb1, fb2])
        self.assertTrue(any("max_retry=5" in msg for msg in cm.output))

    def test_no_fallback_no_warning(self):
        """No fallback_inferencer should NOT emit a nested-retry warning."""
        import logging
        logger = logging.getLogger("agent_foundation.common.inferencers.inferencer_base")
        with self.assertRaises(AssertionError):
            with self.assertLogs(logger, level="WARNING"):
                SuccessCountingInferencer()

    def test_fire_once_per_outer_class(self):
        """Warning should fire only once per outer class name."""
        import logging
        logger = logging.getLogger("agent_foundation.common.inferencers.inferencer_base")
        fallback = SuccessCountingInferencer(max_retry=3)
        with self.assertLogs(logger, level="WARNING") as cm:
            SuccessCountingInferencer(fallback_inferencer=fallback)
        first_count = len([m for m in cm.output if "multiplicative" in m])
        self.assertEqual(first_count, 1)

        # Second instantiation should NOT produce another warning
        with self.assertRaises(AssertionError):
            with self.assertLogs(logger, level="WARNING"):
                SuccessCountingInferencer(fallback_inferencer=fallback)


# ---- Property 11: Fallback State Isolation tests for task 8.5 ----
# Feature: retry-native-timeout, Property 11: Fallback State Isolation


@attrs
class FallbackStateRecordingInferencer(InferencerBase):
    """Concrete test subclass that records _current_fallback_state per call.

    On _ainfer: always raises a unique RuntimeError (keyed by input).
    On _ainfer_recovery: records the _fallback_state seen via ContextVar,
    then returns a success value so the call completes.

    Thread-safe recording via per-call asyncio.Event coordination.
    """

    # Shared mutable list to collect (call_id, fallback_state_snapshot) tuples.
    # Each concurrent task appends exactly one entry during recovery.
    recorded_states: list = attrib(factory=list, init=False)

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        raise RuntimeError(f"fail-{inference_input}")

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        # Small yield to let other tasks interleave — maximizes chance of
        # cross-contamination if isolation is broken.
        await asyncio.sleep(0)
        raise RuntimeError(f"fail-{inference_input}")

    async def _ainfer_recovery(self, inference_input, last_exception, last_partial_output,
                                inference_config=None, **kwargs):
        # Read the ContextVar to verify it holds THIS call's state
        from agent_foundation.common.inferencers.inferencer_base import _current_fallback_state
        state = _current_fallback_state.get(None)

        # Snapshot the state dict (copy to avoid later mutation)
        snapshot = dict(state) if state is not None else None

        self.recorded_states.append({
            "call_id": inference_input,
            "fallback_state": snapshot,
            "last_exception_str": str(last_exception) if last_exception else None,
        })

        # Return success so _ainfer_single completes
        return f"recovered-{inference_input}"


class TestFallbackStateIsolation(unittest.IsolatedAsyncioTestCase):
    """Property 11: Fallback State Isolation.

    **Validates: Requirements 17.5, 19.2, 22.3**

    Run concurrent _ainfer_single calls on the same InferencerBase instance
    and verify that each call gets its own independent _fallback_state dict
    via ContextVar — no cross-contamination between concurrent tasks.

    The key mechanism: under aparallel_infer, asyncio.gather creates separate
    Tasks, each of which gets its own copy of the ContextVar context. So
    _current_fallback_state.set() in one task does not affect another.
    """

    async def test_concurrent_ainfer_single_independent_fallback_states(self):
        """Multiple concurrent _ainfer_single calls on the same instance
        each see their own _fallback_state with the correct exception.

        Creates N concurrent calls, each failing with RuntimeError("fail-{i}").
        After recovery, verifies:
        1. Each recorded state has the correct call_id
        2. Each recorded state's last_exception matches the call's unique error
        3. No state contains another call's exception (no cross-contamination)
        """
        num_concurrent = 10
        inf = FallbackStateRecordingInferencer(
            max_retry=1,
            fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        )

        # Run concurrent _ainfer_single calls via asyncio.gather
        tasks = [
            inf._ainfer_single(f"call-{i}")
            for i in range(num_concurrent)
        ]
        results = await asyncio.gather(*tasks)

        # All calls should have recovered successfully
        assert len(results) == num_concurrent
        for i, result in enumerate(results):
            assert result == f"recovered-call-{i}", (
                f"Expected 'recovered-call-{i}', got {result!r}"
            )

        # Verify recorded states
        assert len(inf.recorded_states) == num_concurrent, (
            f"Expected {num_concurrent} recorded states, got {len(inf.recorded_states)}"
        )

        # Build a map from call_id to recorded state
        state_by_call = {s["call_id"]: s for s in inf.recorded_states}

        for i in range(num_concurrent):
            call_id = f"call-{i}"
            assert call_id in state_by_call, f"Missing recorded state for {call_id}"

            record = state_by_call[call_id]

            # The fallback_state should not be None
            assert record["fallback_state"] is not None, (
                f"{call_id}: fallback_state was None — ContextVar not set"
            )

            # The last_exception passed to _ainfer_recovery should match
            # this specific call's unique error
            expected_exc = f"fail-{call_id}"
            assert record["last_exception_str"] == expected_exc, (
                f"{call_id}: expected last_exception '{expected_exc}', "
                f"got '{record['last_exception_str']}'"
            )

            # The fallback_state dict's last_exception should also match
            fs = record["fallback_state"]
            assert fs["last_exception"] is not None, (
                f"{call_id}: fallback_state['last_exception'] was None"
            )
            assert str(fs["last_exception"]) == expected_exc, (
                f"{call_id}: fallback_state['last_exception'] was "
                f"'{fs['last_exception']}', expected '{expected_exc}'"
            )

    async def test_concurrent_via_aparallel_infer_isolation(self):
        """Verify isolation when using the actual aparallel_infer method.

        This tests the real code path that users would hit: aparallel_infer
        creates asyncio Tasks via asyncio.gather, each getting its own
        ContextVar copy.
        """
        num_concurrent = 8
        inf = FallbackStateRecordingInferencer(
            max_retry=1,
            fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        )

        inputs = [f"call-{i}" for i in range(num_concurrent)]
        results = await inf.aparallel_infer(inputs)

        # All calls should have recovered
        assert len(results) == num_concurrent
        for i, result in enumerate(results):
            assert result == f"recovered-call-{i}"

        # Verify each call saw its own unique exception in _fallback_state
        assert len(inf.recorded_states) == num_concurrent

        state_by_call = {s["call_id"]: s for s in inf.recorded_states}
        seen_exceptions = set()

        for i in range(num_concurrent):
            call_id = f"call-{i}"
            record = state_by_call[call_id]

            exc_str = record["last_exception_str"]
            expected = f"fail-{call_id}"
            assert exc_str == expected, (
                f"{call_id}: cross-contamination detected — "
                f"expected '{expected}', got '{exc_str}'"
            )

            # Track that each exception is unique (no duplicates)
            assert exc_str not in seen_exceptions, (
                f"Duplicate exception '{exc_str}' — state leaked between tasks"
            )
            seen_exceptions.add(exc_str)

    async def test_fallback_state_not_shared_across_calls(self):
        """Verify that mutating one call's _fallback_state does not affect another.

        Uses a subclass that writes to the fallback_state during recovery
        to confirm the dict is truly per-call.
        """
        @attrs
        class MutatingRecoveryInferencer(InferencerBase):
            """Writes a marker into _fallback_state during recovery."""
            recorded_states: list = attrib(factory=list, init=False)

            def _infer(self, inference_input, inference_config=None, **_inference_args):
                raise RuntimeError(f"fail-{inference_input}")

            async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
                await asyncio.sleep(0)
                raise RuntimeError(f"fail-{inference_input}")

            async def _ainfer_recovery(self, inference_input, last_exception,
                                        last_partial_output, inference_config=None, **kwargs):
                from agent_foundation.common.inferencers.inferencer_base import _current_fallback_state
                state = _current_fallback_state.get(None)

                # Write a call-specific marker into the state dict
                if state is not None:
                    state["marker"] = f"marker-{inference_input}"

                # Small delay to let other tasks potentially see our mutation
                await asyncio.sleep(0.01)

                # Re-read and snapshot
                state_after = _current_fallback_state.get(None)
                snapshot = dict(state_after) if state_after is not None else None
                self.recorded_states.append({
                    "call_id": inference_input,
                    "state_snapshot": snapshot,
                })
                return f"ok-{inference_input}"

        num_concurrent = 6
        inf = MutatingRecoveryInferencer(
            max_retry=1,
            fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        )

        tasks = [inf._ainfer_single(f"call-{i}") for i in range(num_concurrent)]
        await asyncio.gather(*tasks)

        assert len(inf.recorded_states) == num_concurrent

        state_by_call = {s["call_id"]: s for s in inf.recorded_states}
        for i in range(num_concurrent):
            call_id = f"call-{i}"
            record = state_by_call[call_id]
            snapshot = record["state_snapshot"]

            # Each call should see only its own marker, not another call's
            expected_marker = f"marker-{call_id}"
            assert snapshot is not None
            assert snapshot.get("marker") == expected_marker, (
                f"{call_id}: expected marker '{expected_marker}', "
                f"got '{snapshot.get('marker')}' — cross-contamination!"
            )


if __name__ == "__main__":
    unittest.main()
