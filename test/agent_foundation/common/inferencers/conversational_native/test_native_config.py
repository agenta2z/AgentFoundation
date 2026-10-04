"""Registration and YAML construction of the native orchestrator:
``ConversationalNative`` alias, ``configs/conversational_native`` and
``_ci_host.build_native_from_config``, and ``NativeBackendSpec.from_inferencer``.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import agent_foundation
import later.unittest
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeConversationalInferencer,
    NativeRuntimeManager,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session import (
    backend as backend_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    Evidence,
    L2Channel,
    NativeBackendSpec,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.factory import (
    backend_class,
)
from agent_foundation.common.workflow.sop_state import SOPState
from agent_foundation.resources.tools._ci_host import build_native_from_config
from fakes import FAKE_CAPABILITIES, FakeBackendFactory
from helpers import (
    FIXTURE_SOPS,
    make_native,
    RecordingExecutor,
    shared_dir,
    tool_registry,
)

CONFIG = (
    Path(agent_foundation.__file__).parent
    / "resources"
    / "configs"
    / "conversational_native"
    / "default.yaml"
)


class BuildNativeFromConfigTest(later.unittest.TestCase):
    async def _build(self, backend: str, **kwargs) -> NativeConversationalInferencer:
        runtime = NativeRuntimeManager()
        store = InMemoryRecordStore()
        native = build_native_from_config(
            CONFIG,
            backend=backend,
            backend_overrides={"cwd": tempfile.mkdtemp()},
            tool_registry=tool_registry(),
            tool_executor=RecordingExecutor(),
            runtime_manager=runtime,
            record_store=store,
            conversation_key="conv-cfg",
            **kwargs,
        )
        self.addAsyncCleanup(runtime.aclose_all)
        self.assertIs(native.runtime_manager, runtime)
        self.assertIs(native.record_store, store)
        return native

    async def test_each_tool_capable_backend_builds(self) -> None:
        for kind in ("claude_sdk", "claude_cli", "devmate_dm", "codex_cli"):
            native = await self._build(kind)
            self.assertIsInstance(native, NativeConversationalInferencer)
            self.assertEqual(native.backend.kind, kind)
            self.assertEqual(native.conversation_key, "conv-cfg")

    async def test_yaml_policies_and_host_overrides_apply(self) -> None:
        native = await self._build("claude_sdk", rewind_on_repeat_turn=True)
        self.assertEqual(native.on_l1_drift, "notice")
        self.assertEqual(native.on_session_loss, "recap")
        self.assertEqual(native.native_tool_result_max_chars, 16000)
        self.assertTrue(native.rewind_on_repeat_turn)  # host override wins
        self.assertEqual(native.backend.model, "opus[1m]")

    async def test_claude_backends_refuse_a_result_size_above_claude_codes_cap(
        self,
    ) -> None:
        for kind in ("claude_sdk", "claude_cli"):
            with self.subTest(kind):
                with self.assertRaisesRegex(ValueError, "at most 500,000"):
                    await self._build(kind, native_tool_result_max_chars=500_001)
                native = await self._build(kind, native_tool_result_max_chars=500_000)
                self.assertEqual(native.native_tool_result_max_chars, 500_000)

    async def test_other_backends_take_any_result_size(self) -> None:
        for kind in ("devmate_dm", "codex_cli"):
            with self.subTest(kind):
                native = await self._build(kind, native_tool_result_max_chars=900_000)
                self.assertEqual(native.native_tool_result_max_chars, 900_000)

    async def test_bridge_surface_knobs_come_from_the_yaml(self) -> None:
        native = await self._build("claude_sdk")
        self.assertEqual(
            native.sop_control_tools,
            ["enter_sop", "resume_sop", "pause_sop", "exit_sop", "sop_status"],
        )
        self.assertFalse(native.expose_tool_argument_form)
        native = await self._build("claude_sdk", sop_control_tools=[])
        self.assertEqual(native.sop_control_tools, [])
        self.assertFalse(
            {"enter_sop", "sop_status"} & {s.name for s in native.bridge.manifest()}
        )

    async def test_envelope_backends_opt_in_explicitly(self) -> None:
        for kind in ("devmate_dm", "codex_cli"):
            native = await self._build(kind)
            self.assertTrue(native.backend.l2_envelope_allowed)

    async def test_unknown_backend_lists_available(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_native_from_config(CONFIG, backend="nope")
        self.assertIn("claude_sdk", str(ctx.exception))

    async def test_registered_alias_resolves_to_the_native_class(self) -> None:
        import agent_foundation.common.configs.registered_targets  # noqa: F401
        from omegaconf import OmegaConf
        from rich_python_utils.config_utils import instantiate

        runtime = NativeRuntimeManager()
        self.addAsyncCleanup(runtime.aclose_all)
        native = instantiate(
            OmegaConf.create(
                {
                    "_target_": "ConversationalNative",
                    "_partial_": True,
                    "backend": {"kind": "claude_sdk", "cwd": tempfile.mkdtemp()},
                }
            )
        )(runtime_manager=runtime)
        self.assertIsInstance(native, NativeConversationalInferencer)


class FromInferencerTest(unittest.TestCase):
    def test_config_mapping(self) -> None:
        spec = NativeBackendSpec.from_inferencer(
            {
                "_target_": "ClaudeCodeCLI",
                "model_name": "opus[1m]",
                "permission_mode": "bypassPermissions",
                "target_path": "/repo",
                "idle_timeout_seconds": 1800,
            }
        )
        self.assertEqual(spec.kind, "claude_cli")
        self.assertEqual(spec.model, "opus[1m]")
        self.assertEqual(spec.cwd, "/repo")

    def test_live_instance_and_overrides(self) -> None:
        class CodexCliInferencer:
            model_name = "gpt-5"
            target_path = "/w"

        definition = CodexCliInferencer()
        spec = NativeBackendSpec.from_inferencer(definition, l2_envelope_allowed=True)
        self.assertEqual(
            (spec.kind, spec.model, spec.cwd), ("codex_cli", "gpt-5", "/w")
        )
        self.assertTrue(spec.l2_envelope_allowed)
        self.assertEqual(definition.model_name, "gpt-5")  # read, never mutated

    def test_dotted_target_and_unknown(self) -> None:
        spec = NativeBackendSpec.from_inferencer(
            {
                "_target_": "agent_foundation.x.metamate_sdk_inferencer.MetamateSDKInferencer"
            }
        )
        self.assertEqual(spec.kind, "metamate")
        with self.assertRaises(ValueError):
            NativeBackendSpec.from_inferencer({"_target_": "RovoDevCLI"})


class ConstructionValidationTest(unittest.TestCase):
    def test_unknown_backend_kind_fails_at_construction(self) -> None:
        with self.assertRaises(NativeCapabilityError):
            NativeConversationalInferencer(backend={"kind": "nope"})

    def test_envelope_only_backend_requires_explicit_opt_in(self) -> None:
        caps = dataclasses.replace(FAKE_CAPABILITIES, l2_channels=(L2Channel.ENVELOPE,))
        with self.assertRaises(NativeCapabilityError):
            make_native([], factory=FakeBackendFactory(capabilities=caps))
        shared = shared_dir()
        native, _, _, _ = make_native(
            [],
            factory=FakeBackendFactory(capabilities=caps),
            backend={"kind": "claude_sdk", "cwd": shared, "l2_envelope_allowed": True},
        )
        self.assertTrue(native.backend.l2_envelope_allowed)

    def test_environment_the_backend_cannot_honour_fails_at_construction(
        self,
    ) -> None:
        shared = shared_dir()
        with self.assertRaises(NativeCapabilityError) as ctx:
            NativeConversationalInferencer(
                backend={
                    "kind": "devmate_dm",
                    "cwd": shared,
                    "environment": "hermetic",
                    "l2_envelope_allowed": True,
                },
                prior_context={"native_session_dir": shared},
            )
        self.assertIn("environment 'hermetic'", str(ctx.exception))
        factory = FakeBackendFactory()
        with self.assertRaises(NativeCapabilityError) as ctx:
            make_native(
                [],
                factory=factory,
                backend={
                    "kind": "claude_sdk",
                    "cwd": shared,
                    "environment": "hermetic",
                },
            )
        self.assertIn("hermetic", str(ctx.exception))
        self.assertEqual(factory.instances, [])
        hermetic = dataclasses.replace(
            FAKE_CAPABILITIES, environments=("inherit", "hermetic")
        )
        # A listed environment is used only on its evidence.
        with self.assertRaises(NativeCapabilityError) as ctx:
            make_native(
                [],
                factory=FakeBackendFactory(capabilities=hermetic),
                backend={
                    "kind": "claude_sdk",
                    "cwd": shared,
                    "environment": "hermetic",
                },
            )
        self.assertIn("'hermetic': its evidence is missing", str(ctx.exception))
        native, _, _, _ = make_native(
            [],
            factory=FakeBackendFactory(
                capabilities=dataclasses.replace(
                    hermetic, evidence={"hermetic": Evidence.VERIFIED}
                )
            ),
            backend={"kind": "claude_sdk", "cwd": shared, "environment": "hermetic"},
        )
        self.assertEqual(native.backend.environment, "hermetic")


def _relying_on_resume(evidence: Optional[Evidence]) -> FakeBackendFactory:
    tagged = {} if evidence is None else {"resume": evidence}
    return FakeBackendFactory(
        capabilities=dataclasses.replace(
            FAKE_CAPABILITIES, evidence=tagged, relies_on=("resume",)
        )
    )


class EvidenceGateAtConstructionTest(unittest.TestCase):
    """Invariant 8: a capability the backend relies on is used only on
    verified evidence (experimental with the opt-in), and the refusal comes
    from the inferencer's constructor — before any backend or session."""

    def test_verified_evidence_constructs(self) -> None:
        factory = _relying_on_resume(Evidence.VERIFIED)
        native, _, _, _ = make_native([], factory=factory)
        self.assertFalse(native.backend.allow_experimental)
        self.assertEqual(factory.instances, [])

    def test_experimental_evidence_fails_at_construction(self) -> None:
        factory = _relying_on_resume(Evidence.EXPERIMENTAL)
        with self.assertRaises(NativeCapabilityError) as ctx:
            make_native([], factory=factory)
        self.assertIn("'resume': its evidence is experimental", str(ctx.exception))
        self.assertIn("allow_experimental: true", str(ctx.exception))
        self.assertEqual(factory.instances, [])

    def test_allow_experimental_warns_and_constructs(self) -> None:
        factory = _relying_on_resume(Evidence.EXPERIMENTAL)
        shared = shared_dir()
        with self.assertLogs(backend_module.logger, "WARNING") as logs:
            native, _, _, _ = make_native(
                [],
                factory=factory,
                backend={
                    "kind": "claude_sdk",
                    "cwd": shared,
                    "allow_experimental": True,
                },
            )
        self.assertIn("'resume'", logs.output[0])
        self.assertTrue(native.backend.allow_experimental)
        self.assertEqual(factory.instances, [])

    def test_unsupported_or_missing_evidence_fails_even_with_the_opt_in(
        self,
    ) -> None:
        for evidence in (Evidence.UNSUPPORTED, None):
            with self.subTest(evidence=evidence):
                factory = _relying_on_resume(evidence)
                with self.assertRaises(NativeCapabilityError) as ctx:
                    make_native(
                        [],
                        factory=factory,
                        backend={
                            "kind": "claude_sdk",
                            "cwd": shared_dir(),
                            "allow_experimental": True,
                        },
                    )
                self.assertIn("'resume'", str(ctx.exception))
                self.assertEqual(factory.instances, [])

    def test_a_real_backend_is_gated_without_a_factory(self) -> None:
        cls = backend_class("devmate_dm")
        caps = cls.capabilities
        demoted = dict(caps.evidence, resume=Evidence.EXPERIMENTAL)
        shared = shared_dir()
        backend = {"kind": "devmate_dm", "cwd": shared, "l2_envelope_allowed": True}
        with mock.patch.object(
            cls, "capabilities", dataclasses.replace(caps, evidence=demoted)
        ):
            with self.assertRaises(NativeCapabilityError) as ctx:
                NativeConversationalInferencer(
                    backend=backend, prior_context={"native_session_dir": shared}
                )
            self.assertIn("devmate_dm", str(ctx.exception))
            with self.assertLogs(backend_module.logger, "WARNING"):
                NativeConversationalInferencer(
                    backend={**backend, "allow_experimental": True},
                    prior_context={"native_session_dir": shared},
                )


class ExtraSopDirsTest(unittest.TestCase):
    """The SOP controller owns the extra SOP directories; the constructor
    argument only seeds it (as for the text-protocol orchestrator)."""

    def _assert_controller_finds_the_fixture_sop(self, native) -> None:
        state = SOPState(sop_name="mini_research")
        native.sop_controller.reload_sop_definition(state)
        self.assertIsNotNone(state.sop)
        self.assertEqual(state.tool_phase_map.get("write_brief"), "1")

    def test_constructor_seeds_a_list_the_controller_owns(self) -> None:
        shared = shared_dir()
        dirs = [FIXTURE_SOPS]
        native = NativeConversationalInferencer(
            backend={"kind": "claude_sdk", "cwd": shared},
            backend_factory=FakeBackendFactory(),
            prior_context={"native_session_dir": shared},
            extra_sop_dirs=dirs,
        )
        dirs.clear()
        self.assertEqual(native.sop_controller.extra_sop_dirs, [FIXTURE_SOPS])
        self.assertEqual(native.extra_sop_dirs, [FIXTURE_SOPS])
        self.assertIs(native._extra_sop_dirs, native.sop_controller.extra_sop_dirs)
        self._assert_controller_finds_the_fixture_sop(native)

    def test_assigned_dirs_reach_the_controller_under_either_name(self) -> None:
        native, _, _, _ = make_native([])
        native.extra_sop_dirs = []
        self.assertEqual(native._extra_sop_dirs, [])
        native._extra_sop_dirs = [FIXTURE_SOPS]
        self.assertEqual(native.sop_controller.extra_sop_dirs, [FIXTURE_SOPS])
        self._assert_controller_finds_the_fixture_sop(native)
