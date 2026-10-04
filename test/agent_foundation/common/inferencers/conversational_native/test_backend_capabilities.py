"""Every native backend's capability record is evidence-tagged and names the
vendor versions that evidence was gathered against (plan §8), and a backend
uses a capability only on verified evidence unless the configuration opts in
to experimental ones (invariant 8)."""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

from agent_foundation.common.inferencers.agentic_inferencers import (
    conversational_native,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import (
    _declared_commands,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    NativeConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session import (
    backend as backend_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    BackendCapabilities,
    CallerTools,
    Evidence,
    L2Channel,
    NativeBackendSpec,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.factory import (
    backend_class,
    BACKEND_KINDS,
    make_backend,
    normalize_spec,
)

_METAMATE = (
    "agent_foundation.common.inferencers.agentic_inferencers."
    "conversational_native.session.metamate"
)


# Capability flags whose feature a backend uses whenever the flag is set.
_FLAG_FEATURES = (
    "turn_stop_hook",
    "subagent_attribution",
    "exact_fork",
    "compaction_signal",
)


def _records() -> Iterator[tuple[str, BackendCapabilities]]:
    for kind in BACKEND_KINDS:
        try:
            yield kind, backend_class(kind).capabilities
        except NativeCapabilityError:
            continue  # a backend whose own Buck target is not in this binary


def _with_evidence(kind: str, **evidence: Evidence | None) -> Any:
    """Patch ``kind``'s capability record with ``evidence`` (``None``
    removes a key)."""
    cls = backend_class(kind)
    tagged = dict(cls.capabilities.evidence)
    for feature, value in evidence.items():
        if value is None:
            tagged.pop(feature, None)
        else:
            tagged[feature] = value
    return mock.patch.object(
        cls, "capabilities", dataclasses.replace(cls.capabilities, evidence=tagged)
    )


def _toy() -> BackendCapabilities:
    return BackendCapabilities(
        kind="toy",
        caller_tools=CallerTools.NONE,
        l2_channels=(L2Channel.ENVELOPE,),
        pinned_session_id=False,
        exact_fork=False,
        turn_stop_hook=False,
        subagent_attribution=False,
        compaction_signal=False,
        persistent_process=False,
        environments=("inherit", "hermetic"),
        evidence={"resume": Evidence.EXPERIMENTAL, "hermetic": Evidence.UNSUPPORTED},
        relies_on=("resume",),
    )


class BackendCapabilitiesTest(unittest.TestCase):
    def test_every_backend_records_its_tested_vendor_versions(self) -> None:
        for kind, caps in _records():
            with self.subTest(kind=kind):
                self.assertEqual(caps.kind, kind)
                self.assertTrue(caps.tested_versions)
                for component, version in caps.tested_versions.items():
                    self.assertTrue(component.strip())
                    self.assertTrue(version.strip())

    def test_both_claude_backends_were_tested_on_the_same_claude(self) -> None:
        # The SDK drives the system `claude` binary the CLI backend runs.
        sdk = backend_class("claude_sdk").capabilities.tested_versions
        cli = backend_class("claude_cli").capabilities.tested_versions
        self.assertEqual(sdk["claude"], cli["claude"])

    def test_the_readme_names_every_tested_version(self) -> None:
        readme = Path(conversational_native.__file__).parent / "README.md"
        row = next(
            line
            for line in readme.read_text(encoding="utf-8").splitlines()
            if line.startswith("| tested on |")
        )
        cells = dict(zip(BACKEND_KINDS, [c.strip() for c in row.split("|")[2:-1]]))
        for kind, caps in _records():
            with self.subTest(kind=kind):
                for component, version in caps.tested_versions.items():
                    self.assertIn(version.split()[0], cells[kind], component)

    def test_every_backend_tags_its_evidence(self) -> None:
        for kind, caps in _records():
            with self.subTest(kind=kind):
                self.assertTrue(caps.evidence)
                self.assertTrue(
                    all(isinstance(v, Evidence) for v in caps.evidence.values())
                )
                self.assertIn("inherit", caps.environments)

    def test_every_capability_a_backend_uses_is_evidence_tagged(self) -> None:
        for kind, caps in _records():
            with self.subTest(kind=kind):
                used = set(caps.relies_on)
                self.assertIn("resume", used)  # every backend continues sessions
                for flag in _FLAG_FEATURES:
                    if getattr(caps, flag):
                        self.assertIn(flag, used)
                if caps.caller_tools is not CallerTools.NONE:
                    self.assertIn("caller_tools", used)
                if L2Channel.HOOK in caps.l2_channels:
                    self.assertIn("l2_hook", used)
                self.assertLessEqual(used, set(caps.evidence))
                for environment in caps.environments:
                    if environment != "inherit":
                        self.assertEqual(
                            caps.evidence.get(environment), Evidence.VERIFIED
                        )

    def test_every_backend_constructs_on_verified_evidence_alone(self) -> None:
        for kind, caps in _records():
            for environment in caps.environments:
                with self.subTest(kind=kind, environment=environment):
                    spec = NativeBackendSpec(kind=kind, environment=environment)
                    backend_class(kind)(spec)

    def test_a_backend_refuses_a_capability_it_uses_without_verified_evidence(
        self,
    ) -> None:
        for kind, caps in _records():
            for feature in caps.relies_on:
                for evidence in (Evidence.EXPERIMENTAL, Evidence.UNSUPPORTED, None):
                    with self.subTest(kind=kind, feature=feature, evidence=evidence):
                        with _with_evidence(kind, **{feature: evidence}):
                            with self.assertRaises(NativeCapabilityError) as ctx:
                                backend_class(kind)(NativeBackendSpec(kind=kind))
                        self.assertIn(repr(feature), str(ctx.exception))
                        self.assertIn(kind, str(ctx.exception))

    def test_experimental_resume_needs_the_opt_in(self) -> None:
        with _with_evidence("devmate_dm", resume=Evidence.EXPERIMENTAL):
            with self.assertRaises(NativeCapabilityError) as ctx:
                make_backend(normalize_spec({"kind": "devmate_dm"}))
            self.assertIn("allow_experimental: true", str(ctx.exception))
            spec = normalize_spec({"kind": "devmate_dm", "allow_experimental": True})
            with self.assertLogs(backend_module.logger, "WARNING"):
                make_backend(spec)

    def test_hermetic_is_used_only_where_its_evidence_allows(self) -> None:
        with _with_evidence("claude_sdk", hermetic=Evidence.EXPERIMENTAL):
            with self.assertRaises(NativeCapabilityError):
                make_backend(
                    NativeBackendSpec(kind="claude_sdk", environment="hermetic")
                )
            make_backend(NativeBackendSpec(kind="claude_sdk"))  # inherit unaffected
            make_backend(
                NativeBackendSpec(
                    kind="claude_sdk", environment="hermetic", allow_experimental=True
                )
            )
        with _with_evidence("codex_cli", hermetic=None):
            with self.assertRaises(NativeCapabilityError) as ctx:
                make_backend(
                    NativeBackendSpec(kind="codex_cli", environment="hermetic")
                )
            self.assertIn("missing", str(ctx.exception))

    def test_unsupported_evidence_is_refused_even_with_the_opt_in(self) -> None:
        caps = _toy()
        caps.require_spec(NativeBackendSpec(allow_experimental=True))
        with self.assertRaises(NativeCapabilityError) as ctx:
            caps.require_spec(
                NativeBackendSpec(environment="hermetic", allow_experimental=True)
            )
        self.assertIn("'hermetic': its evidence is unsupported", str(ctx.exception))
        self.assertNotIn("allow_experimental", str(ctx.exception))
        with self.assertRaises(NativeCapabilityError):
            caps.require_spec(NativeBackendSpec(environment="sandboxed"))

    def test_claude_passes_its_local_commands_through(self) -> None:
        # Claude Code runs them without UserPromptSubmit and without a model
        # call (claude 2.1.289), so an L2 sent with them would be wasted.
        for kind in ("claude_sdk", "claude_cli"):
            with self.subTest(kind=kind):
                self.assertEqual(
                    set(backend_class(kind).capabilities.slash_passthrough),
                    {"compact", "context", "cost", "usage"},
                )

    def test_no_passed_through_command_is_one_of_afs_own(self) -> None:
        # AF's commands are handled first; a passthrough entry naming one of
        # them (e.g. `help`) would never reach the vendor.
        own = {
            key
            for _, meta, _ in _declared_commands(NativeConversationalInferencer.__mro__)
            for key in (meta.name, *meta.aliases)
        }
        self.assertIn("help", own)
        for kind, caps in _records():
            with self.subTest(kind=kind):
                self.assertFalse(set(caps.slash_passthrough) & own)

    def test_every_backend_declares_its_lane_routes(self) -> None:
        for kind, caps in _records():
            with self.subTest(kind=kind):
                self.assertTrue(caps.l1_route.strip())
                for channel in caps.l2_channels:
                    routes = caps.prompt_routes(channel)
                    self.assertEqual(routes.l1, caps.l1_route)
                    for route in (routes.l2, routes.l3, routes.user):
                        # Each labels a one-line "View Prompt" header.
                        self.assertTrue(route.strip())
                        self.assertNotIn("\n", route)

    def test_a_backend_left_out_of_the_binary_is_a_capability_error(self) -> None:
        with mock.patch.dict(sys.modules, {_METAMATE: None}):
            with self.assertRaises(NativeCapabilityError) as ctx:
                backend_class("metamate")
        self.assertIn("Buck target", str(ctx.exception))
