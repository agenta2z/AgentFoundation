# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for the per-inferencer prompt-variable expansion framework (Primitive B).

``InferencerBase`` auto-discovers each concrete class's own
``prompt_templates/_variables/`` tree (MRO-derived, most-derived first) and
``TemplatedInferencerBase._rendering_manager()`` applies it -- lazily, memoized,
and leak-free -- as an immutable ``TemplateManager.with_variable_extensions``
fork gated by a master switch plus per-key overrides. These tests pin:

* discovery: adjacent folder / no folder / missing ``_variables`` subdir /
  unresolvable module dir / MRO order / dedup, the file -> dotted-key mapping,
  the real module-dir resolver, and the skip-keys helper;
* the lazy-memoized derivation (never eager in ``__attrs_post_init__``);
* the master switch (off -> base manager, byte-identical) and per-key disable;
* leak-freeness: two inferencers sharing one base ``TemplateManager`` never
  perturb each other or mutate the shared base;
* the metamate ``notes.large_file_writing`` by-product -- a scoped override that
  replaces the ``has_local_access``-gated base with artifact-delivery guidance.

Tokens are dotted (``{{ notes.greeting }}``) to mirror the real metamate
consumers; the sibling RPU suite covers the underscore token form.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from importlib import resources
from pathlib import Path
from typing import Any

from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrs
from rich_python_utils.string_utils.formatting.jinja2_format import format_template
from rich_python_utils.string_utils.formatting.template_manager import (
    TemplateManager,
    TemplateRootPriority,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _var_path(root: Path, key: str) -> Path:
    """On-disk file backing a dotted variable key under ``<root>/_variables``."""
    return root / "_variables" / f"{key.replace('.', '/')}.jinja2"


def _base_manager(base: Path, root_space: str | None = None) -> TemplateManager:
    """A manager mirroring the real AF config -- ``predefined_variables=False``
    and ``enable_templated_feed=True`` -- so render probes exercise the same
    Pass-1 resolution + Pass-2 jinja render that production uses."""
    return TemplateManager(
        templates=str(base),
        template_formatter=format_template,
        predefined_variables=False,
        enable_templated_feed=True,
        active_template_root_space=root_space,
        active_template_type="main",
    )


def _write_base_tree(base: Path, template: str, variables: dict[str, str]) -> None:
    _write(base / "main" / "default.jinja2", template)
    for key, value in variables.items():
        _write(_var_path(base, key), value)


def _write_extension(module_dir: Path, variables: dict[str, str]) -> Path:
    """Lay a ``prompt_templates/_variables`` tree next to a module and return the
    ``prompt_templates`` root discovery finds adjacent to that module."""
    prompt_templates = module_dir / "prompt_templates"
    for key, value in variables.items():
        _write(_var_path(prompt_templates, key), value)
    return prompt_templates


def _agent_foundation_dir() -> Path | None:
    """The installed ``agent_foundation`` package dir, or None if unresolvable
    (e.g. a packaging runtime that does not materialize the resource tree)."""
    try:
        candidate = Path(str(resources.files("agent_foundation")))
    except (TypeError, OSError, ModuleNotFoundError):
        return None
    return candidate if candidate.is_dir() else None


@attrs
class _StubTemplatedInferencer(TemplatedInferencerBase):
    """Minimal concrete leaf: ``_infer`` is the base's only abstract method."""

    def _infer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        **_inference_args: Any,
    ) -> Any:
        raise NotImplementedError


def _discoverable_inferencer(
    module_dir: Path | None, base: type = _StubTemplatedInferencer
) -> type:
    """A leaf whose ``_module_dir_for_class`` maps each class to an injected
    directory, so the real ``_discover_inferencer_variable_roots`` runs unchanged
    while on-disk module adjacency (the separately-trivial ``importlib``/
    ``inspect`` lookup) is bypassed -- isolating the MRO-walk / folder-check /
    dedup logic. Chaining ``base=`` builds a multi-level MRO for the order test."""

    @attrs
    class _Discoverable(base):
        _test_module_dir = module_dir

        @staticmethod
        def _module_dir_for_class(klass: type) -> Path | None:
            return getattr(klass, "_test_module_dir", None)

    return _Discoverable


def _recovery_inferencer(module_dir: Path, recovery_dir: Path) -> type:
    """A discoverable leaf that mirrors ``StreamingInferencerBase``: it appends a
    recovery template root in ``__attrs_post_init__`` (after ``super()``), so the
    root exists only *after* construction. The lazy ``_rendering_manager``
    derivation must therefore capture it -- an eager fork built during a post-init
    would snapshot the manager before the recovery root was added."""

    @attrs
    class _Recovery(_StubTemplatedInferencer):
        _test_module_dir = module_dir

        @staticmethod
        def _module_dir_for_class(klass: type) -> Path | None:
            return getattr(klass, "_test_module_dir", None)

        def __attrs_post_init__(self) -> None:
            super().__attrs_post_init__()
            self.template_manager.add_template_root(
                str(recovery_dir), priority=TemplateRootPriority.LOWEST
            )

    return _Recovery


class InferencerVariableDiscoveryTest(unittest.TestCase):
    """MRO-aware discovery of each class's ``prompt_templates/_variables`` root."""

    def test_adjacent_variables_folder_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module_dir = Path(tmp)
            root = _write_extension(module_dir, {"notes.greeting": "EXT"})
            cls = _discoverable_inferencer(module_dir)
            self.assertEqual(
                cls._discover_inferencer_variable_roots(), [root.resolve()]
            )

    def test_module_without_prompt_templates_discovers_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cls = _discoverable_inferencer(Path(tmp))
            self.assertEqual(cls._discover_inferencer_variable_roots(), [])

    def test_prompt_templates_without_variables_subdir_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "prompt_templates").mkdir()
            cls = _discoverable_inferencer(Path(tmp))
            self.assertEqual(cls._discover_inferencer_variable_roots(), [])

    def test_unresolvable_module_dir_discovers_nothing(self) -> None:
        cls = _discoverable_inferencer(None)
        self.assertEqual(cls._discover_inferencer_variable_roots(), [])

    def test_mro_orders_derived_root_before_base_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_mod = Path(tmp) / "base_mod"
            derived_mod = Path(tmp) / "derived_mod"
            base_root = _write_extension(base_mod, {"notes.greeting": "B"})
            derived_root = _write_extension(derived_mod, {"notes.greeting": "D"})
            base_cls = _discoverable_inferencer(base_mod)
            derived_cls = _discoverable_inferencer(derived_mod, base=base_cls)
            self.assertEqual(
                derived_cls._discover_inferencer_variable_roots(),
                [derived_root.resolve(), base_root.resolve()],
            )

    def test_duplicate_module_dirs_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            module_dir = Path(tmp)
            root = _write_extension(module_dir, {"notes.greeting": "X"})
            base_cls = _discoverable_inferencer(module_dir)
            derived_cls = _discoverable_inferencer(module_dir, base=base_cls)
            self.assertEqual(
                derived_cls._discover_inferencer_variable_roots(), [root.resolve()]
            )

    def test_variable_keys_map_files_to_dotted_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_extension(
                Path(tmp),
                {
                    "notes.greeting": "g",
                    "notes.large_file_writing": "l",
                    "instructions.foo": "f",
                },
            )
            keys = _StubTemplatedInferencer._discover_inferencer_variable_keys([root])
            self.assertEqual(
                keys,
                {"notes.greeting", "notes.large_file_writing", "instructions.foo"},
            )

    def test_real_module_dir_resolver_returns_the_defining_directory(self) -> None:
        resolved = _StubTemplatedInferencer._module_dir_for_class(
            _StubTemplatedInferencer
        )
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.is_dir())


class InferencerVariableSkipKeysTest(unittest.TestCase):
    """Skip keys are exactly the overrides mapped to ``False``."""

    def test_only_false_valued_overrides_become_skip_keys(self) -> None:
        inf = _StubTemplatedInferencer(
            inferencer_variable_overrides={
                "notes.greeting": False,
                "notes.farewell": True,
                "notes.large_file_writing": False,
            }
        )
        self.assertEqual(
            inf._inferencer_variable_skip_keys(),
            {"notes.greeting", "notes.large_file_writing"},
        )


class InferencerRenderingManagerTest(unittest.TestCase):
    """``_rendering_manager()`` derivation: lazy, memoized, switchable."""

    def test_derivation_is_lazy_and_memoized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_base_tree(base, "{{ notes.greeting }}", {"notes.greeting": "BASE"})
            tm = _base_manager(base)
            inf = _discoverable_inferencer(None)(template_manager=tm)
            # Not derived at construction; derived on first use, then memoized.
            self.assertIsNone(inf._extension_manager_cache)
            first = inf._rendering_manager()
            self.assertIsNotNone(inf._extension_manager_cache)
            self.assertIs(first, tm)
            self.assertIs(inf._rendering_manager(), first)

    def test_master_switch_off_returns_base_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            ext = Path(tmp) / "ext"
            _write_base_tree(base, "{{ notes.greeting }}", {"notes.greeting": "BASE"})
            _write_extension(ext, {"notes.greeting": "EXT"})
            tm = _base_manager(base)
            inf = _discoverable_inferencer(ext)(
                template_manager=tm, enable_inferencer_variable_expansion=False
            )
            self.assertIs(inf._rendering_manager(), tm)
            self.assertEqual(inf._rendering_manager()().strip(), "BASE")

    def test_master_switch_on_applies_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            ext = Path(tmp) / "ext"
            _write_base_tree(base, "{{ notes.greeting }}", {"notes.greeting": "BASE"})
            ext_root = _write_extension(ext, {"notes.greeting": "EXT"})
            tm = _base_manager(base)
            inf = _discoverable_inferencer(ext)(template_manager=tm)
            manager = inf._rendering_manager()
            self.assertIsNot(manager, tm)
            self.assertEqual(manager._variable_extension_roots, [ext_root.resolve()])
            self.assertEqual(manager().strip(), "EXT")

    def test_per_key_override_disables_only_that_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            ext = Path(tmp) / "ext"
            _write_base_tree(
                base,
                "{{ notes.greeting }}\n{{ notes.farewell }}",
                {"notes.greeting": "BASE_G", "notes.farewell": "BASE_F"},
            )
            _write_extension(
                ext, {"notes.greeting": "EXT_G", "notes.farewell": "EXT_F"}
            )
            tm = _base_manager(base)
            inf = _discoverable_inferencer(ext)(
                template_manager=tm,
                inferencer_variable_overrides={"notes.greeting": False},
            )
            manager = inf._rendering_manager()
            self.assertEqual(
                manager._variable_extension_disabled_keys,
                frozenset({"notes.greeting"}),
            )
            self.assertEqual(manager().strip(), "BASE_G\nEXT_F")

    def test_unknown_override_key_logs_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            ext = Path(tmp) / "ext"
            _write_base_tree(base, "{{ notes.greeting }}", {"notes.greeting": "BASE"})
            _write_extension(ext, {"notes.greeting": "EXT"})
            tm = _base_manager(base)
            inf = _discoverable_inferencer(ext)(
                template_manager=tm,
                inferencer_variable_overrides={"notes.nonexistent": False},
            )
            with self.assertLogs(level="WARNING") as captured:
                inf._rendering_manager()
            self.assertTrue(
                any("notes.nonexistent" in line for line in captured.output)
            )


class InferencerStreamingPostInitTimingTest(unittest.TestCase):
    """The lazy derivation captures a root added after construction.

    A ``StreamingInferencerBase`` appends a recovery template root in
    ``__attrs_post_init__``; because ``_rendering_manager`` derives the fork at
    first render (never eagerly in a post-init), the fork's rebuilt file space
    sees that post-init root and resolves its variables -- alongside the
    inferencer's own extension override."""

    def test_recovery_root_added_in_post_init_is_seen_by_lazy_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            ext = Path(tmp) / "ext"
            recovery = Path(tmp) / "recovery"
            _write_base_tree(
                base, "{{ notes.greeting }}", {"notes.greeting": "BASE_GREETING"}
            )
            _write_extension(ext, {"notes.greeting": "EXT_GREETING"})
            # A directory source (with a template file) so add_template_root
            # appends it to the root list -- a single-file source returns early
            # without appending. Its _variables tree is what the fork must see.
            _write(recovery / "main" / "default.jinja2", "RECOVERY_TEMPLATE")
            _write(_var_path(recovery, "notes.recovery_note"), "RECOVERY_NOTE")
            tm = _base_manager(base, root_space="")
            inf = _recovery_inferencer(ext, recovery)(template_manager=tm)
            # Lazy: the fork is not built during the recovery-appending post-init.
            self.assertIsNone(inf._extension_manager_cache)
            resolved = inf._rendering_manager().load_variables(
                {"notes.greeting": None, "notes.recovery_note": None},
                root_space="",
            )
            # The inferencer's own extension wins for its key ...
            self.assertEqual(resolved["notes"]["greeting"], "EXT_GREETING")
            # ... and the post-init recovery root's variable is visible too.
            self.assertEqual(resolved["notes"]["recovery_note"], "RECOVERY_NOTE")


class InferencerVariableLeakFreenessTest(unittest.TestCase):
    """Extensions live on per-inferencer forks; the shared base is never mutated."""

    def test_extensions_do_not_leak_across_inferencers_sharing_a_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            ext_a = Path(tmp) / "ext_a"
            ext_b = Path(tmp) / "ext_b"
            _write_base_tree(base, "{{ notes.greeting }}", {"notes.greeting": "BASE"})
            _write_extension(ext_a, {"notes.greeting": "EXT_A"})
            _write_extension(ext_b, {"notes.greeting": "EXT_B"})
            shared = _base_manager(base)
            original_file_space = shared._file_space
            original_loaders = shared._variable_loaders_by_root
            inf_a = _discoverable_inferencer(ext_a)(template_manager=shared)
            inf_b = _discoverable_inferencer(ext_b)(template_manager=shared)
            self.assertEqual(inf_a._rendering_manager()().strip(), "EXT_A")
            self.assertEqual(inf_b._rendering_manager()().strip(), "EXT_B")
            self.assertEqual(shared().strip(), "BASE")
            self.assertIs(shared._file_space, original_file_space)
            self.assertIs(shared._variable_loaders_by_root, original_loaders)


class InferencerPass2ResolutionTest(unittest.TestCase):
    """Pass-2 (``_resolve_templated_feed``) through an extension-aware fork.

    Two Pass-2 branches are pinned: a super-composed value whose base carries a
    ``{% if %}`` guard is re-rendered from the Pass-1 subval (so ``__super__``,
    never a feed key, is not lost to a raw re-read); and a plain override whose
    only variable is a feed key is re-read from the *extension* file, proving the
    rebuilt file space is extension-first on the raw-re-read branch too."""

    def test_super_composed_guard_evaluates_at_pass2_without_losing_super(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            ext = Path(tmp) / "ext"
            _write_base_tree(
                base,
                "{{ notes.guidance }}",
                {"notes.guidance": "{% if show %}BASE_BODY{% endif %}"},
            )
            ext_root = _write_extension(
                ext, {"notes.guidance": "{{ __super__ }}\nDELTA_LINE"}
            )
            derived = _base_manager(base, root_space="").with_variable_extensions(
                [ext_root]
            )
            pass1 = derived.load_variables({"notes.guidance": None}, root_space="")
            composed = pass1["notes"]["guidance"]
            # Pass-1: super pulled the (still-guarded) base body in; token consumed.
            self.assertNotIn("__super__", composed)
            self.assertIn("{% if show %}", composed)
            self.assertIn("DELTA_LINE", composed)
            # Pass-2, guard true: the raw ext file holds a literal {{ __super__ }}
            # (not a feed key), so Pass-2 re-renders the Pass-1 subval -- keeping
            # the composed base body -- rather than re-reading the raw file.
            # deepcopy isolates each call: Step 1 mutates the nested feed in place.
            feed_on = copy.deepcopy(pass1)
            feed_on["show"] = True
            resolved_on = derived._resolve_templated_feed(feed_on, "", "main")
            self.assertIn("BASE_BODY", resolved_on["notes"]["guidance"])
            self.assertIn("DELTA_LINE", resolved_on["notes"]["guidance"])
            # Pass-2, guard false: base gated out, the extension delta survives.
            feed_off = copy.deepcopy(pass1)
            feed_off["show"] = False
            resolved_off = derived._resolve_templated_feed(feed_off, "", "main")
            self.assertNotIn("BASE_BODY", resolved_off["notes"]["guidance"])
            self.assertIn("DELTA_LINE", resolved_off["notes"]["guidance"])

    def test_plain_feed_key_override_is_reread_from_extension_at_pass2(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            ext = Path(tmp) / "ext"
            _write_base_tree(
                base, "{{ notes.status }}", {"notes.status": "BASE:{{ run_id }}"}
            )
            ext_root = _write_extension(ext, {"notes.status": "EXT:{{ run_id }}"})
            base_mgr = _base_manager(base, root_space="")
            derived = base_mgr.with_variable_extensions([ext_root])
            # The override's only var is a feed key (run_id), so Pass-2 takes the
            # raw-re-read branch -- which must read the EXTENSION file, not base.
            pass1 = derived.load_variables({"notes.status": None}, root_space="")
            feed = copy.deepcopy(pass1)
            feed["run_id"] = "R1"
            resolved = derived._resolve_templated_feed(feed, "", "main")
            self.assertEqual(resolved["notes"]["status"], "EXT:R1")
            # Base-only control: the same branch re-reads the base file.
            base_pass1 = base_mgr.load_variables({"notes.status": None}, root_space="")
            base_feed = copy.deepcopy(base_pass1)
            base_feed["run_id"] = "R1"
            base_resolved = base_mgr._resolve_templated_feed(base_feed, "", "main")
            self.assertEqual(base_resolved["notes"]["status"], "BASE:R1")


class MetamateLargeFileWritingByproductTest(unittest.TestCase):
    """The metamate ``notes.large_file_writing`` scoped override (the by-product).

    Resolves the real override through the AF discovery + ``_rendering_manager``
    path via the explicit ``load_variables`` subsystem (what ``_build_template_feed``
    uses), renders it Pass-2 with the metamate feed, and embeds it into each of the
    four real consumer templates end-to-end -- asserting the delivery guidance
    lands exactly once, the ``has_local_access``-gated local-file guidance is
    absent, and no jinja tokens survive. A non-consumer template and a base sibling
    key are pinned unaffected, so the override is scoped to exactly its own key.
    """

    def _real_roots(self) -> tuple[Path, Path]:
        af_dir = _agent_foundation_dir()
        if af_dir is None:
            self.skipTest("agent_foundation package directory is not resolvable")
        base_root = af_dir / "resources" / "prompt_templates"
        metamate_root = (
            af_dir
            / "common"
            / "inferencers"
            / "agentic_inferencers"
            / "external"
            / "metamate"
            / "prompt_templates"
        )
        base_file = _var_path(base_root, "notes.large_file_writing")
        override_file = _var_path(metamate_root, "notes.large_file_writing")
        if not base_file.is_file() or not override_file.is_file():
            self.skipTest(
                "prompt_templates .jinja2 resources are not packaged in this runtime"
            )
        return base_root, metamate_root

    def test_metamate_inferencer_discovers_only_its_own_root(self) -> None:
        _, metamate_root = self._real_roots()
        cls = _discoverable_inferencer(metamate_root.parent)
        self.assertEqual(
            cls._discover_inferencer_variable_roots(), [metamate_root.resolve()]
        )

    def test_override_replaces_gated_base_with_delivery_guidance(self) -> None:
        base_root, metamate_root = self._real_roots()
        inf = _discoverable_inferencer(metamate_root.parent)(
            template_manager=_base_manager(base_root)
        )
        base_only = _base_manager(base_root).load_variables(
            {"notes.large_file_writing": None}, root_space=""
        )["notes"]["large_file_writing"]
        overridden = inf._rendering_manager().load_variables(
            {"notes.large_file_writing": None}, root_space=""
        )["notes"]["large_file_writing"]
        # Pass-1: metamate ships a scoped override (no __super__ compose), so the
        # local-file-writing base text is replaced rather than composed onto.
        self.assertIn("delivering large artifacts", overridden)
        self.assertNotIn("writing large files", overridden)
        self.assertIn("writing large files", base_only)
        # Pass-2 with the metamate feed (has_local_access=False): the base is
        # fully gated -> empty; the override has no guard -> full delivery text.
        feed = {"has_local_access": False}
        self.assertEqual(format_template(base_only, feed=feed).strip(), "")
        rendered = format_template(overridden, feed=feed)
        self.assertEqual(rendered.count("## NOTES (on delivering large artifacts):"), 1)
        self.assertIn("<Response>", rendered)
        self.assertIn("COMPLETE artifact", rendered)
        for gated in ("create_file", "write_file", "heredoc", "ARG_MAX"):
            self.assertNotIn(gated, rendered)
        for jinja_token in ("{#", "{%", "{{"):
            self.assertNotIn(jinja_token, rendered)

    def test_base_large_file_writing_renders_when_local_access(self) -> None:
        base_root, _ = self._real_roots()
        base_only = _base_manager(base_root).load_variables(
            {"notes.large_file_writing": None}, root_space=""
        )["notes"]["large_file_writing"]
        rendered = format_template(base_only, feed={"has_local_access": True})
        self.assertIn("writing large files", rendered)
        self.assertIn("create_file", rendered)

    def _metamate_delivery_override(self, base_root: Path, metamate_root: Path) -> str:
        """The metamate override, resolved through the fork (Pass-1) and rendered
        Pass-2 with the metamate feed -- i.e. exactly the value the feed build
        (``_resolve_templated_feed``) places in ``{{ notes.large_file_writing }}``:
        the leading comment stripped and any guards evaluated."""
        inf = _discoverable_inferencer(metamate_root.parent)(
            template_manager=_base_manager(base_root)
        )
        pass1 = inf._rendering_manager().load_variables(
            {"notes.large_file_writing": None}, root_space=""
        )["notes"]["large_file_writing"]
        return format_template(pass1, feed={"has_local_access": False})

    def test_four_real_consumer_templates_render_delivery_override(self) -> None:
        base_root, metamate_root = self._real_roots()
        consumers = [
            base_root / "plan" / "main" / "initial.jinja2",
            base_root / "plan" / "main" / "followup.jinja2",
            base_root / "implementation" / "main" / "initial.jinja2",
            base_root / "implementation" / "main" / "followup.jinja2",
        ]
        if not all(c.is_file() for c in consumers):
            self.skipTest("consumer .jinja2 templates are not packaged in this runtime")
        override = self._metamate_delivery_override(base_root, metamate_root)
        render_feed = {
            "notes": {"large_file_writing": override},
            "has_local_access": False,
        }
        for consumer in consumers:
            label = f"{consumer.parent.parent.name}/{consumer.name}"
            with self.subTest(template=label):
                rendered = format_template(consumer.read_text(), feed=render_feed)
                # The metamate delivery guidance lands once, embedded in-context.
                self.assertEqual(
                    rendered.count("## NOTES (on delivering large artifacts):"), 1
                )
                self.assertIn("delivering large artifacts", rendered)
                self.assertGreater(len(rendered), len(override))
                # The has_local_access-gated local-file guidance never appears.
                for gated in ("create_file", "write_file", "heredoc", "ARG_MAX"):
                    self.assertNotIn(gated, rendered)
                # Pass-2 leaves no unrendered jinja behind.
                for jinja_token in ("{#", "{%", "{{"):
                    self.assertNotIn(jinja_token, rendered)

    def test_non_consumer_template_and_sibling_key_are_unaffected(self) -> None:
        base_root, metamate_root = self._real_roots()
        inf = _discoverable_inferencer(metamate_root.parent)(
            template_manager=_base_manager(base_root)
        )
        fork = inf._rendering_manager()
        base_mgr = _base_manager(base_root)
        # A base sibling key metamate does NOT override resolves identically on
        # the fork and the base; only notes.large_file_writing diverges.
        self.assertEqual(
            fork.load_variables({"notes.local_search_efficiency": None}, root_space=""),
            base_mgr.load_variables(
                {"notes.local_search_efficiency": None}, root_space=""
            ),
        )
        self.assertNotEqual(
            fork.load_variables({"notes.large_file_writing": None}, root_space="")[
                "notes"
            ]["large_file_writing"],
            base_mgr.load_variables({"notes.large_file_writing": None}, root_space="")[
                "notes"
            ]["large_file_writing"],
        )
        # A non-consumer template renders with the delivery override in the feed
        # yet never references the key -> the override cannot leak into it.
        non_consumer = base_root / "recovery" / "judge.jinja2"
        if not non_consumer.is_file():
            self.skipTest("recovery/judge.jinja2 resource is not packaged")
        override = self._metamate_delivery_override(base_root, metamate_root)
        rendered = format_template(
            non_consumer.read_text(),
            feed={
                "notes": {"large_file_writing": override},
                "has_local_access": False,
            },
        )
        self.assertNotIn("## NOTES (on delivering large artifacts):", rendered)
        self.assertNotIn("delivering large artifacts", rendered)
