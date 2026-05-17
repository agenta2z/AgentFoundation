"""Phase 7 — Permanent regression invariants for the inferencer axes design.

These tests are PERMANENT — they pin invariants that the entire axes
refactor depends on. Any future contributor breaking them gets immediate
CI failure.

See: _docs/_plans/inferencer_axes_INTEGRATED_v5_plan.md §11
"""

import attr
from attr import attrs, attrib

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_base import (
    TerminalInferencerBase,
    TerminalTemplatedInferencerBase,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    TerminalSessionInferencerBase,
    TerminalSessionTemplatedInferencerBase,
)


class TestAxesIsinstanceMatrix:
    """Pin the isinstance results for all axis combinations."""

    def test_sib_is_not_templated(self):
        assert not issubclass(StreamingInferencerBase, TemplatedInferencerBase)

    def test_tib_is_not_templated(self):
        assert not issubclass(TerminalInferencerBase, TemplatedInferencerBase)

    def test_tsib_is_not_templated(self):
        assert not issubclass(TerminalSessionInferencerBase, TemplatedInferencerBase)

    def test_tstib_is_templated(self):
        assert issubclass(TerminalSessionTemplatedInferencerBase, TemplatedInferencerBase)

    def test_ttib_is_templated(self):
        assert issubclass(TerminalTemplatedInferencerBase, TemplatedInferencerBase)

    def test_tsib_is_terminal(self):
        assert issubclass(TerminalSessionInferencerBase, TerminalInferencerBase)

    def test_tsib_is_streaming(self):
        assert issubclass(TerminalSessionInferencerBase, StreamingInferencerBase)

    def test_tstib_is_terminal(self):
        assert issubclass(TerminalSessionTemplatedInferencerBase, TerminalInferencerBase)

    def test_tstib_is_streaming(self):
        assert issubclass(TerminalSessionTemplatedInferencerBase, StreamingInferencerBase)


class TestThreeDiamondMROs:
    """Pin the C3-linearized MROs for all three diamond classes."""

    def _mro_names(self, cls):
        return [c.__name__ for c in cls.__mro__
                if c.__name__ not in ("object",)]

    def test_tsib_mro(self):
        mro = self._mro_names(TerminalSessionInferencerBase)
        tib_idx = mro.index("TerminalInferencerBase")
        sib_idx = mro.index("StreamingInferencerBase")
        ib_idx = mro.index("InferencerBase")
        assert tib_idx < sib_idx < ib_idx

    def test_tstib_mro(self):
        mro = self._mro_names(TerminalSessionTemplatedInferencerBase)
        tsib_idx = mro.index("TerminalSessionInferencerBase")
        tib_idx = mro.index("TerminalInferencerBase")
        sib_idx = mro.index("StreamingInferencerBase")
        template_idx = mro.index("TemplatedInferencerBase")
        ib_idx = mro.index("InferencerBase")
        assert tsib_idx < tib_idx < sib_idx < template_idx < ib_idx

    def test_ttib_mro(self):
        mro = self._mro_names(TerminalTemplatedInferencerBase)
        tib_idx = mro.index("TerminalInferencerBase")
        template_idx = mro.index("TemplatedInferencerBase")
        ib_idx = mro.index("InferencerBase")
        assert tib_idx < template_idx < ib_idx


class TestDiamondAttrsSlotsConsistency:
    """All classes in the three diamond MROs must use slots=False.

    If a future contributor migrates one of them to @attrs.define
    (modern API, defaults to slots=True), the diamond inheritance
    breaks MRO-based field resolution.
    """

    def test_diamond_attrs_slots_consistency(self):
        classes = [
            InferencerBase,
            TemplatedInferencerBase,
            StreamingInferencerBase,
            TerminalInferencerBase,
            TerminalSessionInferencerBase,
            TerminalTemplatedInferencerBase,
            TerminalSessionTemplatedInferencerBase,
        ]
        for cls in classes:
            slots = getattr(cls, "__slots__", None)
            assert slots is None or slots == (), (
                f"{cls.__name__} has non-empty __slots__={slots!r} — "
                "diamond inheritance with mixed slots will break MRO-based "
                "field resolution. Stay on the legacy `from attr import attrs` "
                "API for all classes in the inferencer axes diamonds."
            )


class TestNoDuplicateFieldsUnderDiamond:
    """No diamond class should have duplicate field names."""

    def test_tsib_no_duplicate_fields(self):
        names = [f.name for f in attr.fields(TerminalSessionInferencerBase)]
        assert len(names) == len(set(names)), (
            f"Duplicate fields in TSIB: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_tstib_no_duplicate_fields(self):
        names = [f.name for f in attr.fields(TerminalSessionTemplatedInferencerBase)]
        assert len(names) == len(set(names)), (
            f"Duplicate fields in TSTIB: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_ttib_no_duplicate_fields(self):
        names = [f.name for f in attr.fields(TerminalTemplatedInferencerBase)]
        assert len(names) == len(set(names)), (
            f"Duplicate fields in TTIB: "
            f"{[n for n in names if names.count(n) > 1]}"
        )


class TestLoadBearingConstraints:
    """Pin the constraints that the _configure_for_workspace guard depends on."""

    def test_tib_target_path_field_default_is_None(self):
        """target_path MUST default to None (NEVER os.getcwd()).
        If this fails, the _configure_for_workspace guard becomes broken
        for orchestrator-spawned children. See plan §2.1.
        """
        assert attr.fields(TerminalInferencerBase).target_path.default is None

    def test_tib_timeout_default_is_None(self):
        """TIB.timeout must default to None (no subprocess cap).
        Historic value of 300 was a footgun for session subclasses that
        inherited it silently. See plan §2.
        """
        assert attr.fields(TerminalInferencerBase).timeout.default is None

    def test_tsib_inherits_timeout_None_default(self):
        """TSIB must not silently activate a subprocess timeout."""
        assert attr.fields(TerminalSessionInferencerBase).timeout.default is None
