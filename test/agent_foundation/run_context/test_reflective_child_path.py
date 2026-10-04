"""ReflectiveInferencer runs its reflection call at its own child path."""

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ReflectionStyles,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.reflective_inferencer import (
    ReflectiveInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    RunContext,
)
from attr import attrib, attrs
from rich_python_utils.string_utils.formatting.python_str_format import (
    format_template as str_format_template,
)

PATHS = []


@pytest.fixture(autouse=True)
def _fresh_paths():
    PATHS.clear()
    yield
    PATHS.clear()


@attrs
class _PathRecorder(InferencerBase):
    _label = attrib(default="stub")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        ctx = active_run_context()
        PATHS.append((self._label, ctx.path if ctx else None))
        return f"{self._label}:{inference_input}"


def _reflective(style, num_reflections=1):
    return ReflectiveInferencer(
        base_inferencer=_PathRecorder(label="base"),
        reflection_inferencer=_PathRecorder(label="reflect"),
        num_reflections=num_reflections,
        reflection_style=style,
        reflection_prompt_template="{prompt}|{response}",
        reflection_prompt_formatter=str_format_template,
        unpack_single_response=True,
    )


@pytest.mark.parametrize(
    "style,num_reflections",
    [
        (ReflectionStyles.IntegrateAll, 1),
        (ReflectionStyles.Separate, 2),
        (ReflectionStyles.Sequential, 2),
    ],
)
def test_reflection_runs_at_reflect_child_path(style, num_reflections):
    ri = _reflective(style, num_reflections)

    ri.infer("q", run_context=RunContext.root(workspace=None))

    assert [label for label, _ in PATHS] == ["base"] + ["reflect"] * num_reflections
    assert dict(PATHS) == {"base": "/base", "reflect": "/reflect"}


def test_reflection_path_nests_under_the_reflective_node():
    ri = _reflective(ReflectionStyles.Separate)
    root = RunContext.root(workspace=None)

    ri.infer("q", run_context=root.child("critic"))

    assert PATHS == [("base", "/critic/base"), ("reflect", "/critic/reflect")]
