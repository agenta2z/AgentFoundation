"""§9.2 / §9.5: the layer-3 boundary filter drops stray kwargs at the API splat
while keeping the named provider params; run_context never reaches the splat."""

from rich_python_utils.common_utils.function_helper import get_relevant_named_args


def _fake_generate_text(
    inference_input, model=None, api_key=None, temperature=None,
    max_new_tokens=None, top_p=None, stop=None,
):
    return {"text": "ok", "model": model, "temperature": temperature}


def test_boundary_filter_drops_strays_keeps_named_provider_params():
    filtered = get_relevant_named_args(
        _fake_generate_text,
        model="m",
        api_key="k",
        temperature=0.5,
        max_new_tokens=256,
        # strays that must be dropped:
        STRAY="x",
        run_context="RC",
        foo=123,
    )
    assert filtered["model"] == "m"
    assert filtered["temperature"] == 0.5
    assert filtered["max_new_tokens"] == 256
    assert "STRAY" not in filtered
    assert "run_context" not in filtered
    assert "foo" not in filtered


def test_api_inferencer_base_uses_the_filter():
    """ApiInferencerBase._infer routes its splat through get_relevant_named_args."""
    import inspect

    from agent_foundation.common.inferencers.api_inferencer_base import (
        ApiInferencerBase,
    )

    src = inspect.getsource(ApiInferencerBase._infer)
    assert "get_relevant_named_args" in src, (
        "§9.2 layer-3 boundary filter missing from ApiInferencerBase._infer"
    )


def test_run_context_is_keyword_only_so_it_never_enters_inference_args():
    """Layer-1: run_context keyword-only on infer/ainfer -> never in **_inference_args."""
    import inspect

    from agent_foundation.common.inferencers.inferencer_base import InferencerBase

    for m in (InferencerBase.infer, InferencerBase.ainfer):
        p = inspect.signature(m).parameters["run_context"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
