# pyre-strict

"""Defensive normalization of heterogeneous inferencer results to text.

Orchestrators do not uniformly return ``str``: a ``DualInferencer`` /
``MultiFlowDual`` returns a ``DualInferencerResponse`` and a
``BreakdownThenAggregate`` with aggregation disabled returns a per-worker
tuple. :func:`extract_result_text` probes the known response shapes
(``.base_response`` -> ``.result`` -> ``.output`` -> tuple -> ``str()``) so a
caller can treat any inferencer's output as text.

Promoted from ``resources/tools/task/executor.py`` so ``common/`` code (e.g.
the ``agentic_functions`` decorator's stage-0 normalize) can reuse it without
importing up from the ``resources/tools/`` layer; the executor re-binds the old
private names to these functions so its call sites are unchanged.
"""

from __future__ import annotations

from typing import Any


def extract_result_text(result: Any) -> str:
    """Defensive normalization across PTI / BTA / Dual / single result shapes.

    Multi-element tuples (e.g. a ``disable_aggregator`` BTA / MFDual that returns
    one output per worker) are serialized in FULL via ``_serialize_multi_output``
    so no worker output is silently dropped.
    """
    if result is None:
        return ""
    base = getattr(result, "base_response", None)
    if isinstance(base, str) and base:
        return base
    plain = getattr(result, "result", None)
    if isinstance(plain, str) and plain:
        return plain
    output = getattr(result, "output", None)
    if isinstance(output, str) and output:
        return output
    if isinstance(result, tuple):
        non_none = [r for r in result if r is not None]
        if not non_none:
            return ""
        if len(non_none) == 1:
            return extract_result_text(non_none[0])
        return _serialize_multi_output(non_none)
    return str(result)


def _serialize_multi_output(parts: list[Any]) -> str:
    """Serialize multiple worker outputs (no-aggregate mode) into one markdown
    document so the FULL list survives into the calling conversation.

    Without this, ``extract_result_text`` collapsed a multi-worker tuple to
    ``result[0]`` and silently dropped the rest -- which made the no-aggregate /
    list-of-outputs-to-conversation pattern impossible. Each part renders under a
    ``### Worker N`` header.
    """
    blocks: list[str] = []
    for i, part in enumerate(parts, start=1):
        text = extract_result_text(part)
        blocks.append(f"### Worker {i}\n\n{text}".rstrip())
    return "\n\n".join(blocks)
