# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""``ExperimentCombosBridge`` — Stage 2 of the split-experiment plan.

Thin re-export over :class:`ExperimentBridge` (the breakdown-disabled
combo path of the original `/experiment` bridge) plus the pre-flight
flag check that blocks combos referencing flags missing from the
codebase. Centralising the pre-flight here keeps `tool_executor` clean.

Plan: ``/home/zgchen/.claude/plans/humming-tinkering-wirth.md`` Phase 2.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from agent_foundation.experiment_hub.experiment_bridge import (
    Combo,
    ExperimentBridge,
    parse_combos_arg,
    parse_hypothesis_ids,
)


__all__ = (
    "Combo",
    "ExperimentCombosBridge",
    "aggregate_only_run",
    "collect_aggregator_input_from_disk",
    "collect_aggregator_input_from_submissions",
    "find_latest_experiment_workspace",
    "parse_combos_arg",
    "parse_hypothesis_ids",
    "preflight_check_flags",
    "resolve_active_multi_task_id",
)

logger: logging.Logger = logging.getLogger(__name__)

# Re-export under the new name so callers can switch incrementally.
ExperimentCombosBridge = ExperimentBridge


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight flag check
# ─────────────────────────────────────────────────────────────────────────────

# Best-effort: flags follow the existing ``enable_<name>: bool`` contract
# from ``hypothesis_implementation/default.jinja2``. The check just greps
# the search root for any string that looks like a flag declaration.
_FLAG_DECL_RE: re.Pattern[str] = re.compile(r"\benable_([A-Za-z0-9_]+)\b")

# How many bytes to read per file before giving up. Keeps the pre-flight
# cheap on large repos — flag declarations always live in small config /
# python source files anyway.
_MAX_BYTES_PER_FILE: int = 256 * 1024

# Suffixes worth scanning for ``enable_<name>`` declarations. Keep narrow
# to avoid drowning in vendored / generated noise.
_SCAN_SUFFIXES: tuple[str, ...] = (".py", ".yaml", ".yml", ".json", ".jinja2")


def _flag_for_id(hypothesis_id: str) -> str:
    return f"enable_{hypothesis_id.lower()}"


def preflight_check_flags(
    combos: list[list[str]],
    search_root: Path,
    *,
    max_files: int = 4000,
) -> dict[str, list[str]]:
    """Walk ``search_root`` looking for ``enable_<id>`` declarations and
    return ``{combo_key: [missing_flags]}`` for any combo whose flags
    aren't all declared. Empty dict ⇒ all combos pass.

    Best-effort: silently skips unreadable files. Returns an empty dict
    if the search root doesn't exist (caller decides whether to fail).
    """
    if not combos or not search_root.exists():
        return {}
    seen: set[str] = set()
    scanned = 0
    try:
        for path in search_root.rglob("*"):
            if scanned >= max_files:
                break
            if not path.is_file() or path.suffix not in _SCAN_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")[
                    :_MAX_BYTES_PER_FILE
                ]
            except OSError:
                continue
            for m in _FLAG_DECL_RE.finditer(text):
                seen.add(m.group(1).lower())
            scanned += 1
    except OSError as e:
        logger.warning("preflight_check_flags: rglob failed: %s", e)
        return {}

    blocked: dict[str, list[str]] = {}
    for combo in combos:
        missing = [_flag_for_id(hid) for hid in combo if hid.lower() not in seen]
        if missing:
            blocked[",".join(combo)] = missing
    return blocked


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator-only refresh path (--aggregate-only)
# ─────────────────────────────────────────────────────────────────────────────

# Default glob for per-combo analyses produced by ExperimentBridge.
# See experiment_bridge.py:303,571-573 for the workspace contract.
_DEFAULT_ANALYSIS_GLOB: str = "combos/*/analysis/combo_*.md"


# Per-doc and total-prompt size caps for the embedded-content path. The
# aggregator prompt embeds the full body of each per-combo analysis so the
# LLM doesn't need to issue per-file read tool calls. Caps protect against
# pathological combos and runaway total prompt size.
#
# Why we embed: BTA's `_build_aggregator_only_input` path-only-dict
# optimization (`bta:179, 996, 1224`) only fires when the aggregator's
# `has_local_access=True`. DevmateCliInferencer inherits the InferencerBase
# default of `False` (verified at inferencer_base.py:132; devmate_cli_inferencer.py
# never overrides). So BTA falls back to embedding full content — and we
# do the same here in the bypass path for consistency.
#
# NOTE: `has_local_access` ONLY gates BTA's path-only optimization and
# `InferencerBase._build_template_feed` (the single-inferencer template
# path, :513-516). It does NOT gate DualInferencer's `output_path` —
# DualInferencer renders templates via its own `_render_role_prompt`
# (`dual_inferencer.py:996-1015`) which spreads `**inference_config`
# directly, bypassing the gate. So `aggregate_only_run` can (and does)
# wire `output_path` via `inference_config` regardless of DevmateCli's
# inherited `has_local_access=False`.
_MAX_PER_COMBO_BYTES: int = 64 * 1024  # 64 KB per analysis doc
_MAX_TOTAL_BODY_BYTES: int = 1024 * 1024  # 1 MB total across all combos


def collect_aggregator_input_from_disk(
    experiment_workspace: Path,
    glob_pattern: str = _DEFAULT_ANALYSIS_GLOB,
) -> list[dict[str, str]]:
    """Discover per-combo analyses on disk and return dicts with FULL body.

    Each entry: ``{"path": "<abs path>", "combo_id": "<id from path>",
    "summary": "<full body, capped to _MAX_PER_COMBO_BYTES>"}``.

    The ``summary`` field is named for compatibility with BTA's
    ``_build_aggregator_only_input`` (which reads it as the inline content
    when the aggregator inferencer has ``has_local_access=False`` — the
    default for DevmateCliInferencer). Embedding the full body is required
    for the LLM to actually see "all the existing per-combo experiment
    docs" rather than just the first few hundred chars.

    Total embedded body is capped at ``_MAX_TOTAL_BODY_BYTES`` to keep the
    aggregator prompt tractable; on overflow, later combos get truncated
    bodies (a `[truncated: budget exceeded]` marker is appended) so the
    LLM still sees their existence.
    """
    results: list[dict[str, str]] = []
    if not experiment_workspace.exists():
        return results
    for path in sorted(experiment_workspace.glob(glob_pattern)):
        if not path.is_file():
            continue
        # combo_id := the leaf-2-up directory under combos/<id>/analysis/...
        try:
            combo_id = path.parents[1].name
        except IndexError:
            combo_id = path.stem
        try:
            full = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            full = ""
        results.append({"path": str(path), "combo_id": combo_id, "summary": full})
    return _apply_size_budget(results)


def _apply_size_budget(
    entries: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Apply per-doc + total-body byte caps with consistent truncation
    markers. Mutates entries in-place and returns them for chainability.

    Extracted so the disk-glob and hub-submissions collectors enforce
    the SAME budget + emit identical `[truncated: ...]` markers — the
    LLM prompt downstream is byte-identical regardless of input source.
    """
    total = 0
    for entry in entries:
        body = entry["summary"]
        if len(body.encode("utf-8")) > _MAX_PER_COMBO_BYTES:
            body = body[:_MAX_PER_COMBO_BYTES] + "\n\n[truncated: per-combo cap]\n"
        body_bytes = len(body.encode("utf-8"))
        if total + body_bytes > _MAX_TOTAL_BODY_BYTES:
            remaining = max(0, _MAX_TOTAL_BODY_BYTES - total)
            if remaining < 256:
                # No meaningful budget left — emit a stub so the LLM at
                # least knows this combo exists.
                body = "[truncated: total-prompt cap exceeded]\n"
            else:
                body = body[:remaining] + "\n\n[truncated: total-prompt cap]\n"
            total = _MAX_TOTAL_BODY_BYTES
        else:
            total += body_bytes
        entry["summary"] = body
    return entries


def collect_aggregator_input_from_submissions(
    submissions: list[dict[str, Any]],
    session_dir: Path,
    *,
    terminal_statuses: tuple[str, ...] = (
        "completed",
        "analyzed",
        "error",
        "cancelled",
        "failed",
        "early_killed",
    ),
    min_epochs: int = 0,
    include_incomparable: bool = True,
    include_errored: bool = True,
) -> list[dict[str, str]]:
    """Build per-combo aggregator inputs from a hub submissions list.

    Returns the same shape as :func:`collect_aggregator_input_from_disk`:
    ``{"path": "<abs path>", "combo_id": "<comboKey>", "summary": "<body>"}``.

    For each terminal submission:
      - ``combo_id`` ← ``comboKey`` (or ``id`` as fallback).
      - ``summary`` ← contents of ``analysisFile`` (with size budget) when
        the file is readable; else falls back to ``analysisSummary``; else
        a stub so the LLM at least knows the combo exists.
      - Non-terminal rows (queued/running/etc) are skipped — their
        analyses aren't yet final.

    Layer 4 filters (all default to back-compat — no filtering):
      - ``min_epochs``: skip rows whose ``epochsCompleted`` < N (missing
        field treated as 0). Default 0 = no filter.
      - ``include_incomparable``: when False, skip rows with
        ``verdict == "incomparable"``. Default True = keep them.
      - ``include_errored``: when False, skip rows whose status is
        ``error`` / ``failed`` / ``cancelled``. Default True = keep them.

    ``session_dir`` is used to resolve relative ``analysisFile`` paths.
    """
    raw_entries: list[dict[str, str]] = []
    for row in submissions:
        if row.get("status") not in terminal_statuses:
            continue
        # Layer 4 filters — order: cheapest first.
        if not include_errored and row.get("status") in (
            "error",
            "failed",
            "cancelled",
        ):
            continue
        if not include_incomparable and row.get("verdict") == "incomparable":
            continue
        if min_epochs > 0 and (row.get("epochsCompleted") or 0) < min_epochs:
            continue

        combo_id = str(row.get("comboKey") or row.get("id") or "")
        if not combo_id:
            continue
        analysis_path_str = row.get("analysisFile")
        analysis_path: Path | None = None
        body: str | None = None
        if analysis_path_str:
            ap = Path(analysis_path_str)
            if not ap.is_absolute():
                ap = session_dir / ap
            if ap.is_file():
                analysis_path = ap
                try:
                    body = ap.read_text(encoding="utf-8", errors="ignore")
                except OSError as e:
                    logger.warning(
                        "collect_from_submissions: read failed %s: %s",
                        ap,
                        e,
                    )
        if body is None:
            body = (row.get("analysisSummary") or "").strip()
            if not body:
                body = (
                    f"[{combo_id}] no analysis content available "
                    f"(analysisFile={analysis_path_str!r}, "
                    f"analysisSummary empty)"
                )
        raw_entries.append(
            {
                "path": str(analysis_path) if analysis_path else "",
                "combo_id": combo_id,
                "summary": body,
            }
        )
    return _apply_size_budget(raw_entries)


def resolve_active_multi_task_id(
    session_dir: Path,
    *,
    explicit: str | None = None,
    workflow_context_active: str | None = None,
) -> tuple[str | None, str | None]:
    """Pick the active hub's ``multi_task_id`` for this session.

    Returns ``(mid, error_msg)`` — exactly one is non-None. Precedence:
      1. ``explicit`` (from ``--reuse-hub``).
      2. ``workflow_context_active`` (from session.workflow_context).
      3. Exactly-one ``hub_*_submissions.json`` on disk → that one.
      4. Otherwise → ``(None, error_msg)`` listing the candidates so
         the user can disambiguate via ``--reuse-hub``.

    Each tier degrades gracefully — if the explicit hint points at a
    non-existent file, the function returns an explicit error rather
    than silently falling through.
    """
    if explicit:
        if not (session_dir / f"hub_{explicit}_submissions.json").exists():
            return None, (
                f"--reuse-hub {explicit}: no hub_{explicit}_submissions.json on disk"
            )
        return explicit, None
    if (
        workflow_context_active
        and (session_dir / f"hub_{workflow_context_active}_submissions.json").exists()
    ):
        return workflow_context_active, None
    hubs = sorted(
        p.name[len("hub_") : -len("_submissions.json")]
        for p in session_dir.glob("hub_*_submissions.json")
    )
    if len(hubs) == 1:
        return hubs[0], None
    if not hubs:
        return None, (
            "No hub submissions found in this session — run /experiment first"
        )
    return None, (
        f"Multiple hubs found ({hubs}); pass --reuse-hub <multi_task_id> "
        "to disambiguate"
    )


def find_latest_experiment_workspace(session_tasks_dir: Path) -> Path | None:
    """Return the most-recently-mtime'd ``exp_*`` workspace under
    ``session_tasks_dir`` that ALSO contains at least one per-combo
    analysis matching ``combos/*/analysis/combo_*.md``. Returns ``None``
    if no qualifying workspace exists.

    The shape check is the SIGNATURE of a Stage-2 orchestrator workspace.
    Bare ``exp_`` prefix is necessary but NOT sufficient — other task
    types (per-hypothesis analysis tasks, manual ``exp_*_rerun`` runs,
    etc.) also use that prefix. Without the shape filter the picker
    happily returned a workspace whose ``combos/`` glob is empty, which
    surfaces as a confusing "no per-combo analyses at <path>" error
    pointing at a workspace that was NEVER intended to be a Stage-2
    orchestrator output.
    """
    if not session_tasks_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in session_tasks_dir.glob("exp_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cand in candidates:
        if next(cand.glob(_DEFAULT_ANALYSIS_GLOB), None) is not None:
            return cand
    return None


def _build_aggregate_only_prompt_builder(
    precompute_envelope: dict | None,
):
    """Construct an aggregator_prompt_builder for BTA that prepends a
    clearly-labeled precompute-envelope section before the per-combo
    analyses, matching what the meta_mrs_rankevolve task_preamble +
    hypothesis_reranking_and_combo task_instructions expect.

    The builder receives ``flattened_results`` (the per-combo bodies that
    BTA's ``_build_aggregator_only_input`` already extracted from
    path-dicts) plus ``worker_output_paths`` for the file-ref hints.
    """
    import json as _json

    if precompute_envelope is None:
        precompute_section = None
    else:
        precompute_section = (
            "## Deterministic learnings_actions precompute envelope\n\n"
            "The Implementation Hub side panel parses these structural rows. "
            "The text fields (`rationale`, `title`, `risk`) and `openQuestions[]` "
            "are EMPTY for you to fill. Pass-through fields (the rest) MUST be "
            "copied verbatim into your emitted `learnings_actions` JSON fence.\n\n"
            "```json\n" + _json.dumps(precompute_envelope, indent=2) + "\n```\n"
        )

    def _builder(
        worker_results,
        original_query=None,
        worker_output_paths=None,
        bta=None,
    ) -> str:
        parts: list[str] = []
        if precompute_section is not None:
            parts.append(precompute_section)
        for idx, res in enumerate(worker_results):
            path = (
                worker_output_paths[idx]
                if worker_output_paths and idx < len(worker_output_paths)
                else None
            )
            path_ref = f"\n(Full output at: `{path}`)" if path else ""
            parts.append(f"### Per-combo analysis {idx + 1}\n{res}{path_ref}")
        return "\n\n".join(parts)

    return _builder


def _flatten_inputs_for_capture(
    inputs: list[dict[str, str]],
) -> tuple[list[str], list[str | None]]:
    """Mirror BTA._build_aggregator_only_input's path-dict flattening so
    we can capture the EXACT prompt the aggregator sees into a sub-task
    workspace artifact for inspection. Pure function — no side effects.
    """
    flattened: list[str] = []
    paths: list[str | None] = []
    for item in inputs:
        if isinstance(item, dict) and "path" in item:
            paths.append(str(item["path"]))
            flattened.append(item.get("summary") or item.get("path", ""))
        else:
            paths.append(None)
            flattened.append(item if isinstance(item, str) else str(item))
    return flattened, paths


async def aggregate_only_run(
    experiment_workspace: Path,
    target_path: Path,
    *,
    aggregator_factory: Any,  # pyre-ignore[2] — closure injected by tool_executor
    glob_pattern: str = _DEFAULT_ANALYSIS_GLOB,
    precompute_envelope: dict | None = None,
    task_workspace: Path | None = None,
    inputs: list[dict[str, str]] | None = None,
) -> str:
    """Run the aggregation-only BTA over the on-disk per-combo analyses.

    When ``precompute_envelope`` is supplied (typically the result of
    ``learnings_generator.precompute_actions(...)`` computed by the
    bridge before the LLM call), it's prepended to the LLM input as a
    clearly-labeled section so the aggregator can fill rationale/title/
    risk fields and author openQuestions. When None, the LLM only sees
    the per-combo bodies and per the task_instructions emits empty
    rerank/combo arrays (the deterministic structure still shows up via
    ``regenerate_accumulated_learnings``).

    When ``task_workspace`` is supplied, the bridge writes the LLM's
    full input (``llm_input.md``) and raw response (``llm_response.md``)
    into that directory so the corresponding sub-task chip's tab can
    surface them for inspection. Capture failures are best-effort and
    must not break the LLM call.

    Returns the LLM-rendered markdown body. Caller is responsible for
    routing through ``learnings_generator.regenerate_accumulated_learnings``
    so the deterministic structural fields are merged with the LLM's
    enriched fence.
    """
    # Streaming-gap-fix L1: dropped the BTA import — we now bypass
    # `BreakdownThenAggregateInferencer` for the aggregator-only path
    # since it's structurally overkill (no workers to break down) and
    # its `children/aggregator/` workspace reroute pushes SessionLogger
    # output one level deeper than the chip workspace expects. Direct
    # `agg_inf.ainfer(prompt_str)` below replaces the BTA wrapper.

    # When `inputs` is supplied (hub-driven path), use it directly and
    # skip on-disk discovery. The disk-glob path stays available for
    # callers that don't pre-load inputs (Stage-2 `--combos` workflow).
    if inputs is None:
        inputs = collect_aggregator_input_from_disk(experiment_workspace, glob_pattern)
    if not inputs:
        raise ValueError(
            "--aggregate-only: no per-combo analyses available "
            f"(workspace={experiment_workspace}/{glob_pattern}; no "
            "hub-sourced inputs supplied). Run /experiment-hypothesis-combos "
            "--combos first, or wait for hub submissions to reach a "
            "terminal status."
        )

    agg_inf = aggregator_factory()
    builder = _build_aggregate_only_prompt_builder(precompute_envelope)

    # Build the prompt UNCONDITIONALLY — we need it for the direct
    # `agg_inf.ainfer(prompt_str)` call below. The capture to
    # llm_input.md is best-effort; the build itself must always happen.
    flattened, paths = _flatten_inputs_for_capture(inputs)
    prompt_str = builder(flattened, worker_output_paths=paths)

    # Capture the prompt to the sub-task workspace BEFORE the LLM call.
    # AFTER the BTA bypass below, this capture is DEFINITIONALLY accurate
    # — we own the prompt build; what we capture IS what gets sent as
    # `{{ input }}` to the template (see legend in summary.md). Distinct
    # from SessionLogger's post-template capture at
    # logs/session/<Inferencer>.jsonl.parts/InferenceInput/<ts>_*.txt.
    if task_workspace is not None:
        try:
            task_workspace.mkdir(parents=True, exist_ok=True)
            (task_workspace / "llm_input.md").write_text(prompt_str, encoding="utf-8")
        except OSError as e:
            logger.warning(
                "aggregate_only_run: failed to capture llm_input.md to %s: %s",
                task_workspace,
                e,
            )

    # Streaming-gap-fix L1: bypass BTA on the aggregator-only path.
    #
    # When `disable_workers=True` + `predefined_worker_results=inputs` +
    # we already have a pre-built `prompt_str`, BTA's only contributions
    # are (a) re-building the prompt via its `aggregator_prompt_builder`
    # (duplicating our `builder(...)` call above) and (b) rerouting the
    # aggregator's `_workspace` to a `children/aggregator/` subdir
    # (`breakdown_then_aggregate_inferencer.py:1070-1073`) which buries
    # the canonical SessionLogger output (logs/session/...) one level
    # deeper than the chip workspace expects. Bypassing BTA gives:
    #   - DualInferencer's logs/session/<Inferencer>.jsonl.parts/...
    #     directly under task_workspace (matches `_exec_task` pattern).
    #   - Captured `prompt_str` IS the actual {{ input }} body (no
    #     fragile "best-effort" invariant — we own the call).
    #   - No more `<task_workspace>/_aggregate_only/children/aggregator/`
    #     scaffolding clutter.
    # The DevmateCli's cache_folder (where stream_*.txt files land) is
    # already correctly wired by `_agg_factory`'s `workspace_path=`
    # passthrough — see tool_executor._exec_aggregator_only_refresh.
    if task_workspace is not None and hasattr(agg_inf, "_workspace"):
        from agent_foundation.common.inferencers.inferencer_workspace import (  # @manual
            InferencerWorkspace,
        )

        # Setting `_workspace` triggers InferencerBase's setter which
        # auto-configures cache_folder + redirects loggers. Mirrors what
        # BTA does on line 1073 (same private contract, same call site).
        agg_inf._workspace = InferencerWorkspace(root=str(task_workspace))
        # ensure_dirs creates the 4 core dirs (outputs, artifacts,
        # checkpoints, logs); add the extras we want for canonical layout.
        agg_inf._workspace.ensure_dirs("analysis", "results", "_runtime")

    # Direct call — no BTA wrapper. The prompt_str we built above is
    # passed as the inference input; DualInferencer's prompt_formatter
    # threads it through the template (aggregation/main/initial.jinja2).
    # The captured llm_input.md (above) is the value of `{{ input }}`;
    # the post-template rendered prompt lands in the canonical
    # SessionLogger capture at logs/session/<Inferencer>.jsonl.parts/
    # InferenceInput/<ts>_*.txt.
    #
    # `inference_config["output_path"]` enables the dual's per-round
    # file-reference substitution (dual_inferencer.py:_maybe_replace_with
    # _file_reference legacy mode, :1292+):
    #   - initial.jinja2:26 + review.jinja2:13 + followup.jinja2:56
    #     branches flip from "Emit inline" to "Write to <path>" — DevmateCli
    #     uses its file-write tools (or the framework writes the inline
    #     response itself as fallback at :1308-1326).
    #   - Round 2+ review/followup prompts get a one-line file reference
    #     in the `<ProposedAggregation>` slot instead of re-embedding the
    #     full ~22 KB body. Compounds with max_iterations.
    # Path matches the canonical workspace layout established by
    # `_finalize_response` (dual_inferencer.py:754-771) — per-round under
    # `artifacts/`. We use legacy mode (inference_config) rather than
    # workspace mode (dual.output_path attribute) deliberately:
    #   - Avoids `_finalize_response` creating a redundant outputs/
    #     copy that would conflict with tool_executor.py:4146-4157's
    #     post-merge `outputs/accumulated_learnings.md`.
    #   - Keeps the path naming local to this caller, not baked into
    #     `_build_dual` (which is shared with ResearchProposeBridge).
    inference_config: dict = {}
    if task_workspace is not None:
        artifacts_dir = task_workspace / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        inference_config["output_path"] = (
            f"{artifacts_dir}/round{{{{ round_index }}}}_aggregation.md"
        )
    result = await agg_inf.ainfer(prompt_str, inference_config=inference_config)
    llm_md = str(result) if result is not None else ""

    # When `output_path` was set, the dual's final return value is the
    # per-round file-reference substitution string ("The complete output
    # has been written to: <path>... Read that file...") rather than the
    # actual proposal body — by design, for inter-round token efficiency
    # (dual_inferencer.py:_maybe_replace_with_file_reference :700, :930-931).
    # Recover the real content from the highest-numbered artifact so the
    # downstream validation/staging pipeline sees the full markdown.
    #
    # ──────────────────────────────────────────────────────────────────
    # NOTE — INTENTIONAL fail-loud canary; do NOT change to a "skip
    # non-substantive, fall back to earlier round" heuristic without
    # reading the silent-failure analysis at the top of:
    #   /home/zgchen/.claude/plans/humming-tinkering-wirth.md
    # (the "Refresh Learnings — fix the LLM-edit-in-place pattern" plan)
    #
    # Briefly: when round_N (the highest round) is the framework's
    # short `<Response>` fallback summary instead of canonical refined
    # content, this code propagates the summary verbatim and the
    # downstream validator at tool_executor.py:_exec_aggregator_only_refresh
    # rejects it ("body lacks any '## ' heading"). The chip lands error
    # and the user investigates upstream — the LLM probably edited a
    # PRIOR round's file in place (see the "Targeted edits" bullet in
    # `aggregation/main/followup.jinja2` for the prevention guidance)
    # OR the LLM's tool calls failed silently.
    #
    # A "smart recovery" that falls back to round_(N-1) when round_N is
    # non-substantive would silently ship the prior round's content as
    # canonical — which is WRONG when the LLM's tool calls failed
    # because the prior round won't have the reviewer-flagged fixes.
    # The current strictness IS the diagnostic that tells us the
    # artifact pipeline broke; suppressing it converts a loud failure
    # into a silent regression of correctness.
    #
    # If you ever genuinely need to recover from this state non-
    # destructively, you must FIRST add a way to distinguish "LLM
    # legitimately edited prior round in place" from "LLM tool failed
    # silently" — e.g. compare round_(N-1) mtime against the bridge's
    # start time, or hook the LLM's tool-call event stream. Without
    # that distinction, do not weaken the check.
    # ──────────────────────────────────────────────────────────────────
    if task_workspace is not None and inference_config.get("output_path"):
        artifacts_dir = task_workspace / "artifacts"
        if artifacts_dir.is_dir():

            def _round_idx(p: Path) -> int:
                # round{N}_aggregation.md → N; tolerate zero-padding.
                stem = p.name[len("round") :].split("_", 1)[0]
                try:
                    return int(stem)
                except ValueError:
                    return -1

            rounds = sorted(
                (
                    p
                    for p in artifacts_dir.glob("round*_aggregation.md")
                    if _round_idx(p) >= 0
                ),
                key=_round_idx,
            )
            if rounds:
                final_artifact = rounds[-1]
                try:
                    artifact_md = final_artifact.read_text(encoding="utf-8")
                except OSError as e:
                    logger.warning(
                        "aggregate_only_run: failed to read final artifact "
                        "%s: %s — falling back to in-memory return value",
                        final_artifact,
                        e,
                    )
                else:
                    if artifact_md.strip():
                        logger.info(
                            "aggregate_only_run: recovered final aggregation "
                            "from %s (%d bytes; in-memory return was %d bytes)",
                            final_artifact,
                            len(artifact_md),
                            len(llm_md),
                        )
                        llm_md = artifact_md

    # Capture the LLM response (post-LLM, pre-merge).
    if task_workspace is not None:
        try:
            (task_workspace / "llm_response.md").write_text(llm_md, encoding="utf-8")
        except OSError as e:
            logger.warning(
                "aggregate_only_run: failed to capture llm_response.md to %s: %s",
                task_workspace,
                e,
            )

    return llm_md
