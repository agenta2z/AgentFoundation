# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""``ImplementHypothesisBridge`` — orchestrates the ``/implement-hypothesis`` slash command.

Composition (per the plan at ``/home/zgchen/.claude/plans/humming-tinkering-wirth.md``):

    outer BTA:
      breakdown    = DualInferencer(hypothesis_batch_grouper template)
                     Devmate emits list[Batch{batch_id, items, rationale}].
                     Soft validator caps batch size + drops unknown IDs.
      worker_fac   = PlanThenImplementInferencer per batch (parallel)
                     Reuses the existing `implementation` template space
                     verbatim (already prescribes the enable_<name> feature
                     flag contract + state_dict compat).
      aggregator   = DualInferencer(aggregation/main/initial.jinja2 +
                     task_preamble="@implementation_report" variant)
                     Writes <workspace>/results/implementation_report.md +
                     optionally appends a row to hub_<mid>_implementations.json.

Workspace convention: ``<session_dir>/tasks/implhyp_<ts>_<hex>/`` — matches
the chip / queue / persistence convention. Per-batch workspaces nest at
``<workspace>/batches/<batch_id>/``.

This bridge is **deterministic plumbing** — the LLM-template body lives in
``src/resources/prompt_templates/{hypothesis_batch_grouper,aggregation}/``
(the latter via ``task_preamble="@implementation_report"`` variant) + the
existing ``implementation/`` template space. The shape & invariants
the bridge enforces are stable; the prose inside the templates can evolve
without changing this file.

See :class:`ResearchProposeBridge` for the closest reference shape (same
template-resolution / workspace / DualInferencer construction patterns).
"""

from __future__ import annotations

import json
import logging
import re
import secrets
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.experiment_hub.stream_bridge import StreamBridgeAdapter
from rich_python_utils.string_utils.formatting.template_manager.template_manager import (  # @manual
    TemplateManager,
)


logger: logging.Logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configurable defaults
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MAX_BATCH_SIZE: int = 5
_DEFAULT_MAX_PARALLEL: int = 2
# Soft band: LLM may exceed `max_batch_size` by up to this multiplier;
# anything beyond gets re-prompted. Matches the prompt's "1–2 over the cap"
# guidance in the soft band; rejects 10+ for a cap of 5.
_HARD_BATCH_CAP_MULTIPLIER: float = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Batch dataclass + parsing
# ─────────────────────────────────────────────────────────────────────────────


class Batch:
    """One implementation batch — N hypotheses to be implemented together.

    Plain class (not attrs) so callers without an agentic_foundation dep
    can introspect.
    """

    __slots__ = ("batch_id", "items", "rationale", "workspace")

    def __init__(
        self,
        batch_id: str,
        items: list[str],
        rationale: str = "",
        workspace: Optional[str] = None,
    ) -> None:
        self.batch_id = batch_id
        self.items = items
        self.rationale = rationale
        self.workspace = workspace

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "items": self.items,
            "rationale": self.rationale,
            "workspace": self.workspace,
        }

    def __repr__(self) -> str:
        return f"Batch({self.batch_id!r}, items={self.items!r})"


_BATCH_RESPONSE_RE: re.Pattern[str] = re.compile(
    r"<Response>(.*?)</Response>", re.DOTALL
)
_FENCED_JSON_RE: re.Pattern[str] = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def parse_batches_response(raw: Any, valid_ids: set[str]) -> list[Batch]:
    """Parse the grouper LLM's structured-JSON output into ``list[Batch]``.

    Tries (in order): ``<Response>``-tagged content → fenced JSON block →
    bare JSON. Filters items to ``valid_ids`` (drops hallucinated IDs with
    a warning). Returns ``[]`` on unparseable input — caller decides
    whether to fall back to one-batch-per-hypothesis or re-prompt.
    """
    text = str(raw)
    response_match = _BATCH_RESPONSE_RE.search(text)
    body = response_match.group(1) if response_match else text
    json_match = _FENCED_JSON_RE.search(body)
    json_blob = json_match.group(1) if json_match else body
    try:
        data = json.loads(json_blob)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "parse_batches_response: JSON parse failed (%s); raw=%r", e, text[:200]
        )
        return []
    if not isinstance(data, dict):
        logger.warning(
            "parse_batches_response: expected dict, got %s", type(data).__name__
        )
        return []
    raw_batches = data.get("batches")
    if not isinstance(raw_batches, list):
        logger.warning("parse_batches_response: missing or non-list 'batches'")
        return []
    out: list[Batch] = []
    for entry in raw_batches:
        if not isinstance(entry, dict):
            continue
        batch_id = str(entry.get("batch_id") or "").strip()
        items_raw = entry.get("items") or []
        if not isinstance(items_raw, list):
            continue
        # Filter to valid IDs; drop hallucinations with a warning.
        items: list[str] = []
        for it in items_raw:
            sit = str(it).strip()
            if sit and sit in valid_ids:
                items.append(sit)
            elif sit:
                logger.warning(
                    "parse_batches_response: dropping unknown hypothesis id %r in batch %r",
                    sit,
                    batch_id,
                )
        if not batch_id or not items:
            continue
        rationale = str(entry.get("rationale") or "").strip()
        out.append(Batch(batch_id=batch_id, items=items, rationale=rationale))
    return out


def validate_and_repair_batches(
    batches: list[Batch],
    selected_ids: list[str],
    max_batch_size: int,
) -> tuple[list[Batch], list[str]]:
    """Apply soft constraints. Returns ``(repaired_batches, warnings)``.

    Constraints (in order):
      1. Hard cap: any batch larger than ``max_batch_size *
         _HARD_BATCH_CAP_MULTIPLIER`` is split into chunks of size
         ``max_batch_size`` (with a warning — caller can re-prompt instead).
      2. De-duplicate IDs across batches: each ID may appear at most once.
         If the LLM put the same ID in two batches, keep the first
         occurrence and drop subsequent ones (with a warning).
      3. Coverage: any selected ID not appearing in any batch is added as
         its own single-item batch (with a warning — the LLM forgot it).

    The result is always a valid batching: every selected ID appears in
    exactly one batch, every batch is within the hard cap. Returns also
    the list of warning strings the caller can surface to the user / log.
    """
    warnings_out: list[str] = []
    seen: set[str] = set()
    out: list[Batch] = []
    selected_set = set(selected_ids)
    hard_cap = int(max_batch_size * _HARD_BATCH_CAP_MULTIPLIER)
    for b in batches:
        # De-dup within batch + against prior batches.
        deduped: list[str] = []
        for it in b.items:
            if it in seen:
                warnings_out.append(
                    f"hypothesis {it!r} appeared in multiple batches; "
                    f"kept first occurrence (dropped from {b.batch_id})"
                )
                continue
            seen.add(it)
            deduped.append(it)
        if not deduped:
            warnings_out.append(
                f"batch {b.batch_id} became empty after de-dup; dropped"
            )
            continue
        # Hard-cap split.
        if len(deduped) > hard_cap:
            warnings_out.append(
                f"batch {b.batch_id} had {len(deduped)} items > hard cap "
                f"{hard_cap}; auto-split into chunks of {max_batch_size}"
            )
            for i in range(0, len(deduped), max_batch_size):
                chunk = deduped[i : i + max_batch_size]
                chunk_id = (
                    b.batch_id if i == 0 else f"{b.batch_id}_p{i // max_batch_size}"
                )
                out.append(Batch(batch_id=chunk_id, items=chunk, rationale=b.rationale))
        else:
            out.append(Batch(batch_id=b.batch_id, items=deduped, rationale=b.rationale))
    # Coverage check: any missing IDs become single-item batches.
    missing = [hid for hid in selected_ids if hid in selected_set and hid not in seen]
    for i, hid in enumerate(missing):
        warnings_out.append(
            f"hypothesis {hid!r} was not covered by any batch; added as singleton"
        )
        out.append(
            Batch(
                batch_id=f"orphan_{i}_{hid}",
                items=[hid],
                rationale=f"Auto-added: not covered by LLM grouping",
            )
        )
    return out, warnings_out


# ─────────────────────────────────────────────────────────────────────────────
# Workspace + template helpers (mirror ResearchProposeBridge)
# ─────────────────────────────────────────────────────────────────────────────


def _get_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".sl").is_dir():
            return parent
    return current.parent


# TODO(port): hub prompt_templates not yet vendored — _build_dual/_create_llm_inferencer paths require them at runtime.
def _get_templates_dir() -> Path:
    """Locate ``rankevolve/src/resources/prompt_templates``."""
    try:
        import importlib.resources as pkg_resources

        ref = pkg_resources.files(
            "agent_foundation.experiment_hub.resources.prompt_templates"
        )
        candidate = Path(str(ref))
        if candidate.is_dir():
            return candidate
    except (ImportError, TypeError, NotADirectoryError):
        pass
    candidate = Path(__file__).resolve().parent / "resources" / "prompt_templates"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        "prompt_templates directory not found. Tried importlib.resources "
        f"and filesystem path {candidate!r}. Ensure "
        "//rankevolve/src/resources:prompt_templates is in your deps."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Command-line option parsing for /implement-hypothesis
# ─────────────────────────────────────────────────────────────────────────────


def parse_implement_hypothesis_options(args: str) -> tuple[str, dict[str, Any]]:
    """Parse the ``/implement-hypothesis`` flag tail.

    Recognized flags:
      --select H1,H17,H8         Hypothesis IDs to implement
      --plan <abs-path>          Markdown plan to mine for hypothesis bodies
      --max-batch-size N         Soft cap on per-batch size (default 5)
      --max-parallel N           Concurrent PTI worker cap (default 2)
      --reuse-task <task_id>     Resume into an existing implementation task
      --workspace <abs-path>     Override workspace location
      --base-inferencer <type>   LLM inferencer (default 'devmate_cli')
      --workflow-target-path <p> Codebase root path
      --model <name>             LLM model override
    """
    tokens = args.split()
    options: dict[str, Any] = {}
    request_parts: list[str] = []
    i = 0

    def _consume(key: str) -> None:
        nonlocal i
        if i + 1 < len(tokens):
            i += 1
            options[key] = tokens[i]

    while i < len(tokens):
        tok = tokens[i]
        if tok == "--select":
            _consume("select")
        elif tok == "--plan":
            _consume("plan")
        elif tok == "--max-batch-size":
            _consume("max_batch_size")
        elif tok == "--max-parallel":
            _consume("max_parallel")
        elif tok == "--reuse-task":
            _consume("reuse_task")
        elif tok == "--workspace":
            _consume("workspace")
        elif tok == "--base-inferencer":
            _consume("base_inferencer")
        elif tok == "--workflow-target-path":
            _consume("workflow_target_path")
        elif tok == "--model":
            _consume("model")
        else:
            request_parts.append(tok)
        i += 1
    return " ".join(request_parts), options


# ─────────────────────────────────────────────────────────────────────────────
# Bridge
# ─────────────────────────────────────────────────────────────────────────────


class ImplementHypothesisBridge:
    """Bridge for the ``/implement-hypothesis`` slash command.

    Constructs a :class:`BreakdownThenAggregateInferencer` whose breakdown
    is a Devmate-driven batch grouper, whose workers are PTI per batch
    (rendering the existing ``implementation`` template space with its
    feature-flag contract), and whose aggregator writes the
    implementation report.

    Workspace convention: ``<session_dir>/tasks/implhyp_<ts>_<hex>/`` —
    matches chip / queue / persistence. Per-batch workspaces nest at
    ``<workspace>/batches/<batch_id>/``.

    Streaming: every nested inferencer writes to
    ``<workspace>/_runtime/inferencer_cache/<id>/stream_*.txt`` — the
    agent service bridge's ``WorkspaceStreamTailer`` discovers them
    automatically. No new transport.
    """

    def __init__(
        self,
        session_tasks_dir: Path,
        plan_text: str,
        selected_ids: list[str],
        *,
        model: Optional[str] = None,
        base_inferencer_type: str = "devmate_cli",
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
        max_parallel: int = _DEFAULT_MAX_PARALLEL,
        workflow_target_path: str = "",
        workspace_path: Optional[Path] = None,
        reuse_task: Optional[str] = None,
        session_context: Optional[dict[str, Any]] = None,
        # Allow tests to inject mocked sub-inferencers without subclassing.
        breakdown_factory: Optional[Any] = None,
        worker_factory_override: Optional[Any] = None,
        aggregator_factory: Optional[Any] = None,
        # Optional conflict pairs (e.g. from HYPOTHESIS_SLOTS slot-overlap).
        conflict_pairs: Optional[list[tuple[str, str, str]]] = None,
        hypothesis_slots: Optional[dict[str, list[str]]] = None,
        # Optional structured per-H metadata sourced from the hub's
        # selectionSnapshot. When provided, the grouper sees a markdown
        # table (id / phase / title / slots / description) instead of
        # bare H IDs — gives it real signal for category/relatedness
        # batching. The prompt template renders these only when
        # non-empty (defensive `{% if %}` guards mirror the existing
        # slots/conflicts blocks).
        hypothesis_metadata: Optional[list[dict[str, Any]]] = None,
        hypothesis_phase: Optional[dict[str, str]] = None,
        # Hub binding: when set, downstream consumers (Selection-tab
        # "Done" badges, Apply Combos preflight gate) read the
        # implementation evidence from the HUB's
        # ``hub_<hub_id>_implementations.json``. None → standalone
        # mode → bridge falls back to its own task id.
        hub_id: Optional[str] = None,
    ) -> None:
        self._session_tasks_dir = session_tasks_dir
        self._plan_text = plan_text
        self._selected_ids = list(selected_ids or [])
        self._model = model
        self._base_inferencer_type = base_inferencer_type
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_parallel = max(1, min(5, int(max_parallel)))  # hard cap at 5
        self._workflow_target_path = workflow_target_path
        self._session_context: dict[str, Any] = dict(session_context or {})
        # Phase C3: collect user-visible notices about how this bridge
        # interpreted its construction args. Currently surfaces the
        # silent ``max_parallel`` clamp; the executor reads this list
        # after construction and propagates entries onto the outer
        # task_status chip's metadata so the frontend can render them.
        self._notices: list[dict[str, str]] = []
        try:
            _user_max_parallel = int(max_parallel)
        except (TypeError, ValueError):
            _user_max_parallel = self._max_parallel
        if _user_max_parallel > 5:
            self._notices.append(
                {
                    "code": "max_parallel_capped",
                    "message": (
                        f"--max-parallel was capped to 5 (you requested "
                        f"{_user_max_parallel}). Effective concurrency may be "
                        f"raised to len(batches)+1 to avoid the BTA "
                        f"aggregator-deadlock condition; see "
                        f"breakdown_then_aggregate_inferencer.py:350-360."
                    ),
                }
            )
        self._adapter = StreamBridgeAdapter()
        self._breakdown_factory = breakdown_factory
        self._worker_factory_override = worker_factory_override
        self._aggregator_factory = aggregator_factory
        self._conflict_pairs = conflict_pairs or []
        self._hypothesis_slots = hypothesis_slots or {}
        self._hypothesis_metadata = list(hypothesis_metadata or [])
        self._hypothesis_phase = dict(hypothesis_phase or {})
        self._hub_id = hub_id
        # Per-call observability callbacks; populated by ``run()``.
        # Default no-ops keep the bridge UI-agnostic.
        self._on_batches_grouped: Optional[Callable[[list[Batch]], Awaitable[None]]] = (
            None
        )
        self._on_batch_status: Optional[Callable[..., Awaitable[None]]] = None

        # Workspace under session-tasks. --reuse-task resumes into an
        # existing one; otherwise mint a fresh ts+hex tail.
        if workspace_path is not None:
            self._workspace = workspace_path
        elif reuse_task:
            self._workspace = session_tasks_dir / reuse_task
            if not self._workspace.is_dir():
                # Treat as a relative path; let the bridge fail loudly later
                # if the resume target genuinely doesn't exist.
                self._workspace = session_tasks_dir / reuse_task
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            tail = secrets.token_hex(3)
            self._workspace = session_tasks_dir / f"implhyp_{ts}_{tail}"
        self._workspace.mkdir(parents=True, exist_ok=True)
        for sub in (
            "outputs",
            "results",
            "logs",
            "batches",
            "_runtime/inferencer_cache",
            "_runtime/tmp_output_files",
        ):
            (self._workspace / sub).mkdir(parents=True, exist_ok=True)

        # Persisted batch records — populated as workers complete.
        self._batches: list[Batch] = []
        self._validation_warnings: list[str] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def notices(self) -> list[dict[str, str]]:
        """Human-readable notices about how the bridge interpreted its
        construction args (e.g., silent max_parallel cap). Tool executor
        propagates these onto the outer task_status chip metadata so
        the frontend can render them."""
        return list(self._notices)

    @property
    def token_stream(self) -> StreamBridgeAdapter:
        return self._adapter

    @property
    def batches(self) -> list[Batch]:
        return list(self._batches)

    @property
    def validation_warnings(self) -> list[str]:
        return list(self._validation_warnings)

    # ------------------------------------------------------------------
    # Inferencer factories — mirror ResearchProposeBridge pattern
    # ------------------------------------------------------------------

    def _create_llm_inferencer(self, role: str) -> InferencerBase:
        """Build a CLI-backed LLM inferencer for the given role.

        Mirror of :meth:`ResearchProposeBridge._create_inferencer`.
        """
        cache_folder = str(self._workspace / "_runtime" / "inferencer_cache")
        tmp_dir = str(self._workspace / "_runtime" / "tmp_output_files")
        inf_id = f"{self._base_inferencer_type}_{role}"
        if self._base_inferencer_type == "devmate_cli":
            from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (  # @manual
                DevmateCliInferencer,
            )

            kwargs: dict[str, Any] = {
                "id": inf_id,
                "cache_folder": cache_folder,
                "large_arg_temp_dir": tmp_dir,
            }
            if self._model:
                kwargs["model_name"] = self._model
            if self._session_context.get("session_root_path"):
                kwargs["root_folder"] = self._session_context["session_root_path"]
            return DevmateCliInferencer(**kwargs)
        from agent_foundation.common.inferencers.agentic_inferencers import (  # @manual
            MetamateSDKInferencer,
        )

        return MetamateSDKInferencer(
            model_id=self._model or "",
            cache_folder=cache_folder,
            id=inf_id,
        )

    def _build_dual(
        self,
        role: str,
        template_space: str,
        template_version: str = "",
        template_variables: dict[str, str] | None = None,
    ) -> DualInferencer:
        """Initial-only DualInferencer with the requested template space.

        ``template_variables`` selects per-purpose variant bodies under
        ``_variables/<slot>/<variant>/default.jinja2`` — e.g.
        ``{"task_preamble": "@implementation_report"}`` for the generic
        ``aggregation/`` template space.
        """
        inf = self._create_llm_inferencer(role)
        templates_dir = _get_templates_dir()
        prompt_tm = TemplateManager(
            templates=str(templates_dir),
            active_template_root_space=template_space,
            enable_templated_feed=True,
            predefined_variables=True,
            template_version=template_version,
        )
        dual = DualInferencer(
            base_inferencer=inf,
            review_inferencer=inf,
            consensus_config=ConsensusConfig(max_iterations=0),
            prompt_formatter=prompt_tm,
            initial_prompt="initial",
            review_prompt="review",
            followup_prompt="followup",
            id=f"{template_space.title().replace('_', '')}_{role}",
        )
        if template_variables:
            dual.template_variables = dict(template_variables)
        return dual

    def _build_breakdown(self) -> InferencerBase:
        if self._breakdown_factory is not None:
            return self._breakdown_factory()
        return self._build_dual(
            role="batch_grouper",
            template_space="hypothesis_batch_grouper",
        )

    def _build_per_batch_pti(self, batch: Batch) -> InferencerBase:
        if self._worker_factory_override is not None:
            return self._worker_factory_override(batch)
        # Plan + implement DualInferencers using the existing
        # `implementation` template space (which already prescribes the
        # enable_<name> feature flag contract).
        plan_dual = self._build_dual(
            role=f"plan_{batch.batch_id}",
            template_space="implementation",
        )
        impl_dual = self._build_dual(
            role=f"impl_{batch.batch_id}",
            template_space="implementation",
        )
        return PlanThenImplementInferencer(
            planner_inferencer=plan_dual,
            executor_inferencer=impl_dual,
            workspace_path=str(self._workspace / "batches" / batch.batch_id),
            resume_with_saved_results=True,
        )

    def _build_aggregator(self) -> InferencerBase:
        if self._aggregator_factory is not None:
            return self._aggregator_factory()
        # Generic aggregation template + implementation_report variant.
        # Variant selection is via `template_version` — the working
        # mechanism is FileBasedVariableManager's Phase 2 versioned-folder
        # fallback (file_based.py:691-697) which resolves
        # `{{ task_preamble }}` against
        # `_variables/task_preamble/implementation_report/default.jinja2`.
        # `template_variables` (the {"task_preamble": "@variant"} dict) is
        # an aspirational API: load_variable() is referenced from
        # inferencer_base.py:491 but never implemented.
        return self._build_dual(
            role="implementation_report",
            template_space="aggregation",
            template_version="implementation_report",
        )

    # ------------------------------------------------------------------
    # Worker factory passed to BTA
    # ------------------------------------------------------------------

    def _worker_factory(self, sub_query: Any, index: int) -> InferencerBase:
        """BTA dispatches one worker per batch.

        When ``on_batch_status`` is set (Layer 0b/c — UI-agnostic
        observability hook), patch the per-batch PTI's ``ainfer`` method
        in-place to fire ``running`` before the call and ``completed`` /
        ``error`` after. We do an instance-level method swap (not a
        wrapper subclass) so BTA's downstream attribute access on the
        worker (``_workspace``, ``name``, ``resolve_output_path``,
        Resumable interface) continues to hit the real PTI without any
        proxy boilerplate. ``InferencerBase`` is a legacy ``@attrs``
        class (no slots), so instance attribute assignment shadows the
        class method as expected.

        Phase A6: BTA rebinds ``worker._workspace`` to
        ``self._workspace.child(f"worker_{i}")`` AFTER this factory
        runs (see breakdown_then_aggregate_inferencer.py:782). Mirror
        BTA's naming on ``batch.workspace`` so the sidecar callback at
        tool_executor.py:3699+ records the *actual* on-disk worker
        directory rather than a phantom ``batches/<bid>/`` path that
        BTA never wrote to.
        """
        batch = sub_query if isinstance(sub_query, Batch) else self._batches[index]
        # Match BTA's worker workspace naming (set just-after by BTA).
        batch_workspace = self._workspace / f"worker_{index}"
        batch_workspace.mkdir(parents=True, exist_ok=True)
        batch.workspace = str(batch_workspace)
        inner = self._build_per_batch_pti(batch)
        cb = self._on_batch_status
        if cb is None:
            return inner
        original_ainfer = inner.ainfer

        async def _wrapped_ainfer(*args: Any, **kwargs: Any) -> Any:
            try:
                await cb(batch, "running")
            except Exception as e:  # pragma: no cover — best-effort
                logger.warning(
                    "on_batch_status(running) failed for batch=%s: %s",
                    batch.batch_id,
                    e,
                )
            try:
                result = await original_ainfer(*args, **kwargs)
            except Exception as e:
                try:
                    await cb(batch, "error", error_message=str(e))
                except Exception as cb_err:  # pragma: no cover
                    logger.warning(
                        "on_batch_status(error) failed for batch=%s: %s",
                        batch.batch_id,
                        cb_err,
                    )
                raise
            try:
                await cb(batch, "completed")
            except Exception as e:  # pragma: no cover — best-effort
                logger.warning(
                    "on_batch_status(completed) failed for batch=%s: %s",
                    batch.batch_id,
                    e,
                )
            return result

        inner.ainfer = _wrapped_ainfer  # type: ignore[assignment]
        return inner

    # ------------------------------------------------------------------
    # Combined breakdown + validation pipeline
    # ------------------------------------------------------------------

    async def _resolve_batches(self) -> list[Batch]:
        """Run the breakdown LLM → parse → validate → return batches.

        Resume path: if a prior successful grouper run wrote
        ``<workspace>/results/batch_plan.json``, reuse those batches
        instead of re-running the LLM. Keeps Resume cheap and ensures
        ``batch_id`` stability across runs (per-batch PTI checkpoints
        are keyed on ``batch_id``).

        For tests, ``breakdown_factory`` can return a pre-baked inferencer
        that emits a known JSON string (no real LLM call needed).
        """
        # Resume path — cached batches from a prior successful grouper run.
        cache_path = self._workspace / "results" / "batch_plan.json"
        if cache_path.is_file():
            try:
                doc = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_raw = doc.get("batches") if isinstance(doc, dict) else None
                if isinstance(cached_raw, list):
                    cached = [
                        Batch(
                            batch_id=b["batch_id"],
                            items=list(b.get("items") or []),
                            rationale=b.get("rationale", ""),
                        )
                        for b in cached_raw
                        if isinstance(b, dict) and b.get("batch_id") and b.get("items")
                    ]
                    if cached:
                        logger.info(
                            "Resuming from cached batches at %s (%d batches)",
                            cache_path,
                            len(cached),
                        )
                        # Restore validation warnings if present so the
                        # human summary still flags any soft-cap repairs.
                        cached_warnings = (
                            doc.get("validation_warnings")
                            if isinstance(doc, dict)
                            else None
                        )
                        if isinstance(cached_warnings, list):
                            self._validation_warnings = list(cached_warnings)
                        return cached
            except (OSError, ValueError, KeyError) as e:
                logger.warning(
                    "Cached batch_plan.json invalid (%s); re-running grouper",
                    e,
                )

        breakdown = self._build_breakdown()
        # Render the input as a structured list the prompt can consume.
        # We pass selected_ids; richer hypothesis bodies (parsed from
        # plan_text) flow in via inference_config.
        formatted_input = self._format_grouper_input()
        try:
            raw = await breakdown.ainfer(
                formatted_input,
                inference_config=self._inference_config(),
            )
        except Exception as e:  # pragma: no cover — defensive
            logger.exception("breakdown LLM call failed: %s", e)
            return self._fallback_singleton_batches()
        valid_ids = set(self._selected_ids)
        parsed = parse_batches_response(raw, valid_ids)
        if not parsed:
            logger.warning(
                "breakdown returned no usable batches; falling back to singletons"
            )
            return self._fallback_singleton_batches()
        validated, warnings = validate_and_repair_batches(
            parsed, self._selected_ids, self._max_batch_size
        )
        self._validation_warnings = warnings
        if warnings:
            for w in warnings:
                logger.info("[grouper validator] %s", w)
        return validated

    def _format_grouper_input(self) -> str:
        """Render the selected hypotheses as a markdown table the grouper
        LLM can read directly. Falls back to bare bullets when no
        structured metadata was provided (back-compat for tests / CLI
        callers that pass only ``selected_ids``).
        """
        lines: list[str] = ["# Selected hypotheses\n"]
        if self._hypothesis_metadata:
            # Index metadata by id for O(1) lookup.
            meta_by_id = {
                str(m.get("id", "")): m
                for m in self._hypothesis_metadata
                if m and m.get("id")
            }
            lines.append("| ID | Phase | Title | Slots | Description |")
            lines.append("|---|---|---|---|---|")
            for hid in self._selected_ids:
                m = meta_by_id.get(hid, {}) or {}
                phase = self._hypothesis_phase.get(hid, "") or m.get("phase", "")
                title = (m.get("title") or "").replace("|", "\\|").replace("\n", " ")
                slots = ", ".join(m.get("slots") or [])
                desc = (
                    (m.get("description") or "").replace("|", "\\|").replace("\n", " ")
                )
                # Cap description so the prompt doesn't blow up for verbose entries.
                if len(desc) > 240:
                    desc = desc[:237] + "..."
                lines.append(f"| **{hid}** | {phase} | {title} | {slots} | {desc} |")
        else:
            for hid in self._selected_ids:
                lines.append(f"- **{hid}**")
        if self._plan_text:
            lines.append("\n# Plan context\n")
            lines.append(self._plan_text)
        return "\n".join(lines)

    def _inference_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "max_batch_size": self._max_batch_size,
        }
        if self._workflow_target_path:
            cfg["workflow_target_path"] = self._workflow_target_path
        if self._conflict_pairs:
            cfg["conflict_pairs"] = self._conflict_pairs
        if self._hypothesis_slots:
            cfg["hypothesis_slots"] = self._hypothesis_slots
        if self._hypothesis_phase:
            cfg["hypothesis_phase"] = self._hypothesis_phase
        for k, v in self._session_context.items():
            if isinstance(v, (str, int, float, bool)):
                cfg[k] = v
        return cfg

    def _fallback_singleton_batches(self) -> list[Batch]:
        """When the LLM fails, every selected hypothesis becomes its own
        batch. Always-correct fallback; just expensive (one PTI per H)."""
        return [
            Batch(
                batch_id=f"singleton_{i}_{hid}",
                items=[hid],
                rationale="Fallback singleton (grouper unavailable)",
            )
            for i, hid in enumerate(self._selected_ids)
        ]

    # ------------------------------------------------------------------
    # BTA wiring
    # ------------------------------------------------------------------

    def _format_batch_input(self, batch: Batch) -> str:
        """Render a Batch as the input string the per-batch PTI consumes.

        BTA passes ``predefined_sub_queries[i]`` directly to
        ``worker.ainfer(q, ...)`` (``bta.py:870``). The PTI's
        ``_setup_iteration_workspace`` then writes the input verbatim to
        ``request.txt`` (``plan_then_implement_inferencer.py:625``) — so
        the input MUST be a string. Passing Batch objects (the previous
        attempt) triggered ``TypeError: write() argument must be str,
        not Batch`` because Batch has no ``__str__`` override and
        Python's default repr isn't text the file-write path accepts.

        The Batch object itself is still recovered by
        ``_worker_factory`` via ``self._batches[index]`` (the existing
        fallback at ``:621``) so the worker can configure its workspace
        + observability against the real batch metadata. Only the
        worker's INPUT STRING travels through BTA's sub_query plumbing.
        """
        lines: list[str] = [
            f"# Implement batch {batch.batch_id}",
            f"Hypotheses to implement: {', '.join(batch.items)}",
        ]
        rationale = (getattr(batch, "rationale", "") or "").strip()
        if rationale:
            lines.append("")
            lines.append(f"Grouping rationale: {rationale}")
        if self._plan_text:
            lines.append("")
            lines.append("## Plan context")
            lines.append(self._plan_text)
        return "\n".join(lines)

    def _build_outer_bta(self) -> BreakdownThenAggregateInferencer:
        """Construct the outer BTA with pre-resolved batches.

        Pre-resolved batches are passed via ``predefined_sub_queries`` so
        BTA's ``_ainfer`` takes the predefined branch
        (``breakdown_then_aggregate_inferencer.py:1684-1728``) and skips
        the breakdown LLM call entirely. The previous implementation tried
        a "cache write to ``breakdown_result.json``" trick but that was
        silently broken in three ways: wrong file format (BTA expects
        ``{"sub_queries": [...]}`` not a bare list), wrong precondition
        (BTA's ``_load_breakdown_checkpoint`` short-circuits unless
        ``resume_with_saved_results=True``), and wrong code path (cache
        loads only on resume, not first run). The cache trick triggered
        ``ValueError: breakdown_inferencer must be set when
        predefined_sub_queries is None`` at ``bta.py:1732`` for every
        fresh implhyp run.

        Callers must invoke ``_resolve_batches()`` BEFORE
        ``_build_outer_bta()`` so ``self._batches`` is populated
        (``run()`` already does — see the call site below).

        ``max_concurrency`` interacts with ``aggregator_inferencer`` such
        that the Mth worker's downstream-aggregator slot needs an (M+1)th
        semaphore slot — see
        ``breakdown_then_aggregate_inferencer.py:350-360``. With M ==
        ``self._max_parallel`` and N batches where N ≥ M, the BTA
        deadlocks. Honor the user's ``max_parallel`` as the soft
        steady-state throttle but always raise to ``len(self._batches) +
        1`` so the aggregator slot is always available.
        """
        safe_concurrency = max(self._max_parallel, len(self._batches) + 1)
        # Pass pre-formatted INPUT STRINGS (not Batch objects). The
        # worker_factory recovers the Batch via self._batches[index]
        # (see _worker_factory at line ~654) — that path is preserved
        # by the existing isinstance check `if isinstance(sub_query,
        # Batch) else self._batches[index]`. With strings here, we
        # take the else-branch and use the index lookup.
        return BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            predefined_sub_queries=[self._format_batch_input(b) for b in self._batches],
            worker_factory=self._worker_factory,
            aggregator_inferencer=self._build_aggregator(),
            checkpoint_dir=str(self._workspace / "checkpoints" / "bta"),
            workspace_root=str(self._workspace),
            max_concurrency=safe_concurrency,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        request: str = "",
        *,
        on_batches_grouped: Optional[Callable[[list[Batch]], Awaitable[None]]] = None,
        on_batch_status: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> str:
        """Resolve batches → run outer BTA → write report → return summary.

        Returns a 2000-char-truncated human summary suitable for
        appending to the conversation. Full structured result lives at
        ``<workspace>/results/implementation_summary.json``.

        Optional UI-agnostic observability callbacks (Layer 0b/c —
        plan: implement-selected-auto-mode-and-hub-nesting.md):

        * ``on_batches_grouped(batches)`` — called once right after the
          LLM grouper returns the validated batch list. Caller can emit
          one ``task_status: queued`` per batch so chips appear in the
          Progress tab BEFORE workers spawn.
        * ``on_batch_status(batch, status, **kw)`` — called by the
          per-batch worker wrapper at start (``status="running"``) and
          after the worker returns (``status="completed"`` or
          ``status="error"`` with ``error_message=str``). Caller emits
          per-batch ``task_status`` AND writes to
          ``hub_<mid>_implementations.json`` so downstream gates see
          the implementation evidence.

        Both callbacks are optional; defaults are no-op (back-compat).
        """
        if not self._selected_ids:
            return (
                "/implement-hypothesis: no hypotheses selected. "
                "Pass --select H1,H17,..."
            )
        # Stash callbacks for the worker_factory closure (which BTA
        # invokes lazily, after we return from this stack frame).
        self._on_batches_grouped = on_batches_grouped
        self._on_batch_status = on_batch_status
        batches = await self._resolve_batches()
        if not batches:
            return "/implement-hypothesis: no batches resolved (empty selection?)."
        self._batches = batches

        # Fire the grouper-complete hook so the executor can emit one
        # task_status: queued chip per batch (BEFORE workers spawn).
        if on_batches_grouped is not None:
            try:
                await on_batches_grouped(batches)
            except Exception as e:  # pragma: no cover — best-effort
                logger.warning("on_batches_grouped failed (non-fatal): %s", e)

        # Pre-stage the batch plan so a watcher sees the planned set
        # BEFORE workers spawn.
        plan_path = self._workspace / "results" / "batch_plan.json"
        plan_path.write_text(
            json.dumps(
                {
                    "batches": [b.to_dict() for b in batches],
                    "validation_warnings": self._validation_warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # _build_outer_bta now passes batches via predefined_sub_queries;
        # the prior breakdown_result.json cache write was broken (wrong
        # format + wrong precondition) and is no longer needed. The BTA
        # takes the predefined branch directly from the constructor arg.
        bta = self._build_outer_bta()

        error_message: Optional[str] = None
        try:
            await bta.ainfer(request, inference_config=self._inference_config())
            status = "completed"
        except Exception as e:  # pragma: no cover — defensive
            logger.exception(
                "ImplementHypothesisBridge.run failed (hub_id=%s, batches=%s): %s",
                self._hub_id,
                [b.batch_id for b in batches],
                e,
            )
            status = "error"
            error_message = f"{type(e).__name__}: {e}"

            # Phase D1 (v7): persist a structured spawn-error artifact so
            # the failure is post-mortem-able from disk without needing
            # terminal scroll-back. Catches ANY future failure between
            # grouper-success and per-batch-completion (rate-limit, auth
            # refresh, OOM, network blip, etc.).
            try:
                import traceback as _traceback

                spawn_err = self._workspace / "results" / "spawn_error.json"
                spawn_err.parent.mkdir(parents=True, exist_ok=True)
                spawn_err.write_text(
                    json.dumps(
                        {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": _traceback.format_exc(),
                            "hub_id": self._hub_id,
                            "batch_ids": [b.batch_id for b in batches],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as persist_err:  # pragma: no cover — best-effort
                logger.warning("spawn_error.json persist failed: %s", persist_err)

            # Phase D1 (v7): for each batch that hadn't yet entered the
            # worker phase (still queued from `on_batches_grouped`),
            # synthesize an `error` transition so the per-batch chips
            # don't hang in `queued` forever. Best-effort; failures
            # here must not mask the original exception.
            cb = self._on_batch_status
            if cb is not None:
                for b in batches:
                    try:
                        await cb(b, "error", error_message=error_message)
                    except Exception as cb_err:  # pragma: no cover
                        logger.warning(
                            "spawn-error chip transition failed for batch=%s: %s",
                            b.batch_id,
                            cb_err,
                        )

        summary_path = self._workspace / "results" / "implementation_summary.json"
        summary: dict[str, Any] = {
            "status": status,
            "workspace": str(self._workspace),
            "batches": [b.to_dict() for b in batches],
            "validation_warnings": self._validation_warnings,
            "selected_ids": self._selected_ids,
        }
        # Phase D1 (v7): persist structured error_message in the summary
        # so the tool_executor's terminal-chip emit (Phase D2) can
        # surface it on the outer chip's metadata for the View-error
        # dialog without re-parsing the spawn_error.json.
        if error_message:
            summary["error_message"] = error_message
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return self._human_summary(summary)

    def _human_summary(self, summary: dict[str, Any]) -> str:
        lines: list[str] = [
            f"/implement-hypothesis {summary['status']} — "
            f"{len(summary['batches'])} batches, "
            f"{len(summary['selected_ids'])} hypotheses",
            f"  workspace: {summary['workspace']}",
        ]
        for b in summary["batches"][:10]:
            lines.append(f"  • {b['batch_id']}: items={b['items']}")
        if len(summary["batches"]) > 10:
            lines.append(f"  … and {len(summary['batches']) - 10} more")
        if summary["validation_warnings"]:
            lines.append(
                f"  ⚠ {len(summary['validation_warnings'])} validation warnings; "
                "see batch_plan.json"
            )
        text = "\n".join(lines)
        return text[:2000]
