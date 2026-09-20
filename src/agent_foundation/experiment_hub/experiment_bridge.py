# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""``ExperimentBridge`` — orchestrates the ``/experiment`` slash command.

Composition (per the plan at ``/home/zgchen/.claude/plans/humming-tinkering-wirth.md`` §2):

    outer BTA:
      breakdown   = PlanThenImplementInferencer
                       planner   = DualInferencer (combo planner)
                       executor  = DualInferencer (per-combo gin/script gen)
      worker_fac  = make_tool_chain([launch_tool, monitor_tool, analyze_dual])
                     where launch_tool = ToolAsInferencer(rankevolve_train),
                           monitor_tool = ToolAsInferencer(rankevolve_monitor),
                           analyze_dual = DualInferencer(combo_analysis)
      aggregator  = None for Phase 2 (the inner aggregator BTA lands in Phase 3)

Workspace convention: ``<session_dir>/tasks/exp_<ts>_<hex>/`` to align with the
chip / queue / persistence machinery used elsewhere. Per-combo workspaces nest
at ``<exp_workspace>/combos/<combo_id>/``.

This bridge is **deterministic plumbing** — the LLM-template body lives in
``src/resources/prompt_templates/{experiment_plan,experiment_implement,combo_analysis}/``
and is intended to be iterated against real LLM runs. The shape & invariants
the bridge enforces are stable; the prose inside the templates can evolve
without changing this file.

See also:
    :class:`rankevolve.src.server.research_propose_bridge.ResearchProposeBridge`
        — the closest reference. Shares the workspace / template-resolution /
        DualInferencer-construction patterns. Mirror-copied where appropriate.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
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
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.tool_inferencers import (
    make_tool_chain,
    ToolAsInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.experiment_hub.stream_bridge import StreamBridgeAdapter
from rich_python_utils.string_utils.formatting.template_manager.template_manager import (  # @manual
    TemplateManager,
)


logger: logging.Logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configurable defaults — kept top-of-module so they're easy to find/tune.
# ─────────────────────────────────────────────────────────────────────────────

# Default outer-BTA worker semaphore. Local GPU contention is the binding
# constraint; 2 concurrent training runs is the safe v1 number.
_DEFAULT_MAX_CONCURRENCY: int = 2

# Default per-monitor max-wait. Matches `rankevolve_monitor`'s 6h default;
# raises early on training that hangs forever rather than blocking the
# aggregator gate indefinitely.
_DEFAULT_MAX_WAIT_SECONDS: float = 21600.0

# Allowlist of ${KEY} substitutions the per-combo workers expect. Keeps
# the template-binding contract explicit so a typo surfaces as a clear
# `CmdHelperError` rather than as silent argv weirdness.
_LAUNCH_TOOL_KEYS: frozenset[str] = frozenset(
    {"SCRIPT", "LAUNCH", "FLAGS", "EXP", "WS", "ROOT"}
)
_MONITOR_TOOL_KEYS: frozenset[str] = frozenset({"WS", "FLOW_URI", "MAST_JOB"})


# ─────────────────────────────────────────────────────────────────────────────
# Plan parsing
# ─────────────────────────────────────────────────────────────────────────────


_HYPOTHESIS_HEADER_RE: re.Pattern[str] = re.compile(
    r"^#+\s*(H\d+[\w_-]*)\b", re.MULTILINE
)


def parse_hypothesis_ids(text: str) -> list[str]:
    """Extract Hxx hypothesis IDs from a markdown plan.

    Recognizes any H-prefixed alphanumeric token at the start of a markdown
    heading line (``# H1``, ``## H17``, ``### H4_BROKEN``). Order-preserving
    and de-duplicated. Returns ``[]`` if no hypotheses are mentioned (caller
    decides whether that's an error).

    Lifted from the same heuristic ``tool_executor._build_hypothesis_task_query``
    uses (kept in sync manually — tests cover the contract).
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _HYPOTHESIS_HEADER_RE.finditer(text):
        hid = m.group(1)
        if hid in seen:
            continue
        seen.add(hid)
        out.append(hid)
    return out


def parse_combos_arg(combos_arg: str) -> list[list[str]]:
    """Parse ``--combos H1;H17,H8;H56_BASELINE`` into ``[["H1"],["H17","H8"],["H56_BASELINE"]]``.

    Empty input → ``[]`` (caller falls back to default-combo generation).
    Whitespace and empty entries are tolerated.
    """
    out: list[list[str]] = []
    for raw in combos_arg.split(";"):
        members = [m.strip() for m in raw.split(",") if m.strip()]
        if members:
            out.append(members)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Combo dataclass
# ─────────────────────────────────────────────────────────────────────────────


class Combo:
    """Per-combo planning record. Plain class (not attrs) so callers
    that don't depend on agentic_foundation can introspect it freely."""

    __slots__ = (
        "combo_id",
        "items",
        "config_name",
        "script_path",
        "launch_path",
        "workspace",
    )

    def __init__(
        self,
        combo_id: str,
        items: list[str],
        config_name: str = "",
        script_path: Optional[str] = None,
        launch_path: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> None:
        self.combo_id = combo_id
        self.items = items
        self.config_name = config_name or "_".join(items)
        self.script_path = script_path
        self.launch_path = launch_path
        self.workspace = workspace

    def to_dict(self) -> dict[str, Any]:
        return {
            "combo_id": self.combo_id,
            "items": self.items,
            "config_name": self.config_name,
            "script_path": self.script_path,
            "launch_path": self.launch_path,
            "workspace": self.workspace,
        }

    def __repr__(self) -> str:
        return f"Combo({self.combo_id!r}, items={self.items!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Bridge
# ─────────────────────────────────────────────────────────────────────────────


def _get_repo_root() -> Path:
    """Walk up looking for ``.sl`` (sapling root). Falls back to file parent."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".sl").is_dir():
            return parent
    return current.parent


# TODO(port): hub prompt_templates not yet vendored — _build_dual/_create_llm_inferencer paths require them at runtime.
def _get_templates_dir() -> Path:
    """Locate ``rankevolve/src/resources/prompt_templates``. Tries
    ``importlib.resources`` (Buck) then a filesystem fallback (dev)."""
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


def parse_experiment_options(args: str) -> tuple[str, dict[str, Any]]:
    """Parse the ``/experiment`` flag tail (the non-flag remainder is the
    plan reference / free text passed to the bridge).

    Recognized flags (additive — anything else falls through into the request):
      --plan <abs-path>           Path to a unified-proposal markdown file
      --select H1,H17,H8          Hypothesis-ID allowlist
      --combos H1;H17,H8;...      Explicit combos; semicolon between combos
      --max-concurrency N         BTA worker semaphore (default 2)
      --rounds N                  Multi-round outer LWI count (Phase 4)
      --reuse-hub <multi_task_id> Resume into an existing hub
      --workspace <abs-path>      Override workspace location
      --workspace-keep-only-final Reap intermediate artifacts post-completion
      --max-wait <seconds>        Per-monitor watchdog timeout
      --base-inferencer <type>    LLM inferencer for planner/analyzer
      --workflow-target-path <p>  Codebase root for ${CODEBASE_ROOT} resolution
      --model <name>              LLM model override
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
        if tok == "--plan":
            _consume("plan")
        elif tok == "--select":
            _consume("select")
        elif tok == "--combos":
            _consume("combos")
        elif tok == "--max-concurrency":
            _consume("max_concurrency")
        elif tok == "--rounds":
            _consume("rounds")
        elif tok == "--reuse-hub":
            _consume("reuse_hub")
        elif tok == "--workspace":
            _consume("workspace")
        elif tok == "--workspace-keep-only-final":
            options["workspace_keep_only_final"] = True
        elif tok == "--max-wait":
            _consume("max_wait")
        elif tok == "--base-inferencer":
            _consume("base_inferencer")
        elif tok == "--workflow-target-path":
            _consume("workflow_target_path")
        elif tok == "--model":
            _consume("model")
        elif tok == "--implement-default":
            options["implement_default"] = True
        elif tok == "--group-by":
            _consume("group_by")
        else:
            request_parts.append(tok)
        i += 1
    return " ".join(request_parts), options


class ExperimentBridge:
    """Bridge for the ``/experiment`` slash command.

    Constructs a :class:`BreakdownThenAggregateInferencer` whose breakdown
    is a combo-planner PTI, whose workers are 3-step LWI tool chains
    (launch → monitor → analyze), and whose aggregator is deferred to
    Phase 3.

    Workspace convention: ``<session_dir>/tasks/exp_<ts>_<hex>/`` —
    matches the chip / queue / persistence convention. Per-combo
    workspaces nest at ``<exp_workspace>/combos/<combo_id>/``.

    Streaming: every nested inferencer writes to
    ``<workspace>/_runtime/inferencer_cache/<id>/stream_*.txt`` — the
    agent service bridge's existing ``WorkspaceStreamTailer`` discovers
    them automatically. No new transport required.
    """

    def __init__(
        self,
        session_tasks_dir: Path,
        plan_text: str,
        selected_ids: Optional[list[str]] = None,
        combos: Optional[list[list[str]]] = None,
        *,
        model: Optional[str] = None,
        base_inferencer_type: str = "devmate_cli",
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        max_wait_seconds: float = _DEFAULT_MAX_WAIT_SECONDS,
        workflow_target_path: str = "",
        workspace_path: Optional[Path] = None,
        session_context: Optional[dict[str, Any]] = None,
        # Phase 3 — when True, build the aggregator inner BTA. When False
        # (the safer default for first-time use), the outer BTA returns
        # per-combo analyses unaggregated and the existing synth-driven
        # accumulated_learnings.md continues to populate the drawer.
        enable_aggregator: bool = False,
        # Phase 4 — multi-round outer LWI wrapper count. Anything > 1
        # turns the outer BTA into one step of an LWI that loops back
        # while the previous round's next_round_recommendations.json
        # signals "continue".
        rounds: int = 1,
        # Allow tests to inject mock tool inferencers without subclassing.
        launch_tool_factory: Optional[Any] = None,
        monitor_tool_factory: Optional[Any] = None,
    ) -> None:
        self._session_tasks_dir = session_tasks_dir
        self._plan_text = plan_text
        self._selected_ids = selected_ids or []
        self._explicit_combos = combos or []
        self._model = model
        self._base_inferencer_type = base_inferencer_type
        self._max_concurrency = max(1, int(max_concurrency))
        self._max_wait_seconds = float(max_wait_seconds)
        self._workflow_target_path = workflow_target_path
        self._session_context: dict[str, Any] = dict(session_context or {})
        self._adapter = StreamBridgeAdapter()
        self._launch_tool_factory = launch_tool_factory
        self._monitor_tool_factory = monitor_tool_factory
        self._enable_aggregator = bool(enable_aggregator)
        self._rounds = max(1, int(rounds))

        # Workspace under session-tasks (chip-friendly). Suffix uses both
        # a timestamp (sortability) and a 6-hex random tail (uniqueness if
        # two /experiment calls land in the same wall-clock second).
        if workspace_path is not None:
            self._workspace = workspace_path
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            tail = secrets.token_hex(3)
            self._workspace = session_tasks_dir / f"exp_{ts}_{tail}"
        self._workspace.mkdir(parents=True, exist_ok=True)
        for sub in (
            "outputs",
            "results",
            "logs",
            "combos",
            "_runtime/inferencer_cache",
            "_runtime/tmp_output_files",
        ):
            (self._workspace / sub).mkdir(parents=True, exist_ok=True)

        # Persisted per-combo records — populated as workers complete.
        self._combos: list[Combo] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def token_stream(self) -> StreamBridgeAdapter:
        return self._adapter

    @property
    def combos(self) -> list[Combo]:
        return list(self._combos)

    # ------------------------------------------------------------------
    # Plan + combo resolution
    # ------------------------------------------------------------------

    def resolve_combos(self) -> list[Combo]:
        """Pick the combos to run.

        Priority order:
          1. ``self._explicit_combos`` (from ``--combos`` flag) wins outright.
          2. Else, if ``self._selected_ids`` is non-empty, treat each as a
             single-hypothesis combo (the simplest "default" mode — no
             slot-disjoint scoring needed when the user supplied an explicit
             set).
          3. Else, parse hypothesis IDs out of ``self._plan_text`` and
             treat each as a single-hypothesis combo.

        Phase 2 deliberately keeps default-combo generation simple. The
        slot-disjoint enumeration via ``learnings_generator._generate_combos``
        kicks in once Phase 3's aggregator has real combo results to
        feed back into the suggester (see plan §3.3).
        """
        if self._explicit_combos:
            members_lists = self._explicit_combos
        elif self._selected_ids:
            members_lists = [[hid] for hid in self._selected_ids]
        else:
            members_lists = [[hid] for hid in parse_hypothesis_ids(self._plan_text)]
        combos: list[Combo] = []
        for idx, members in enumerate(members_lists):
            combo_id = "_".join(members) or f"combo_{idx}"
            combos.append(Combo(combo_id=combo_id, items=members))
        self._combos = combos
        return combos

    # ------------------------------------------------------------------
    # Inferencer factories — mirror ResearchProposeBridge pattern
    # ------------------------------------------------------------------

    def _create_llm_inferencer(self, role: str) -> InferencerBase:
        """Build a CLI-backed LLM inferencer for the given role.

        Mirrors :meth:`ResearchProposeBridge._create_inferencer` so the
        same workspace / cache / logger plumbing applies. Lazy-imports
        the concrete CLI inferencer class to keep the import-time
        footprint small.
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
        # Fallback: metamate SDK (matches ResearchProposeBridge fallback).
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
        max_iterations: int = 0,
        debug_mode: bool = False,
    ) -> DualInferencer:
        """DualInferencer with the requested template space.

        Default (``max_iterations=0``): propose-only — short-circuits the
        review/fix loop so the inferencer just renders the initial
        template once and returns the LLM response. Matches
        :meth:`ResearchProposeBridge._build_dual` for the single-shot
        synthesis path used by ResearchPropose and the original agg path.

        When ``max_iterations >= 1`` (caller opt-in): full propose →
        review → fix loop runs up to N cycles, with consensus early-exit
        when reviewer's `overall_severity` is at most COSMETIC. Used by
        the agg-refresh path which now wants real review for synthesis
        quality. Each iteration is one LLM call; N=5 means worst case
        5x baseline LLM time, but consensus typically stops earlier.

        ``debug_mode`` (caller opt-in): when True, sets
        ``DualInferencer.debug_mode=True`` so DEBUG-level logger events
        (RawBaseResponse, RawReviewResponse, RawFollowupResponse,
        InferenceResponse, Message, ParentChildDebuggableLink) are
        captured in the SessionLogger output. Without this, only INFO-
        level events (InitialPrompt, InitialResponse, ReviewPrompt, …)
        are recorded — the threshold check at ``debuggable.py:784-789``
        filters DEBUG entries by default.

        ``template_variables`` selects per-purpose variant bodies under
        ``_variables/<slot>/<variant>/default.jinja2`` — e.g.
        ``{"task_preamble": "@implementation_report"}`` for the generic
        ``aggregation/`` template space. Forwarded to the underlying
        ``InferencerBase.template_variables`` field per the loader at
        ``inferencer_base._build_template_feed``.
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
            consensus_config=ConsensusConfig(max_iterations=max_iterations),
            prompt_formatter=prompt_tm,
            initial_prompt="initial",
            review_prompt="review",
            followup_prompt="followup",
            # Match the de-facto convention used by every aggregation/plan/
            # implementation/proposal template (`{{ main_response }}`) and
            # the reference `dual_inferencer_bridge.py:528,544`. Without
            # this override the framework default `"proposal"` keys the
            # feed dict against a name no template references → review/
            # followup prompts render an empty `<ProposedAggregation>`.
            placeholder_proposal="main_response",
            id=f"{template_space.title().replace('_', '')}_{role}",
            debug_mode=debug_mode,
        )
        if template_variables:
            dual.template_variables = dict(template_variables)
        return dual

    # ------------------------------------------------------------------
    # Tool factories (per-combo workers)
    # ------------------------------------------------------------------

    def _build_launch_tool(self, combo: Combo) -> ToolAsInferencer:
        """``rankevolve_train`` ToolAsInferencer for one combo."""
        if self._launch_tool_factory is not None:
            return self._launch_tool_factory(combo)
        return ToolAsInferencer(
            tool_name=f"launch_{combo.combo_id}",
            command=["rankevolve_train"],
            args_template=[
                "--script-path",
                "${SCRIPT}",
                "--launch-json",
                "${LAUNCH}",
                "--enable-flags",
                "${FLAGS}",
                "--experiment-name",
                "${EXP}",
                "--workspace",
                "${WS}",
                "--workflow-target-path",
                "${ROOT}",
            ],
            allowed_binaries=frozenset({"rankevolve_train"}),
            cache_folder=str(self._workspace / "_runtime" / "inferencer_cache"),
        )

    def _build_monitor_tool(self, combo: Combo) -> ToolAsInferencer:
        """``rankevolve_monitor`` ToolAsInferencer for one combo."""
        if self._monitor_tool_factory is not None:
            return self._monitor_tool_factory(combo)
        return ToolAsInferencer(
            tool_name=f"monitor_{combo.combo_id}",
            command=["rankevolve_monitor"],
            args_template=[
                "--workspace",
                "${WS}",
                "--max-wait",
                str(int(self._max_wait_seconds)),
            ],
            allowed_binaries=frozenset({"rankevolve_monitor"}),
            cache_folder=str(self._workspace / "_runtime" / "inferencer_cache"),
        )

    def _build_analyze_dual(self, combo: Combo) -> DualInferencer:
        """Per-combo analysis DualInferencer.

        Renders ``prompt_templates/combo_analysis/initial.jinja2`` against
        the launch + monitor outputs and the parsed epoch trajectory.
        Output is the per-combo markdown matching the synth schema (see
        plan §2.3) — written by the worker chain to
        ``<combo_workspace>/analysis/combo_<id>.md``.
        """
        return self._build_dual(
            role=f"analyze_{combo.combo_id}",
            template_space="combo_analysis",
        )

    def _worker_factory(self, sub_query: Any, index: int) -> InferencerBase:
        """BTA invokes this to construct one worker per combo.

        ``sub_query`` here is the Combo instance handed back by the
        breakdown PTI (BTA passes whatever the breakdown emitted; we
        bypass an LLM-based breakdown for v1 by pre-computing combos
        from ``--combos`` / ``--select`` / plan parsing in
        :meth:`resolve_combos`).
        """
        combo = sub_query if isinstance(sub_query, Combo) else self._combos[index]
        # Per-combo workspace nests under the bridge workspace; the LWI
        # uses it for checkpoint files and per-step in-progress markers.
        combo_workspace = self._workspace / "combos" / combo.combo_id
        combo_workspace.mkdir(parents=True, exist_ok=True)
        combo.workspace = str(combo_workspace)

        chain = make_tool_chain(
            name=f"combo_{combo.combo_id}",
            tools=[
                self._build_launch_tool(combo),
                self._build_monitor_tool(combo),
                self._build_analyze_dual(combo),
            ],
            workspace_path=str(combo_workspace),
            state_threading="independent",
        )
        return chain

    # ------------------------------------------------------------------
    # BTA wiring
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase 3 — aggregator inner BTA
    # ------------------------------------------------------------------

    def _build_aggregator_inner_bta(self) -> BreakdownThenAggregateInferencer:
        """Inner aggregator BTA per plan §3.1.

        Topology:
          breakdown    = DualInferencer(aggregator_breakdown)
          worker_fact  = dispatch by recommendation.kind
                          ("research"     → ResearchProposeBridge worker)
                          ("investigate"  → DualInferencer(search_recommendation_investigation))
          aggregator   = DualInferencer(aggregator_final) → next_round_recommendations.json

        The breakdown's output (a JSON envelope) seeds the worker queue:
        each ``search_recommendations[i]`` becomes one sub-query; the
        worker factory looks at ``sub_query.kind`` to pick a worker class.
        """
        breakdown_inf = self._build_dual(
            role="aggregator_breakdown",
            template_space="aggregator_breakdown",
        )
        # Latent-bug fix: template_space="aggregator_final" never existed
        # on disk. Repoint to the generic aggregation template + the
        # next_round_recommendations variant, which holds the (formerly
        # in-`aggregation/main/initial.jinja2`) body.
        # Variant selection: pass template_version (NOT template_variables —
        # that's an aspirational API; see tool_executor _exec_aggregator_only_refresh
        # comment for details). The resolver's Phase 2 versioned-folder
        # fallback (file_based.py:691-697) loads
        # `_variables/task_preamble/next_round_recommendations/default.jinja2`.
        agg_inf = self._build_dual(
            role="aggregator_final",
            template_space="aggregation",
            template_version="next_round_recommendations",
        )

        def _agg_worker_factory(sub_query: Any, index: int) -> InferencerBase:
            kind = (
                sub_query.get("kind", "investigate")
                if isinstance(sub_query, dict)
                else "investigate"
            )
            # `kind == "research"` is a richer dive — defer to the existing
            # ResearchProposeBridge composition. For now the v1 path uses
            # the same DualInferencer for both kinds; the worker factory
            # is the right seam to specialize when ResearchProposeBridge
            # is wired in (plan §3.1, deferred follow-up).
            _ = kind
            return self._build_dual(
                role=f"agg_worker_{index}",
                template_space="search_recommendation_investigation",
            )

        return BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown_inf,
            worker_factory=_agg_worker_factory,
            aggregator_inferencer=agg_inf,
            checkpoint_dir=str(self._workspace / "checkpoints" / "agg_bta"),
            workspace_root=str(self._workspace / "aggregator"),
            # Inner workers are light search/research — higher concurrency
            # is safe (no GPU contention).
            max_concurrency=4,
        )

    def _build_outer_bta(self) -> BreakdownThenAggregateInferencer:
        """Construct the outer BTA. Aggregator wiring is gated on
        ``self._enable_aggregator`` so Phase 2-only deployments don't
        accidentally trigger Phase 3 LLM cost.

        Pre-resolved combos are passed via ``predefined_sub_queries`` so
        BTA's ``_ainfer`` takes the predefined branch and skips the
        breakdown LLM call entirely. Caller must invoke
        ``resolve_combos()`` BEFORE this so ``self._combos`` is populated
        (``run()`` already does — see line 792).

        Sister-fix to the same broken cache trick that wedged
        ``ImplementHypothesisBridge``: writing
        ``checkpoints/bta/breakdown_result.json`` with a bare list never
        worked (BTA expects ``{"sub_queries": [...]}`` and only loads on
        resume). Pass ``predefined_sub_queries`` directly.
        """
        aggregator = (
            self._build_aggregator_inner_bta() if self._enable_aggregator else None
        )
        # Pass per-combo INPUT STRINGS (the combo_id as a stable label)
        # — not Combo objects. BTA forwards each entry verbatim to
        # `worker.ainfer(q, ...)`, and downstream tool-chain tools treat
        # the input as text. Passing Combo objects would risk
        # ``TypeError: write() argument must be str, not Combo`` in any
        # tool that writes the input to disk. The Combo object itself is
        # still recovered by ``_worker_factory`` via
        # ``self._combos[index]`` (existing fallback at line 609).
        return BreakdownThenAggregateInferencer(
            breakdown_inferencer=None,
            predefined_sub_queries=[c.combo_id for c in self._combos],
            worker_factory=self._worker_factory,
            aggregator_inferencer=aggregator,
            checkpoint_dir=str(self._workspace / "checkpoints" / "bta"),
            workspace_root=str(self._workspace),
            max_concurrency=self._max_concurrency,
        )

    # ------------------------------------------------------------------
    # Phase 4 — multi-round outer LWI wrapper
    # ------------------------------------------------------------------

    def _wrap_in_outer_lwi(
        self, bta: BreakdownThenAggregateInferencer
    ) -> InferencerBase:
        """When ``rounds > 1``, wrap the outer BTA in a one-step LWI that
        loops back to itself while ``next_round_recommendations.json``
        signals "continue" (and ``max_loop_iterations = rounds - 1``).

        Per-iteration workspace is ``<exp_workspace>/round_N/`` (handed
        out by the LWI's ``iteration_workspace_factory``).
        """
        if self._rounds <= 1:
            return bta

        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
            WorkflowStepConfig,
        )

        def _iter_workspace(base: str, n: int) -> str:
            ws = Path(base) / f"round_{n + 1}"
            ws.mkdir(parents=True, exist_ok=True)
            return str(ws)

        def _loop_condition(state: dict[str, Any], result: Any) -> bool:
            """Continue while the previous round emitted a non-empty
            promoted_combos list. Stops cleanly when the aggregator
            decides we're done.
            """
            recommendations_path = (
                self._workspace
                / f"round_{state.get('iteration', 0) + 1}"
                / "results"
                / "next_round_recommendations.json"
            )
            if not recommendations_path.is_file():
                return False
            try:
                doc = json.loads(recommendations_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            promoted = doc.get("promoted_combos") or []
            return bool(promoted)

        return LinearWorkflowInferencer(
            step_configs=[
                WorkflowStepConfig(
                    name="outer_round",
                    inferencer=bta,
                    input_builder=lambda state: state.get("input", ""),
                    output_extractor=lambda r: r,
                    output_state_key="last_round_result",
                    loop_back_to="outer_round",
                    loop_condition=_loop_condition,
                    max_loop_iterations=self._rounds - 1,
                )
            ],
            workspace_path=str(self._workspace),
            iteration_workspace_factory=_iter_workspace,
            initial_state_factory=lambda inp: {"input": inp, "iteration": 0},
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, request: str = "") -> str:
        """Resolve combos → run outer BTA → write summary → return human str.

        Returns a 2000-char-truncated human summary suitable for
        appending to the conversation. The full structured result is
        written to ``<workspace>/results/experiment_summary.json``.
        """
        combos = self.resolve_combos()
        if not combos:
            return (
                "ExperimentBridge: no combos resolved. Pass --combos "
                "'H1;H17,H8' or --select H1,H17 (or supply a plan with "
                "Hxx headings)."
            )

        # Pre-stage the combo plan so a watcher (UI, log scraper) sees
        # what we're about to launch BEFORE training starts.
        plan_path = self._workspace / "results" / "combo_plan.json"
        plan_path.write_text(
            json.dumps(
                {"combos": [c.to_dict() for c in combos]},
                indent=2,
            ),
            encoding="utf-8",
        )

        # _build_outer_bta now passes combos via predefined_sub_queries;
        # the prior breakdown_result.json cache write was broken (wrong
        # format + wrong precondition) and is no longer needed. The BTA
        # takes the predefined branch directly from the constructor arg.
        bta = self._build_outer_bta()

        # Phase 4: wrap in a one-step LWI when rounds > 1. The wrapper is
        # a no-op when rounds == 1 (returns the BTA unchanged), so the
        # default path stays as cheap as Phase 2.
        runner = self._wrap_in_outer_lwi(bta)

        try:
            # BTA's ainfer signature: (request, inference_config=None).
            # We pass the original request through so workers' templates
            # can reference it via ``inference_config``.
            await runner.ainfer(request, inference_config=self._inference_config())
            status = "completed"
        except Exception as e:  # pragma: no cover — defensive
            logger.exception("ExperimentBridge.run failed: %s", e)
            status = "error"

        summary_path = self._workspace / "results" / "experiment_summary.json"
        summary = {
            "status": status,
            "workspace": str(self._workspace),
            "combos": [c.to_dict() for c in combos],
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return self._human_summary(summary)

    def _inference_config(self) -> dict[str, Any]:
        """Session-context kwargs threaded into worker templates."""
        cfg: dict[str, Any] = {}
        if self._workflow_target_path:
            cfg["workflow_target_path"] = self._workflow_target_path
        for k, v in self._session_context.items():
            if isinstance(v, (str, int, float, bool)):
                cfg[k] = v
        return cfg

    def _human_summary(self, summary: dict[str, Any]) -> str:
        lines: list[str] = [
            f"/experiment {summary['status']} — {len(summary['combos'])} combos",
            f"  workspace: {summary['workspace']}",
        ]
        for c in summary["combos"][:10]:
            lines.append(f"  • {c['combo_id']}: items={c['items']}")
        if len(summary["combos"]) > 10:
            lines.append(f"  … and {len(summary['combos']) - 10} more")
        text = "\n".join(lines)
        return text[:2000]
