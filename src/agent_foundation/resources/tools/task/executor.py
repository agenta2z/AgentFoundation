"""Task executor — runs an arbitrary agent topology from the /task slash command.

Pipeline (10 stages):
    1. Parse arguments + reject conflicting mode flags
    2. Resolve --agent-config to a YAML source (preset / file / inline / alias)
    3. Validate PTI-only flags against topology kind
    4. Resume + workspace allocation (R5b safety; R5.1 PTI native field)
    5. Initial-plan handling (R5.3 PTI native field)
    6. Build override map
    7. Load + post-process cfg (model walk, dual collapse, OmegaConf->dict)
    8. Instantiate + wire UI (graph_reporter, interactive)
    9. Run with cancellation propagation
    10. Return ToolExecutionResult (consumed by both slash + agent paths)
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_logger = logging.getLogger(__name__)

# configs/ dir lives alongside this file (formerly topologies/)
_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# Preset PTI is the canonical PTI YAML; PTI has no _target_ alias so bare-alias
# resolution always falls through to the preset path.
_PTI_PRESET_NAMES = {"pti", "pti-simple"}

# Friendly `--config` names that map to a differently-named preset file.
# Used because (a) a root-level `_import_:` alias YAML does NOT work — the
# executor sniffs `_target_` from the raw YAML without resolving imports, so an
# import-only alias would report an empty `_target_` and mis-drive PTI/plan
# detection — and (b) some canonical files are named for history, not for the
# `--config` value users type.
_CONFIG_ALIASES: dict[str, str] = {
    "full-plan": "breakdown-multiflow-plan",   # coverage + diversity (existing file)
    "pti": "default",                          # full PTI plan+implement
    "multiflow": "multiflow-plan",             # diversity-only (file formerly multiple.yaml)
    "conversation": "disabled",                # conversational router (Phase 2)
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _camel_to_kebab(s: str) -> str:
    """Acronym-aware camelCase -> kebab-case (R2.4 two-rule regex).

    MultiFlowDual -> multi-flow-dual
    BTADual       -> bta-dual
    ClaudeCodeCLI -> claude-code-cli
    """
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "-", s).lower()


def _resolve_agent_config(spec: str, configs_dir: Path = _CONFIGS_DIR) -> tuple[str, Any]:
    """Resolve --agent-config <spec> to ('file', Path) or ('inline', dict).

    Detection priority (R2):
        1. starts with '{' -> inline JSON/YAML
        2. file path (contains / \\ or .yaml/.yml suffix) -> file
        3. lower-cased input matches a configs/*.yaml preset -> file
        4. acronym-aware camelCase->kebab match -> file
        5. otherwise ValueError listing presets + close matches
    """
    spec = (spec or "default").strip()
    if not spec:
        spec = "default"

    # Rule 1: inline JSON/YAML
    if spec.startswith("{"):
        parsed = yaml.safe_load(spec)
        if not isinstance(parsed, dict):
            raise ValueError(f"--agent-config inline value must parse to a dict, got: {type(parsed).__name__}")
        return ("inline", parsed)

    # Rule 2: file path
    looks_like_path = ("/" in spec or "\\" in spec or spec.endswith((".yaml", ".yml")))
    if looks_like_path:
        path = Path(spec)
        if not path.is_file():
            raise ValueError(f"--agent-config file not found: {spec}")
        return ("file", path)

    # Rule 2.5: friendly named aliases (e.g. full-plan, pti, multiflow, conversation).
    # Only resolves when the target preset file actually exists, so a Phase-2
    # alias (conversation -> disabled) cleanly falls through until disabled.yaml lands.
    aliased = _CONFIG_ALIASES.get(spec.lower())
    if aliased:
        alias_path = configs_dir / f"{aliased}.yaml"
        if alias_path.is_file():
            return ("file", alias_path)

    # Rule 3: lower-cased preset filename match
    preset_path = configs_dir / f"{spec.lower()}.yaml"
    if preset_path.is_file():
        return ("file", preset_path)

    # Rule 4: acronym-aware camelCase->kebab normalization
    kebab = _camel_to_kebab(spec)
    if kebab != spec.lower():
        kebab_path = configs_dir / f"{kebab}.yaml"
        if kebab_path.is_file():
            return ("file", kebab_path)

    # Rule 5: error with helpful suggestions
    import difflib
    available = sorted(p.stem for p in configs_dir.glob("*.yaml"))
    available.extend(k for k in sorted(_CONFIG_ALIASES) if k not in available)
    close = difflib.get_close_matches(spec.lower(), available, n=3)
    suggest = f"Did you mean: {', '.join(close)}?" if close else f"Available presets: {', '.join(available)}"
    raise ValueError(f"--agent-config '{spec}' is not a known preset, file path, or registered alias. {suggest}")


def _topology_target_str(source: tuple[str, Any]) -> str:
    """Read root _target_ string of a resolved source — without full instantiation."""
    kind, payload = source
    if kind == "inline":
        return str(payload.get("_target_", "")) if isinstance(payload, dict) else ""
    text = Path(payload).read_text(encoding="utf-8")
    parsed = yaml.safe_load(text) or {}
    return str(parsed.get("_target_", "")) if isinstance(parsed, dict) else ""


def _topology_is_pti(source: tuple[str, Any]) -> bool:
    """Detect whether the resolved topology contains a PTI — either as root or nested.

    The default.yaml wraps PTI inside an outer Dual, so checking only the root
    _target_ misses it. Read the full config text instead.
    """
    kind, payload = source
    try:
        text = str(payload) if kind == "inline" else Path(payload).read_text(encoding="utf-8")
    except OSError:
        return False
    return "PlanThenImplementInferencer" in text or "_target_: PTI" in text


_CONVERSATIONAL_TARGETS = {
    "Conversational", "ConversationalInferencer",
    "agent_foundation.common.inferencers.agentic_inferencers.conversational"
    ".conversational_inferencer.ConversationalInferencer",
}

# Hard cap on nested task/router recursion (router -> task -> router -> ...).
_MAX_TASK_DEPTH = 2


def _topology_is_conversational(source: tuple[str, Any]) -> bool:
    """True if the resolved topology root is a ConversationalInferencer (the
    `disabled`/`conversation` router), which the executor hosts via a dedicated
    agentic-loop path instead of the normal instantiate+ainfer path."""
    return _topology_target_str(source) in _CONVERSATIONAL_TARGETS


def _config_supports_implementation(source: tuple[str, Any]) -> bool:
    """True if the config contains a PTI / implementation phase that ``--plan`` can
    toggle or swap (i.e. the full plan+implement ``default.yaml``).

    Plan-only presets (``breakdown``, ``multiflow-plan``, ``full-plan`` / the standalone
    planner) have NO implementation phase, so ``--plan`` must be a clean no-op for
    them — never the ``enable_implementation=False`` fallback (which would inject a
    constructor kwarg the plan-only root does not accept).

    Detection is text-based on the raw source (cheap; no instantiation), looking for
    PTI markers or the implementation toggles a PTI YAML exposes.
    """
    kind, payload = source
    try:
        text = str(payload) if kind == "inline" else Path(payload).read_text(encoding="utf-8")
    except OSError:
        return True  # conservative: preserve legacy behavior if the file can't be read
    return (
        "PlanThenImplementInferencer" in text
        or "_target_: PTI" in text
        or "enable_implementation" in text
    )


def _parse_yaml_scalar(s: str) -> Any:
    """Parse a string as a YAML scalar (preserves int/float/bool/string semantics)."""
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def _parse_overrides(items) -> dict:
    """Normalize --override list to dict with parsed scalar values."""
    if items is None:
        return {}
    if isinstance(items, str):
        items = [items]
    out: dict = {}
    for item in items:
        if "=" not in item:
            _logger.warning("--override missing '=': %s (ignored)", item)
            continue
        key, _, val = item.partition("=")
        out[key.strip()] = _parse_yaml_scalar(val.strip())
    return out


def _resolve_proposal_plan(
    proposal_path: str, proposal_ids_str: Optional[str],
    top_k: Optional[int] = None,
) -> Optional[str]:
    """Load proposals, filter by IDs, format as plan file, return temp file path.

    Selection precedence: explicit ``proposal_ids_str`` wins; otherwise, if
    ``top_k`` is a positive int, take the top-K by rank (``all_proposals()`` is
    sorted rank-ascending); otherwise take all proposals.
    """
    from agent_foundation.common.data_models.proposal.parser import (
        parse_proposal_file,
    )

    proposal_abs = Path(proposal_path).resolve()
    idx = parse_proposal_file(proposal_abs)
    if idx is None:
        _logger.error("Cannot parse proposals from: %s", proposal_abs)
        return None

    if proposal_ids_str:
        ids = [s.strip() for s in proposal_ids_str.split(",") if s.strip()]
        try:
            selected = idx.get_proposals_by_ids(ids)
        except KeyError as exc:
            _logger.error("%s", exc)
            return None
    elif top_k is not None and top_k > 0:
        selected = idx.all_proposals()[:top_k]
    else:
        selected = idx.all_proposals()

    if not selected:
        _logger.error("No proposals selected from: %s", proposal_abs)
        return None

    lines = [
        "# Initial Plan — Selected from research-propose proposals\n",
        f"_Source: {proposal_abs}_\n",
        f"_Selected: {', '.join(p.id for p in selected)}_\n",
        "",
    ]
    index_dir = proposal_abs.parent
    for p in selected:
        lines.append(f"## {p.id} — {p.title}")
        lines.append(f"**Rank:** {p.rank} | **Impact:** {p.impact or 'n/a'} | **Complexity:** {p.complexity or 'n/a'}\n")
        if p.problem:
            lines.append(f"### Problem\n{p.problem}\n")
        if p.approach:
            lines.append(f"### Approach\n{p.approach}\n")
        if p.dependencies:
            lines.append(f"**Dependencies:** {', '.join(p.dependencies)}\n")
        if p.cross_refs:
            lines.append(f"**Cross-refs:** {p.cross_refs}\n")
        if p.proposal_file:
            detail_path = (index_dir / p.proposal_file).resolve()
            if detail_path.is_file():
                detail = detail_path.read_text(encoding="utf-8", errors="replace")
                lines.append(f"### Full Proposal Detail\n_(from {p.proposal_file})_\n")
                lines.append(detail)
            else:
                lines.append(f"_Full proposal detail at: {p.proposal_file}_\n")
        lines.append("")

    import tempfile
    plan_file = tempfile.NamedTemporaryFile(
        mode="w", suffix="_proposal_plan.md", delete=False, encoding="utf-8",
    )
    plan_text = "\n".join(lines)
    plan_file.write(plan_text)
    plan_file.close()

    import json as _json
    from datetime import datetime, timezone
    audit = {
        "index_path": str(proposal_abs),
        "selected_ids": [p.id for p in selected],
        "picked_at": datetime.now(timezone.utc).isoformat(),
    }
    audit_path = Path(plan_file.name).parent / "_picked_proposals.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        _json.dump(audit, f, indent=2)

    _logger.info("Resolved %d proposals from %s → %s",
                 len(selected), proposal_abs, plan_file.name)
    return plan_file.name


def _derive_mode_from_flags(arguments: dict) -> Optional[str]:
    """Map mutually-exclusive --plan/--execute/--full/--confirm flags to a mode string."""
    for f, m in (("plan", "plan"), ("execute", "execute"), ("full", "full"), ("confirm", "confirm")):
        if arguments.get(f):
            return m
    return None


def _allocate_workspace(
    task_id: str, session_context: Optional[dict] = None
) -> Path:
    """Allocate workspace via the shared helper.

    Path B (server-affiliated): session_context["session_root"] set
        → <session_root>/tasks/task_<TS>_<uuid8>/
    Path A (standalone): no session_root
        → <repo>/_runtime/tasks/task/task_<TS>_<uuid8>/
    """
    from agent_foundation.common.workspace.allocator import (
        allocate_tool_workspace,
    )
    sc = session_context or {}
    tool_name = sc.get("tool_name", "task")
    session_root_str = sc.get("session_root", "")
    if session_root_str:
        base = Path(session_root_str) / "tasks"
        base.mkdir(parents=True, exist_ok=True)
        return allocate_tool_workspace(tool_name, base_dir=base)
    return allocate_tool_workspace(tool_name, base_dir=None)


def _resolve_workspace(session_context: Optional[dict], task_id: str) -> Path:
    """Resolve workspace: accept dispatcher pre-allocated path or allocate fresh.

    Branch 1 (backward-compat): if working_dir is set and looks like a
        per-task workspace (contains /tasks/ or /_runtime/), use it as-is.
        This handles the dispatcher's pre-allocated paths and resume scenarios.
    Branch 2 (default): allocate via shared helper, routing by session_root.
    """
    sc = session_context or {}
    candidate = sc.get("working_dir", "")
    if candidate:
        try:
            posix = Path(candidate).as_posix()
        except Exception:
            posix = ""
        if "/tasks/" in posix or "/_runtime/" in posix:
            ws = Path(candidate)
            ws.mkdir(parents=True, exist_ok=True)
            return ws
    return _allocate_workspace(task_id, session_context)


def _apply_resume(path_str: str, *, copy_workspace: bool, in_place: bool) -> Path:
    """R5.1 — validate + (optionally) copy the resume workspace; return effective path."""
    src = Path(path_str)
    if not src.is_dir():
        raise FileNotFoundError(f"--resume workspace does not exist: {path_str}")
    if copy_workspace:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = src.parent / f"{src.name}_resume_{ts}"
        shutil.copytree(src, dst)
        _logger.info("--copy-workspace: copied %s -> %s", src, dst)
        return dst
    return src


def _walk_replace_model(cfg: Any, new_value: str) -> int:
    """Recursively walk plain dict/list cfg; replace every `model_name` leaf. Returns count."""
    count = 0
    if isinstance(cfg, dict):
        for k, v in list(cfg.items()):
            if k == "model_name" and not isinstance(v, (dict, list)):
                cfg[k] = new_value
                count += 1
            else:
                count += _walk_replace_model(v, new_value)
    elif isinstance(cfg, list):
        for v in cfg:
            count += _walk_replace_model(v, new_value)
    return count


_DUAL_TARGETS = {"Dual", "DualInferencer",
                 "agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer.DualInferencer"}


def _collapse_dual(cfg: Any) -> int:
    """R3.3 — collapse Dual nodes to base_inferencer subtree. Replace at PARENT slot.
    Handles nested Duals by looping while the parent slot keeps resolving to a Dual."""
    count = 0
    if isinstance(cfg, dict):
        for k, v in list(cfg.items()):
            if isinstance(v, dict) and v.get("_target_") in _DUAL_TARGETS:
                # Iteratively collapse nested Duals at the same slot
                while isinstance(cfg[k], dict) and cfg[k].get("_target_") in _DUAL_TARGETS:
                    base = cfg[k].get("base_inferencer")
                    if base is None:
                        break
                    cfg[k] = base
                    count += 1
                count += _collapse_dual(cfg[k])
            else:
                count += _collapse_dual(v)
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            if isinstance(v, dict) and v.get("_target_") in _DUAL_TARGETS:
                while isinstance(cfg[i], dict) and cfg[i].get("_target_") in _DUAL_TARGETS:
                    base = cfg[i].get("base_inferencer")
                    if base is None:
                        break
                    cfg[i] = base
                    count += 1
                count += _collapse_dual(cfg[i])
            else:
                count += _collapse_dual(v)
    return count


_BTA_TARGETS = {
    "BTA", "BreakdownThenAggregateInferencer",
    "agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers"
    ".breakdown_then_aggregate_inferencer.BreakdownThenAggregateInferencer",
}
_MFDUAL_TARGETS = {
    "MultiFlowDual", "MultiFlowDualInferencer",
    "agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers"
    ".multi_flow_dual_inferencer.MultiFlowDualInferencer",
}


def _disable_aggregation(cfg: Any) -> int:
    """--no-aggregate: set ``disable_aggregator=True`` on every BTA node and
    ``multi_flow_disable_aggregator=True`` on every MFDual node, so workers run
    but their outputs are returned as a list (no synthesis). Operates on the
    plain dict cfg (pre-instantiate), same as ``_collapse_dual``. Returns count.

    Note: ``_factory_:`` markers are rewritten to ``_target_:`` by ``load_config``
    before this walk runs, so ``_factory_: MultiFlowDual`` worker entries are
    matched here too.
    """
    count = 0
    if isinstance(cfg, dict):
        tgt = cfg.get("_target_")
        if tgt in _BTA_TARGETS:
            cfg["disable_aggregator"] = True
            count += 1
        elif tgt in _MFDUAL_TARGETS:
            cfg["multi_flow_disable_aggregator"] = True
            count += 1
        for v in cfg.values():
            count += _disable_aggregation(v)
    elif isinstance(cfg, list):
        for v in cfg:
            count += _disable_aggregation(v)
    return count


def _serialize_multi_output(parts: list) -> str:
    """Serialize multiple worker outputs (no-aggregate mode) into one markdown
    document so the FULL list survives into the calling conversation.

    Without this, ``_extract_result_text`` collapsed a multi-worker tuple to
    ``result[0]`` and silently dropped the rest — which made the no-aggregate /
    list-of-outputs-to-conversation pattern impossible. Each part renders under a
    ``### Worker N`` header.
    """
    blocks = []
    for i, part in enumerate(parts, start=1):
        text = _extract_result_text(part)
        blocks.append(f"### Worker {i}\n\n{text}".rstrip())
    return "\n\n".join(blocks)


def _extract_result_text(result: Any) -> str:
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
            return _extract_result_text(non_none[0])
        return _serialize_multi_output(non_none)
    return str(result)


def _discover_artifacts(workspace: Optional[Path]) -> dict:
    """Best-effort discovery of standard output artifacts under the workspace."""
    if workspace is None or not Path(workspace).is_dir():
        return {}
    ws = Path(workspace)
    out = {}
    for relpath, key in (("outputs/plan.md", "plan_path"),
                         ("outputs/implementation.md", "impl_path"),
                         ("outputs/role_document.md", "doc_path"),
                         ("outputs/role_setup_report.md", "report_path")):
        p = ws / relpath
        if p.is_file():
            out[key] = str(p)
    return out


def _error(msg: str):
    """R6.4 error return shape — both invocation paths handle .result + .context_updates."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
        ToolExecutionResult,
    )
    _logger.error("[task] %s", msg)
    return ToolExecutionResult(result=msg, context_updates={"success": False})


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

_TASK_TOOL_NAMES = {"task", "task-plan", "task-execute", "task-full", "task-confirm"}


async def _run_conversational_router(
    *,
    config_path: Any,                  # path to disabled.yaml (the Conversational config)
    request: str,
    model: Optional[str],
    working_dir: Path,
    session_context: dict,
):
    """Host the `--config disabled`/`conversation` router.

    Builds a ConversationalInferencer (shared `_ci_host` scaffolding), gives it the
    `task` tool (forced synchronous so nested dispatches complete inline and the
    router can read their results), runs the agentic loop, and returns the router's
    final message.

    Interactive-hang safety: in the main async chat, ``session_context["interactive"]``
    is present but its receive queue is torn down after the dispatching turn — so a
    clarifying-question round-trip would block forever. We therefore only enable
    interactive when the caller explicitly guarantees a registered receive queue
    via ``session_context["router_interactive_safe"]`` (e.g. the dev-slash `/task`
    path or a CLI terminal). Otherwise the router runs autonomously (yolo).
    """
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
        ToolExecutionResult,
    )
    from agent_foundation.resources.tools import _ci_host
    from agent_foundation.resources.tools.registry import load_all_tools

    sc = session_context or {}
    tool_dirs = sc.get("extra_tool_dirs")
    depth = int(sc.get("task_depth", 0))

    registry = load_all_tools(extra_dirs=tool_dirs)
    # Nested tool calls must complete inline (no fire-and-forget) so the router
    # can read results and synthesize. Mirrors sop/cli.py.
    _ci_host.force_tools_synchronous(registry)

    interactive = sc.get("interactive") if sc.get("router_interactive_safe") else None

    # Depth-guarded dispatcher: increment task_depth on nested `task` calls and,
    # at the cap, coerce a nested router (`disabled`/`conversation`/unset) to
    # `full-plan` so we can never recurse into the router indefinitely.
    nested_sc = {**sc, "task_depth": depth + 1}
    nested_exec = _ci_host.make_tool_executor(nested_sc, tool_dirs=tool_dirs)
    base_exec = _ci_host.make_tool_executor(sc, tool_dirs=tool_dirs)

    async def _router_tool_executor(tool_name: str, arguments: dict) -> Any:
        if tool_name in _TASK_TOOL_NAMES:
            arguments = dict(arguments)
            cfg = str(arguments.get("config") or arguments.get("agent_config") or "").lower()
            if depth + 1 >= _MAX_TASK_DEPTH and cfg in ("", "disabled", "conversation"):
                _logger.info(
                    "[task] router depth cap (%d) reached: coercing nested "
                    "--config %r -> full-plan", _MAX_TASK_DEPTH, cfg or "(unset)",
                )
                arguments["config"] = "full-plan"
            return await nested_exec(tool_name, arguments)
        return await base_exec(tool_name, arguments)

    try:
        ci = _ci_host.build_ci_from_config(
            config_path,
            model=model or "",
            tool_registry=registry,
            tool_executor=_router_tool_executor,
            interactive=interactive,
            target_path=working_dir,
        )
    except Exception as exc:
        return _error(f"Conversational router build failed: {exc}")

    if interactive is None and hasattr(ci, "yolo_mode"):
        ci.yolo_mode = True

    # §9.4 host: mint the root RunContext for the conversational router turn
    # (workspace-rooted) and persist the store for resume (M9). Best-effort; any
    # failure falls back to the legacy call (run_context=None -> byte-identical).
    _conv_root = None
    try:
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )
        from agent_foundation.common.inferencers.run_context import (
            RunContext,
            RunStateStore,
        )

        # M9 resume: rehydrate from a prior snapshot when present.
        _cstore_path = Path(working_dir) / "run_state" / "store.json"
        _cstore = None
        if _cstore_path.exists():
            try:
                _cstore = RunStateStore.load(str(_cstore_path))
            except Exception:  # pragma: no cover
                _cstore = None
        _conv_root = RunContext.root(
            workspace=InferencerWorkspace(root=str(working_dir)),
            store=_cstore,
        )
    except Exception:  # pragma: no cover
        _conv_root = None

    try:
        if _conv_root is not None:
            result = await ci.run_agentic_loop(
                request, interactive=interactive, run_context=_conv_root
            )
        else:
            result = await ci.run_agentic_loop(request, interactive=interactive)
    except Exception as exc:
        _logger.exception("[task] conversational router failed")
        return _error(f"Conversational router failed: {exc}")
    finally:
        # M9: persist on EVERY exit (success/error/cancel) so an interrupted or
        # HITL-paused conversational run can resume. Best-effort; never fatal.
        if _conv_root is not None:
            try:
                _conv_root._store.save(
                    str(Path(working_dir) / "run_state" / "store.json")
                )
            except Exception:  # pragma: no cover - persistence is best-effort
                _logger.debug(
                    "[task] router RunStateStore persist skipped", exc_info=True
                )

    text = getattr(result, "text", None) or _extract_result_text(result)
    artifacts = _discover_artifacts(working_dir)
    context_updates = {"workspace_path": str(working_dir), "router": True, "success": True}
    context_updates.update(artifacts)
    return ToolExecutionResult(result=text, context_updates=context_updates)


async def _run_topology(
    *,
    source: tuple,                              # ("file", Path) | ("inline", dict)
    request: str,
    overrides: Optional[dict] = None,           # dotted-key → already-typed value (NO string parsing)
    model: Optional[str] = None,
    no_dual: bool = False,
    aggregate: bool = True,
    mode: str = "full",
    analysis: bool = False,
    multi_iter: bool = False,
    max_iter: int = 3,
    init_plan_path: Optional[str] = None,       # absolute path; --use-plan feeds PTI's initial_plan_file
    resume_workspace: Optional[str] = None,     # absolute path; takes precedence over auto-allocation
    session_context: Optional[dict] = None,
    env_prefix: Optional[str] = None,           # highest-priority env namespace for _params (e.g. a derived tool's)
    config_defaults: Optional[dict] = None,     # tool defaults applied below env (overridable by env/CLI)
):
    """Programmatic core — Stages 3-10 of the slash pipeline.

    Both `execute()` (slash entry) and tool shims (e.g. /create_role, /role_setup)
    call this. Accepts `overrides` as a Python `dict[str, Any]` (already-typed) —
    no string→YAML round-trip required for programmatic callers.

    Workspace decision:
      - `resume_workspace` (if set) is used as-is (and surfaced as `resume_workspace`
        override to PTI for native resume detection).
      - Else: `_resolve_workspace(session_context, task_id)` — respects safe
        dispatcher-provided `working_dir` hint, falls through to `_allocate_workspace`.
    """
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
        ToolExecutionResult,
    )

    sc = session_context or {}
    overrides = dict(overrides) if overrides else {}
    task_id = sc.get("task_id") or f"task-{uuid.uuid4().hex[:8]}"

    # Stage 3 — PTI-only flag validation
    is_pti = _topology_is_pti(source)
    if (analysis or multi_iter or mode == "confirm") and not is_pti:
        return _error(
            f"--analysis / --multi-iter / --confirm require a PTI topology. "
            f"Got root _target_: '{_topology_target_str(source) or '(none)'}'. "
            f"Use --agent-config pti or --agent-config pti-simple."
        )

    # Stage 4 — Workspace decision
    if resume_workspace:
        working_dir = Path(resume_workspace)
        working_dir.mkdir(parents=True, exist_ok=True)
    else:
        working_dir = _resolve_workspace(sc, task_id)

    # Stage 4b — Conversational router (--config disabled / conversation).
    # Hosted via a dedicated agentic-loop path (NOT instantiate+ainfer), because
    # ConversationalInferencer's loop lives in run_agentic_loop and needs a wired
    # tool_registry + tool_executor.
    if _topology_is_conversational(source):
        if source[0] != "file":
            return _error("Conversational router requires a file config (disabled.yaml).")
        return await _run_conversational_router(
            config_path=source[1],
            request=request,
            model=model,
            working_dir=working_dir,
            session_context=sc,
        )

    # Stage 5 — --use-plan validation (file-IO done by caller; we just consume the path)
    if init_plan_path:
        plan_abs = Path(init_plan_path).resolve()
        if not plan_abs.is_file():
            return _error(f"--use-plan file not found: {init_plan_path}")
        init_plan_path = str(plan_abs)

    # Stage 6 — Build override map.
    #
    # Workspace routing (updated 2026-05-07):
    #   - Topology YAMLs declare `workspace.root: ${_params.workspace_root}`.
    #     The allocator injects the resolved path via the override below.
    #     The YAML's `_params.workspace_root: ???` (MISSING sentinel) fails
    #     loud at load time if the override is missing.
    #   - ClaudeCodeCli / KiroCli (single-leaf YAMLs) → use `target_path`.
    #   - `_target_path` (underscore prefix) cascades to ALL descendant
    #     leaves via _instantiate.py's auto-injection mechanism. Without
    #     the prefix, only the root node gets it — children fall through
    #     to os.getcwd() (a narrow per-task subdir).
    overrides.setdefault("_target_path", str(working_dir))
    overrides["_params.workspace_root"] = str(working_dir)

    # Template roots: AF (framework) is always included. Application-level
    # roots (e.g., OpenStartup's prompt_templates/) are injected via
    # session_context["extra_template_dirs"] — extra roots take precedence
    # over AF defaults (listed first = higher priority in TemplateManager).
    if "_template_manager.templates" not in overrides:
        import agent_foundation.resources as _af_res
        _af_templates = Path(_af_res.__file__).parent / "prompt_templates"
        extra_roots = [str(p) for p in (sc or {}).get("extra_template_dirs", [])]
        overrides["_template_manager.templates"] = extra_roots + [str(_af_templates)]
    if resume_workspace:
        overrides["resume_workspace"] = str(working_dir)
    if init_plan_path and is_pti:
        # PTI is the root _target_ in default.yaml. The base_inferencer
        # prefix is kept for backward compat with any custom config that
        # wraps PTI inside an outer Dual — harmlessly stripped if unused.
        overrides["initial_plan_file"] = init_plan_path
        overrides["base_inferencer.initial_plan_file"] = init_plan_path

    # ----- Mode handling --------------------------------------------------
    # `--plan` mode: swap to the standalone planner topology.
    #
    # Why not just set `enable_implementation=False` on the full PTI YAML?
    # Because the full topology wraps PTI in an OUTER Dual that reviews the
    # final implementation deliverable. With implementation disabled, PTI
    # returns "" (empty string) and the outer Dual ends up reviewing an
    # empty deliverable using `template_root_space=implementation` review
    # criteria — wasted iterations + wrong evaluation criteria.
    #
    # The standalone `breakdown-multiflow-plan.yaml` is purpose-built for
    # plan-only runs: it has its OWN outer Dual that reviews the PLAN
    # itself with `_template_root_space=plan` criteria. The full PTI
    # topology imports this same file via `_import_:`, so there's no
    # drift risk between the two paths.
    if mode == "plan" and not _config_supports_implementation(source):
        # Plan-only presets (breakdown, multiple, full-plan / standalone planner)
        # have no implementation phase. --plan is a clean no-op: do NOT swap and do
        # NOT inject enable_implementation=False (the plan-only root would reject it).
        _logger.info(
            "[task] --plan with plan-only topology (%s): no swap/toggle needed.",
            Path(source[1]).name if source[0] == "file" else "inline",
        )
    elif mode == "plan":
        if source[0] == "file":
            full_yaml_path = Path(source[1])
            standalone_path = full_yaml_path.parent / "breakdown-multiflow-plan.yaml"
            if (
                full_yaml_path.name == "default.yaml"
                and standalone_path.is_file()
            ):
                _logger.info(
                    "[task] --plan: swapping topology %s → %s "
                    "(standalone planner has its own outer Dual reviewing "
                    "the plan; avoids empty-deliverable review).",
                    full_yaml_path.name, standalone_path.name,
                )
                source = ("file", standalone_path)
                # Recompute is_pti: standalone has no PTI wrapper.
                is_pti = _topology_is_pti(source)
                # Don't pass enable_implementation override — standalone
                # YAML has no PTI to receive it (would be a no-op key,
                # but skipping is cleaner).
            else:
                # Custom YAML or standalone path lookup failed — fall back
                # to the legacy flag toggle so behavior is at least
                # consistent (and observable) for unusual configs.
                _logger.warning(
                    "[task] --plan: cannot swap to standalone planner "
                    "(source=%s, standalone exists=%s). Falling back to "
                    "enable_implementation=False on PTI; outer Dual may "
                    "review an empty deliverable.",
                    full_yaml_path, standalone_path.is_file(),
                )
                overrides["enable_implementation"] = False
        else:
            # Inline (dict) source — no file to swap. Same legacy fallback.
            _logger.warning(
                "[task] --plan: inline agent-config can't be swapped to "
                "standalone planner. Using enable_implementation=False; "
                "outer Dual may review an empty deliverable.",
            )
            overrides["enable_implementation"] = False
    if mode == "execute":
        # Execute mode legitimately needs the full PTI YAML — it skips
        # planning and runs implementation. Asymmetric with --plan by
        # design.
        overrides["enable_planning"] = False
    if mode == "confirm":
        overrides["enable_checkpoint_plan_review"] = True
    if analysis:
        overrides["enable_analysis"] = True
    if multi_iter:
        overrides["enable_multiple_iterations"] = True
        overrides["max_meta_iterations"] = max_iter

    # Stage 7 — Load + post-process cfg
    import agent_foundation.common.configs.registered_targets  # noqa: F401 — register aliases
    from rich_python_utils.config_utils import load_config, instantiate
    from omegaconf import OmegaConf, DictConfig

    try:
        if source[0] == "file":
            cfg = load_config(str(source[1]), overrides=overrides,
                              env_prefix=env_prefix, config_defaults=config_defaults)
        else:
            cfg = OmegaConf.merge(OmegaConf.create(source[1]), OmegaConf.create(overrides))
    except Exception as exc:
        return _error(f"load_config failed for source {source}: {exc}")

    # OmegaConf -> plain dict for safe walk-and-mutate, then back to DictConfig
    # for instantiate() (which requires an OmegaConf config object).
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    if model:
        n = _walk_replace_model(cfg, model)
        _logger.info("[task] --model %s replaced %d model_name leaves", model, n)
    if no_dual:
        n = _collapse_dual(cfg)
        _logger.info("[task] --no-dual collapsed %d Dual nodes", n)
    if not aggregate:
        n = _disable_aggregation(cfg)
        _logger.info(
            "[task] --no-aggregate disabled aggregation on %d node(s) "
            "(workers return a list; the caller/conversation aggregates)", n,
        )

    # Re-wrap as DictConfig for instantiate()
    cfg = OmegaConf.create(cfg)

    # For non-PTI topologies, prepend the plan to the request as a fallback
    if init_plan_path and not is_pti:
        try:
            plan_text = Path(init_plan_path).read_text(encoding="utf-8")
            request = f"Plan (preloaded):\n{plan_text}\n\nRequest: {request}"
            _logger.warning("--use-plan with non-PTI topology: prepending plan to request")
        except OSError:
            pass

    # Stage 8 — Instantiate + wire UI
    try:
        inferencer = instantiate(cfg)
    except Exception as exc:
        keys = list(cfg.keys()) if isinstance(cfg, dict) else "(non-dict cfg)"
        return _error(f"Instantiation failed: {exc}\nTopology root keys: {keys}")

    try:
        from agent_foundation.ui.graph_reporter_factory import make_graph_reporter
        inferencer.graph_reporter = make_graph_reporter(sc, task_id)
        if inferencer.graph_reporter is not None:
            _logger.info("[task] graph_reporter attached: %s",
                         type(inferencer.graph_reporter).__name__)
    except Exception as exc:
        _logger.warning("[task] graph_reporter attach failed: %s", exc)

    # /task-confirm: PTI's enable_checkpoint_plan_review (set above) routes to
    # async-native checkpoint_plan_review which uses asend_response/aget_input —
    # natively compatible with WebSocketInteractive. The single_choice
    # (Approve/Modify/Reject) mode renders via the existing SingleChoiceWidget —
    # no custom widget tagging needed.
    if mode == "confirm" and hasattr(inferencer, "interactive") and interactive is not None:
        inferencer.interactive = interactive

    # Stage 9 — Run with cancellation propagation.
    # §9.4 host: mint the root RunContext (workspace-rooted) and thread it so the
    # explicit run-state separation is active for this topology; persist the
    # Tier-1 RunStateStore for resume (M9). Resilient: any failure to build the
    # context falls back to the legacy call (run_context=None -> byte-identical).
    _root_ctx = None
    try:
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )
        from agent_foundation.common.inferencers.run_context import (
            RunContext,
            RunStateStore,
        )

        # M9 resume: rehydrate the Tier-1 store from a prior run's snapshot when
        # present, so dispatch/conversation state is restored; else a fresh store.
        _store_path = Path(working_dir) / "run_state" / "store.json"
        _store = None
        if _store_path.exists():
            try:
                _store = RunStateStore.load(str(_store_path))
            except Exception:  # pragma: no cover - corrupt snapshot -> fresh
                _store = None
        _root_ctx = RunContext.root(
            workspace=InferencerWorkspace(root=str(working_dir)),
            store=_store,
        )
    except Exception:  # pragma: no cover - never block execution on context setup
        _root_ctx = None

    try:
        if _root_ctx is not None:
            result = await inferencer.ainfer(request, run_context=_root_ctx)
        else:
            result = await inferencer.ainfer(request)
    except asyncio.CancelledError:
        if hasattr(inferencer, "cancel"):
            try:
                await inferencer.cancel()
            except Exception:
                pass
        raise
    except Exception as exc:
        _logger.exception("[task] inferencer.ainfer failed")
        return _error(f"Execution failed: {exc}")
    finally:
        # M9: persist the Tier-1 run-state store on EVERY exit — success, error, OR
        # cancel — so an interrupted/HITL-paused run can resume (the partial dispatch
        # + conversation state lives in the store). Best-effort; never fatal.
        if _root_ctx is not None:
            try:
                _root_ctx._store.save(
                    str(Path(working_dir) / "run_state" / "store.json")
                )
            except Exception:  # pragma: no cover - persistence is best-effort
                _logger.debug("[task] RunStateStore persist skipped", exc_info=True)

    # Stage 10 — Return ToolExecutionResult
    artifacts = _discover_artifacts(working_dir)
    context_updates = {"workspace_path": str(working_dir), "success": True}
    context_updates.update(artifacts)
    return ToolExecutionResult(
        result=_extract_result_text(result),
        context_updates=context_updates,
    )


async def execute(arguments: dict, session_context: dict):
    """Slash + agent entry point — parses slash-command arguments then delegates to
    `_run_topology()`. Programmatic callers (tool shims) should call `_run_topology`
    directly with a Python dict to avoid the string→YAML round-trip.
    """
    # Stage 1 — Parse arguments
    request = (arguments.get("request") or "").strip()
    mode = arguments.get("mode") or _derive_mode_from_flags(arguments) or "full"
    spec = arguments.get("agent_config") or arguments.get("config") or "default"
    overrides = _parse_overrides(arguments.get("override", []))
    # Tool config_overrides (from derived_from.defaults) are TOOL DEFAULTS, not
    # forced overrides: thread them as config_defaults so they apply BELOW env.
    # Precedence: CLI --override (overrides) > env (<PREFIX>__<KEY>) > config_defaults > YAML.
    config_defaults = dict(arguments.get("config_overrides") or {}) or None
    model = arguments.get("model")
    no_dual = bool(arguments.get("no_dual"))
    no_aggregate = bool(arguments.get("no_aggregate"))
    analysis = bool(arguments.get("analysis"))
    multi_iter = bool(arguments.get("multi_iter"))
    max_iter = int(arguments.get("max_iterations", 3))
    resume = arguments.get("resume")
    copy_ws = bool(arguments.get("copy_workspace"))
    in_place = bool(arguments.get("in_place", True))
    use_plan = arguments.get("use_plan")
    template_version = arguments.get("template_version")
    template_master_version = arguments.get("template_master_version")

    if sum(bool(arguments.get(f)) for f in ("plan", "execute", "full", "confirm")) > 1:
        return _error("Multiple mode flags provided; use only one of --plan/--execute/--full/--confirm.")

    # Stage 2 — Resolve --agent-config
    try:
        source = _resolve_agent_config(spec, _CONFIGS_DIR)
    except ValueError as e:
        return _error(str(e))

    # Stage 4a — Slash-only file IO: --resume copy/in-place. Resolves to an absolute
    # workspace path that _run_topology will use directly (overrides workspace decision).
    resume_workspace_str = None
    if resume:
        try:
            working_dir = _apply_resume(resume, copy_workspace=copy_ws, in_place=in_place)
            resume_workspace_str = str(working_dir)
        except FileNotFoundError as e:
            return _error(str(e))

    # Stage 5a — --use-plan path validation (file-IO inside
    # _run_topology handles the read for non-PTI fallback).
    init_plan_path = None
    if use_plan:
        plan_abs = Path(use_plan).resolve()
        if not plan_abs.is_file():
            return _error(f"--use-plan file not found: {use_plan}")
        init_plan_path = str(plan_abs)

    use_proposal = arguments.get("use_proposal")
    proposal_ids_str = arguments.get("proposal_ids")
    top_k_raw = arguments.get("top_k")
    top_k_val: Optional[int] = None
    if top_k_raw is not None and str(top_k_raw).strip():
        try:
            top_k_val = int(str(top_k_raw).strip())
        except ValueError:
            return _error(f"--top-k must be an integer, got: {top_k_raw!r}")
    if use_proposal:
        if init_plan_path:
            return _error("--use-proposal and --use-plan are mutually exclusive.")
        init_plan_path = _resolve_proposal_plan(
            use_proposal, proposal_ids_str, top_k=top_k_val
        )
        if init_plan_path is None:
            return _error(f"Failed to resolve proposals from: {use_proposal}")

    if template_version:
        overrides["_template_manager.template_version"] = template_version
    if template_master_version:
        overrides["_template_master_version"] = template_master_version

    tool_name = arguments.get("tool_name")
    if tool_name:
        session_context = {**session_context, "tool_name": tool_name}

    return await _run_topology(
        source=source,
        request=request,
        overrides=overrides,
        model=model,
        no_dual=no_dual,
        aggregate=not no_aggregate,
        mode=mode,
        analysis=analysis,
        multi_iter=multi_iter,
        max_iter=max_iter,
        init_plan_path=init_plan_path,
        resume_workspace=resume_workspace_str,
        session_context=session_context,
        # Per-tool env namespace (e.g. derived tools set derived_from.defaults.env_prefix);
        # highest-priority prefix for _params env overrides on a shared config.
        env_prefix=arguments.get("env_prefix"),
        # Tool config_overrides applied as a defaults layer BELOW env (overridable).
        config_defaults=config_defaults,
    )
