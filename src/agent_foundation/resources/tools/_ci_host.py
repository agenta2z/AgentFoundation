"""Shared scaffolding for tools that host a nested ConversationalInferencer.

Extracted from ``sop/cli.py`` so that both ``sop`` and the ``task`` conversational
router (``--config disabled``) build and wire a CI the same way, without
duplicating the construction / tool-dispatch / force-synchronous logic.

Public surface:
  - ``build_ci_from_config(...)``  — instantiate a Conversational CI from a YAML
    config and wire its tool_registry / tool_executor / interactive.
  - ``force_tools_synchronous(registry)`` — make every tool run inline (no
    fire-and-forget), required when nesting tool calls inside a host CI.
  - ``make_tool_executor(session_context, ...)`` — an async dispatch closure that
    resolves each tool's executor from its ``tool.json`` (``executor: module:func``)
    or ``derived_from``.

None of these import server/transport code; callers inject ``interactive``.
"""

from __future__ import annotations

import importlib
import json as _json
from pathlib import Path
from typing import Any, Optional

# resources/tools/ — the AF tools root (this file lives directly under it).
_AF_TOOLS_ROOT = Path(__file__).resolve().parent


def build_ci_from_config(
    config_path: str | Path,
    *,
    model: str = "",
    backend: Optional[str] = None,
    backend_dir: Optional[str | Path] = None,
    tool_registry: Optional[dict] = None,
    tool_executor: Any = None,
    interactive: Any = None,
    target_path: Optional[str | Path] = None,
    prompt_renderer: Any = None,
    base_inferencer: Any = None,
    extra_sop_dirs: Optional[list] = None,
    allowed_sops: Optional[list] = None,
    disallowed_sops: Optional[list] = None,
) -> Any:
    """Instantiate a ConversationalInferencer from ``config_path`` and wire it.

    Mirrors the proven construction in ``sop/cli.py`` — the only generalizations
    are the explicit ``config_path``/``backend_dir`` (instead of a hard-coded
    ``configs/default.yaml``) and ``target_path`` (instead of always ``cwd``).

    ``prompt_renderer`` (optional): a pre-built renderer to set on the CI,
    overriding any YAML-declared / auto-default renderer. Needed by callers
    (e.g. the OpenStartup factory) that must build the renderer BEFORE the CI
    to drive tool-whitelist filtering, so it cannot be declared in the YAML.

    ``base_inferencer`` (optional): a pre-built backend inferencer to inject.
    When given, the YAML's ``base_inferencer`` node is dropped and the CI is
    instantiated as a Hydra ``_partial_`` (so the required ``base_inferencer``
    arg is not validated against the now-absent YAML node), then called with the
    live ``base_inferencer`` (plus ``prompt_renderer`` / ``extra_sop_dirs``) as
    plain Python kwargs. This is the path for hosts (e.g. the OpenStartup
    factory) that must build the backend themselves with runtime-only params
    (``cache_folder``, ``target_path`` + mkdir, model selection) that the YAML
    leaf cannot express — while still letting the YAML own the CI-wrapper
    config (``max_iterations``, ``soft_max_iterations``, ``compression_threshold``,
    ``_debug_mode`` cascade). ``backend`` and ``base_inferencer`` are mutually
    exclusive; passing both raises ``ValueError``.

    ``extra_sop_dirs`` (optional): SOP discovery dirs for the CI (init kwarg
    ``extra_sop_dirs`` → attr ``_extra_sop_dirs``).

    Raises ``ValueError`` for an unknown ``backend`` so callers can decide how to
    surface it (the SOP CLI prints + exits; the task router logs + falls back).
    """
    import agent_foundation.common.configs.registered_targets  # noqa: F401
    from omegaconf import OmegaConf
    from rich_python_utils.config_utils import instantiate, load_config

    inject_base = base_inferencer is not None
    if inject_base and backend:
        raise ValueError(
            "build_ci_from_config: 'backend' and 'base_inferencer' are mutually "
            "exclusive — pass a backend name to build the base from YAML, OR a "
            "pre-built base_inferencer to inject, not both."
        )

    config_path = Path(config_path)
    cfg = load_config(str(config_path))
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if backend:
        bdir = Path(backend_dir) if backend_dir else config_path.parent / "base_inferencer"
        backend_path = bdir / f"{backend}.yaml"
        if not backend_path.exists():
            available = (
                ", ".join(p.stem for p in bdir.glob("*.yaml")) if bdir.is_dir() else "(none)"
            )
            raise ValueError(f"Unknown backend '{backend}'. Available: {available}")
        cfg["base_inferencer"] = OmegaConf.to_container(
            OmegaConf.load(str(backend_path)), resolve=True,
        )

    if inject_base:
        # Drop the YAML base node and build the CI wrapper as a partial, then
        # call it with the live base. Live attrs objects can't round-trip
        # through OmegaConf, so they MUST be passed as plain Python kwargs.
        cfg.pop("base_inferencer", None)
        cfg["_partial_"] = True
        ctor_kwargs: dict = {"base_inferencer": base_inferencer}
        if prompt_renderer is not None:
            ctor_kwargs["prompt_renderer"] = prompt_renderer
        if extra_sop_dirs is not None:
            ctor_kwargs["extra_sop_dirs"] = list(extra_sop_dirs)
        if allowed_sops is not None:
            ctor_kwargs["allowed_sops"] = list(allowed_sops)
        if disallowed_sops is not None:
            ctor_kwargs["disallowed_sops"] = list(disallowed_sops)
        ci_partial = instantiate(OmegaConf.create(cfg))
        ci = ci_partial(**ctor_kwargs)
    else:
        bi = cfg.get("base_inferencer")
        if model and isinstance(bi, dict):
            if "model_name" in bi:
                bi["model_name"] = model
            elif "model_id" in bi:
                bi["model_id"] = model
            else:
                bi["model_name"] = model
        if isinstance(bi, dict):
            bi["target_path"] = str(target_path) if target_path else str(Path.cwd())
        ci = instantiate(OmegaConf.create(cfg))

    if tool_registry:
        ci.tool_registry = tool_registry
    if tool_executor:
        ci.tool_executor = tool_executor
    if interactive is not None:
        ci.interactive = interactive
    # prompt_renderer / extra_sop_dirs were already passed to the partial ctor in
    # the inject_base path; set them here for the YAML-built-base path.
    if prompt_renderer is not None and not inject_base:
        ci.prompt_renderer = prompt_renderer
    if extra_sop_dirs is not None and not inject_base:
        ci._extra_sop_dirs = list(extra_sop_dirs)
    if allowed_sops is not None and not inject_base:
        ci.allowed_sops = list(allowed_sops)
    if disallowed_sops is not None and not inject_base:
        ci.disallowed_sops = list(disallowed_sops)
    return ci


def force_tools_synchronous(tool_registry: dict) -> None:
    """Force every tool to synchronous mode (no fire-and-forget) so a host CI can
    nest tool calls inline and read their results in the same loop.
    """
    for tool_def in tool_registry.values():
        if hasattr(tool_def, "asynchronous"):
            tool_def.asynchronous = False


def make_tool_executor(
    session_context: dict,
    *,
    tool_dirs: Optional[list] = None,
    af_tools_root: Optional[Path] = None,
):
    """Build an async ``(tool_name, arguments) -> ToolExecutionResult`` dispatch
    closure that resolves each tool's executor from its ``tool.json``.
    """
    root = Path(af_tools_root) if af_tools_root else _AF_TOOLS_ROOT

    async def _tool_executor(tool_name: str, arguments: dict) -> Any:
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
            ToolExecutionResult as _TER,
        )

        search_dirs = [root]
        if tool_dirs:
            search_dirs.extend(Path(d) for d in tool_dirs)
        for tools_root in search_dirs:
            tool_json_path = Path(tools_root) / tool_name / "tool.json"
            if not tool_json_path.exists():
                tool_json_path = Path(tools_root) / tool_name.replace("-", "_") / "tool.json"
            if tool_json_path.exists():
                meta = _json.loads(tool_json_path.read_text())
                executor_ref = meta.get("executor", "")
                if ":" in executor_ref:
                    mod_path, func_name = executor_ref.rsplit(":", 1)
                    mod = importlib.import_module(mod_path)
                    func = getattr(mod, func_name)
                    return await func(arguments, session_context)
                derived_from = meta.get("derived_from")
                if derived_from:
                    from agent_foundation.resources.tools.registry import derived_tool_execute
                    return await derived_tool_execute(
                        arguments, session_context,
                        derived_from=derived_from, tool_name=tool_name,
                    )
        return _TER(result=f"Unknown tool: {tool_name}")

    return _tool_executor
