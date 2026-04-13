"""Convenience factory functions for loading domain objects from YAML configs.

Usage::

    from agent_foundation.common.configs import load_inferencer

    inferencer = load_inferencer('claude_api')
    inferencer = load_inferencer('claude_api', overrides={'model_id': 'other-model'})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rich_python_utils.config_utils import instantiate, load_config

_YAML_DIR = Path(__file__).parent / "yaml"


def load_inferencer(config_name: str, overrides: Optional[Dict[str, Any]] = None):
    """Load an inferencer from a named YAML config.

    Resolves *config_name* to ``yaml/inferencers/{config_name}.yaml``
    relative to this module's directory.
    """
    config_path = _YAML_DIR / "inferencers" / f"{config_name}.yaml"
    if not config_path.exists():
        available = sorted(
            p.stem for p in (_YAML_DIR / "inferencers").glob("*.yaml")
        ) if (_YAML_DIR / "inferencers").is_dir() else []
        raise FileNotFoundError(
            f"No inferencer config named {config_name!r}. "
            f"Available: {available}"
        )
    cfg = load_config(str(config_path), overrides=overrides)
    return instantiate(cfg)


def load_agent(config_name: str, overrides: Optional[Dict[str, Any]] = None):
    """Load an agent from a named YAML config.

    Most Agent fields are callables/protocols that can't come from YAML.
    This factory provides the structural skeleton — callers inject callable
    collaborators (reasoner, actor, etc.) via *overrides* or post-construction.
    """
    config_path = _YAML_DIR / "agents" / f"{config_name}.yaml"
    if not config_path.exists():
        available = sorted(
            p.stem for p in (_YAML_DIR / "agents").glob("*.yaml")
        ) if (_YAML_DIR / "agents").is_dir() else []
        raise FileNotFoundError(
            f"No agent config named {config_name!r}. "
            f"Available: {available}"
        )
    cfg = load_config(str(config_path), overrides=overrides)
    return instantiate(cfg)
