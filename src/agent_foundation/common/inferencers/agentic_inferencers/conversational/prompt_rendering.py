

"""Default Jinja2-based prompt renderer for ConversationalInferencer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

from rich_python_utils.common_utils.map_helper import merge_mappings

logger = logging.getLogger(__name__)


def _load_yaml_candidates(
    candidates: list[Path],
    recursive: bool = True,
    concatenate_lists: bool = True,
) -> dict[str, Any]:
    """Load and merge YAML from a prioritized list of candidate paths.

    Iterates in reverse order (lowest priority first) so that higher-priority
    candidates override via merge_mappings.
    """
    configs: list[dict[str, Any]] = []
    for candidate in reversed(candidates):
        if candidate.is_file():
            try:
                data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    configs.append(data)
            except Exception as e:
                logger.warning("Failed to load YAML %s: %s", candidate, e)
    return merge_mappings(
        configs, recursive=recursive, concatenate_lists=concatenate_lists,
    )


class JinjaPromptRenderer:
    """Renders conversation prompts using Jinja2 templates.

    Conforms to the PromptRenderer protocol.
    """

    def __init__(
        self,
        template_dir: str,
        template_path: str = "conversation/main/initial.jinja2",
        cross_space_root: str = "",
    ) -> None:
        self._env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=False,
        )
        self._template_path = template_path
        self._template_dir = Path(template_dir)
        # Default cross_space_root to template_dir itself (the root containing
        # all space subdirectories like conversation/, plan/, etc.)
        self._cross_space_root = Path(cross_space_root) if cross_space_root else self._template_dir
        self._template_config: dict[str, Any] | None = None
        self._template_variables: dict[str, Any] | None = None
        self._variable_manager = None

    def render(self, variables: dict[str, Any]) -> str:
        template = self._env.get_template(self._template_path)
        return template.render(**variables)

    def render_string(self, template_str: str, context: dict[str, Any]) -> str:
        """Render an arbitrary template string using this renderer's Environment.

        Uses env.from_string() so the template inherits the same FileSystemLoader,
        globals, and filters as the main template — ensuring consistent behavior
        for feed self-resolution.
        """
        template = self._env.from_string(template_str)
        return template.render(**context)

    @property
    def template_source(self) -> str:
        source = self._env.loader.get_source(self._env, self._template_path)
        return source[0]

    def _resolve_template_dir_and_stem(self) -> tuple[Path, str]:
        template_path = self._template_dir / self._template_path
        return template_path.parent, template_path.stem

    @property
    def template_config(self) -> dict[str, Any]:
        """Load and cache the YAML config adjacent to the template.

        Resolution order (highest priority first):
          1. .<template_basename>.config.yaml  (e.g. .initial.config.yaml)
          2. .config.yaml                      (folder-level default)

        Returns an empty dict if no config file exists.
        """
        if self._template_config is not None:
            return self._template_config

        template_dir, stem = self._resolve_template_dir_and_stem()
        self._template_config = _load_yaml_candidates([
            template_dir / f".{stem}.config.yaml",
            template_dir / ".config.yaml",
        ])
        return self._template_config

    @property
    def variable_manager(self):
        """Get or create the FileBasedVariableManager for this template.

        Lazily creates the manager and loads YAML sidecar files.
        The manager supports set(), clear(), aliases, and fuzzy path resolution.
        """
        if self._variable_manager is not None:
            return self._variable_manager

        from rich_python_utils.common_objects.variable_manager.file_based import (
            FileBasedVariableManager,
        )

        template_dir, stem = self._resolve_template_dir_and_stem()

        # Create manager pointed at the template's _variables directory
        self._variable_manager = FileBasedVariableManager(
            base_path=str(template_dir),
        )

        # Load YAML sidecar files (highest priority first)
        yaml_candidates = [
            template_dir / f".{stem}.variables.yaml",
            template_dir / ".variables.yaml",
        ]
        # Cross-space fallback: check parent/shared root
        if self._cross_space_root and self._cross_space_root.is_dir():
            yaml_candidates.extend([
                self._cross_space_root / f".{stem}.variables.yaml",
                self._cross_space_root / ".variables.yaml",
            ])
        for yaml_candidate in yaml_candidates:
            if yaml_candidate.is_file():
                self._variable_manager.load_yaml_sidecar(yaml_candidate)

        return self._variable_manager

    @property
    def template_variables(self) -> dict[str, Any]:
        """Load and cache variable defaults, with overrides applied.

        Uses FileBasedVariableManager for YAML loading, alias processing,
        and override support. Resolution priority:
          overrides > yaml_sidecar > file-based

        Raises ValueError if any top-level key also exists as a subdirectory
        under _variables/ (conflict between YAML and file-based variables).

        Returns an empty dict if no variables file exists.
        """
        if self._template_variables is not None:
            # If variable_manager has overrides, regenerate
            vm = self.variable_manager
            if hasattr(vm, '_overrides') and vm._overrides:
                return vm.get_all_variables()
            return self._template_variables

        # Get variables from the manager (includes YAML + overrides)
        vm = self.variable_manager
        merged = vm.get_all_variables()

        # Conflict detection: YAML keys vs _variables/ folder names
        template_dir, _ = self._resolve_template_dir_and_stem()
        variables_dir = template_dir / "_variables"
        if variables_dir.is_dir() and merged:
            folder_vars = {d.name for d in variables_dir.iterdir() if d.is_dir()}
            conflicts = folder_vars & set(merged.keys())
            if conflicts:
                raise ValueError(
                    f"Variable(s) {conflicts} defined in both .variables.yaml "
                    f"and _variables/ folder. Use one source only."
                )

        self._template_variables = merged
        return self._template_variables

    def find_sop_file(self) -> Path | None:
        """Find the SOP file in _variables/workflow/sop.{jinja2,j2,md,yaml,yml}.

        Returns the path to the first existing SOP file, or None.
        """
        template_dir, _ = self._resolve_template_dir_and_stem()
        variables_dir = template_dir / "_variables" / "workflow"
        if not variables_dir.is_dir():
            return None
        for ext in (".jinja2", ".j2", ".md", ".yaml", ".yml"):
            candidate = variables_dir / f"sop{ext}"
            if candidate.is_file():
                return candidate
        return None
