"""TemplateManagerPromptRenderer — adapter exposing the duck-typed
prompt_renderer API backed by TemplateManager from RichPythonUtils.

This adapter is the SOLE seam between ConversationalInferencer and
TemplateManager. All API translation lives here, NOT in
ConversationalInferencer's render path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from attr import attrib, attrs

from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
    TemplateManager,
)


def _load_yaml_cascade(candidates: list[Path]) -> dict:
    """Load the first existing YAML file from the cascade."""
    for candidate in candidates:
        if candidate.is_file():
            try:
                with open(candidate) as fh:
                    loaded = yaml.safe_load(fh) or {}
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                continue
    return {}


@attrs(slots=False, eq=False)
class TemplateManagerPromptRenderer:
    """Duck-typed wrapper exposing the 7-method prompt_renderer API.

    Members consumed by ConversationalInferencer:
      - render(feed) -> str
      - render_string(template_str, context) -> str
      - template_variables: dict
      - variable_manager: Any
      - template_source: str
      - template_config: dict
      - find_sop_file() -> Optional[Path]
    """

    template_manager: TemplateManager = attrib()
    template_key: str = attrib(default="initial")

    def render(self, feed: Mapping[str, Any]) -> str:
        """Render the active template with the given feed dict."""
        return self.template_manager(self.template_key, **dict(feed))

    def render_string(self, template_str: str, context: Mapping[str, Any]) -> str:
        """Render an arbitrary Jinja2 template string.

        Routes through TemplateManager.__call__() via transient-key
        registration so shared macros/filters/includes work correctly.
        """
        if not template_str:
            return template_str
        transient_key = f"__transient__{id(template_str):x}"
        try:
            tmplates = self.template_manager.templates
            if tmplates is None:
                self.template_manager.templates = {}
                tmplates = self.template_manager.templates
            if "__transient__" not in tmplates:
                tmplates["__transient__"] = {}
            tmplates["__transient__"][transient_key] = template_str
            return self.template_manager(
                transient_key,
                active_template_root_space="__transient__",
                active_template_type="",
                **dict(context),
            )
        except Exception:
            # Fallback: use the formatter directly if transient registration fails
            return self.template_manager.template_formatter(
                template_str, **dict(context)
            )
        finally:
            try:
                if tmplates and "__transient__" in tmplates:
                    tmplates["__transient__"].pop(transient_key, None)
            except Exception:
                pass

    @property
    def template_variables(self) -> dict:
        """Resolved variable cascade for the active template."""
        try:
            return self.template_manager.load_variables(self.template_key) or {}
        except Exception:
            return {}

    @property
    def variable_manager(self):
        """Expose the underlying VariableManager."""
        return getattr(self.template_manager, "_variable_loader", None)

    @property
    def template_source(self) -> str:
        """Raw Jinja2 source of the active template."""
        raw = self.template_manager.get_raw_template(self.template_key)
        return str(raw) if raw else ""

    @property
    def template_config(self) -> dict:
        """YAML sidecar config adjacent to the active template.

        Reconstructs path from _OriginTaggedStr._origin_root + template key.
        """
        raw = self.template_manager.get_raw_template(self.template_key)
        if raw is None:
            return {}
        origin_root = getattr(raw, "_origin_root", None)
        if origin_root is None:
            return {}
        root_space = self.template_manager.active_template_root_space or ""
        ttype = self.template_manager.active_template_type or ""
        for ext in (".jinja2", ".j2", ".md", ".yaml", ".yml"):
            template_path = (
                Path(origin_root) / root_space / ttype / f"{self.template_key}{ext}"
            )
            if template_path.is_file():
                return _load_yaml_cascade([
                    template_path.parent / f".{template_path.stem}.config.yaml",
                    template_path.parent / ".config.yaml",
                ])
        return {}

    def find_sop_file(self) -> Optional[Path]:
        """SOP discovery is owned by SOPRegistry. Returns None."""
        return None
