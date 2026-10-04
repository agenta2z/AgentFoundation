"""Shared SOP/identity prompt sections.

The identity and SOP sections of ``conversation/main/initial.jinja2`` live in
``conversation/sections/*.jinja2`` so the classic template and the native
orchestrator's templates render the same wording from the same feed. Jinja
``{% include %}`` is unavailable (``TemplateManager`` renders via
``from_string``), so the sections are rendered here and handed to the parent
template as ``section_<name>`` feed values.
"""

from __future__ import annotations

from typing import Any, Mapping

from rich_python_utils.string_utils.formatting.jinja2_format import extract_variables

SECTION_NAMES: tuple[str, ...] = (
    "identity",
    "sop_catalog",
    "sop_paused",
    "sop_inprogress",
    "sop_active",
)
SECTIONS_ROOT_SPACE = "conversation"
SECTIONS_TEMPLATE_TYPE = "sections"


class MissingSectionTemplate(LookupError):
    """A shared section template could not be resolved by the TemplateManager."""


def sections_used_by(template_source: str) -> tuple[str, ...]:
    """Names of the shared sections a template references as ``section_<name>``."""
    variables = extract_variables(template_source or "")
    return tuple(n for n in SECTION_NAMES if f"section_{n}" in variables)


def render_sop_sections(
    template_manager: Any,
    feed: Mapping[str, Any],
    names: tuple[str, ...] = SECTION_NAMES,
) -> dict[str, str]:
    """Render the named shared sections with ``feed``; returns
    ``{"section_<name>": text}``.

    Raises ``MissingSectionTemplate`` instead of silently rendering the
    manager's default template when a section is not on any template root.
    """
    rendered: dict[str, str] = {}
    for name in names:
        raw = template_manager.get_raw_template(
            name,
            active_template_type=SECTIONS_TEMPLATE_TYPE,
            active_template_root_space=SECTIONS_ROOT_SPACE,
        )
        if not raw or raw == template_manager.default_template:
            raise MissingSectionTemplate(
                f"Section template '{SECTIONS_ROOT_SPACE}/{SECTIONS_TEMPLATE_TYPE}/"
                f"{name}' was not found on any template root of {template_manager!r}."
            )
        rendered[f"section_{name}"] = template_manager(
            name,
            feed=dict(feed),
            active_template_type=SECTIONS_TEMPLATE_TYPE,
            active_template_root_space=SECTIONS_ROOT_SPACE,
        )
    return rendered
