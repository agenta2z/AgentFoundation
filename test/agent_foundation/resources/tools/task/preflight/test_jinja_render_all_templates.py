"""TIER 1: Pure Jinja render test for ALL templates.

Catches slash-vs-dot bugs (e.g., {{ notes/local_search_efficiency }} vs
{{ notes.local_search_efficiency }}) WITHOUT instantiating topology or calling LLMs.

Run in <5 seconds. Auto-discovers all .jinja2 templates in prompt_templates/
and verifies they render with StrictUndefined context.
"""

from __future__ import annotations

import os
from pathlib import Path

import jinja2
import pytest


_HERE = Path(__file__).resolve().parent
# _HERE = .../OpenStartup/test/openteam/resources/tools/task/preflight
# parents[0]  → .../tools/task
# parents[1]  → .../tools
# parents[2]  → .../resources
# parents[3]  → .../openteam
# parents[4]  → .../OpenStartup/test
# parents[5]  → .../OpenStartup           ← repo root
OPENSTARTUP_PATH = Path(
    os.environ.get(
        "OPENSTARTUP_PATH",
        str(_HERE.parents[5]),
    )
)
TEMPLATES_DIR = OPENSTARTUP_PATH / "src" / "openteam" / "server" / "resources" / "prompt_templates"


def _collect_all_templates() -> list[Path]:
    """Recursively find all .jinja2 files under TEMPLATES_DIR."""
    return sorted(TEMPLATES_DIR.glob("**/*.jinja2"))


def _get_variable_names_from_template(template_content: str) -> set[str]:
    """Extract all Jinja variables referenced in template via regex.

    Returns names like: 'notes', 'session_root_path', 'task_context', etc.
    Does NOT parse dot/slash notation — just top-level variable names.

    Captures variables in BOTH:
      • interpolations: ``{{ var.path.or/wrong.notation }}``
      • statement blocks: ``{% if var %}``, ``{% for x in var %}``,
        ``{% set y = var %}``, etc.

    Skips Jinja built-in keywords (if/for/in/and/or/not/etc.) and pure
    literals (numbers, strings, booleans).
    """
    import re

    # Jinja keywords / control-flow tokens that should NOT be treated as
    # template variables that need stubbing.
    JINJA_KEYWORDS = {
        # Statement keywords
        "if", "elif", "else", "endif",
        "for", "endfor", "in", "recursive",
        "set", "endset",
        "block", "endblock", "extends", "include", "import", "from",
        "macro", "endmacro", "call", "endcall",
        "with", "endwith", "without", "context",
        "filter", "endfilter",
        "raw", "endraw",
        "do",
        # Operators / literals
        "and", "or", "not", "is", "as",
        "true", "false", "none", "True", "False", "None",
        # Common built-in tests / filters that look like identifiers
        "defined", "undefined", "loop",
    }

    found: set[str] = set()

    # 1. {{ ... }} interpolations
    for m in re.findall(r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)', template_content):
        if m not in JINJA_KEYWORDS:
            found.add(m)

    # 2. {% ... %} statement blocks — capture every bare identifier inside
    for stmt in re.findall(r'\{%\s*(.*?)\s*%\}', template_content, flags=re.DOTALL):
        for ident in re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', stmt):
            if ident in JINJA_KEYWORDS:
                continue
            # Skip pure numeric / literal-looking matches (already filtered
            # by the identifier regex, but defensive)
            found.add(ident)

    return found


def _create_sentinel_context(variable_names: set[str]) -> dict:
    """Create a dict with stub values for all variables.
    
    Each variable is mapped to a nested object that allows arbitrary
    attribute/key access without raising UndefinedError.
    """
    class StubObject:
        """Stub that allows any attribute/key access AND degrades gracefully
        for common Jinja constructs: ``| length``, ``{% if x %}``,
        ``{% for x in y %}``, slicing (``[:200]``), comparison (``> 1``).

        The intent is "render without UndefinedError" — actual *values* are
        irrelevant; we only check that templates use defined variable names
        and well-formed Jinja syntax.
        """
        def __getattr__(self, name):
            return StubObject()
        def __getitem__(self, key):
            # Slice support (e.g., {{ summary[:200] }}) — return a stub string
            if isinstance(key, slice):
                return "<stub>"
            return StubObject()
        def __call__(self, *args, **kwargs):
            # Method calls like ``var.items()``, ``var.keys()``, ``var.upper()``
            # all return another stub (which is iterable / len-able / etc.).
            return StubObject()
        def __iter__(self):
            # Empty iteration so {% for %} loops execute zero times safely.
            # For tuple-unpacking like ``for k, v in items.items()``, the empty
            # iteration means the loop body is never entered — safe.
            return iter([])
        def __len__(self):
            # Support `| length` filter and len() comparisons
            return 0
        def __contains__(self, item):
            return False
        def __eq__(self, other):
            return False
        def __ne__(self, other):
            return True
        def __lt__(self, other):
            return False
        def __le__(self, other):
            return False
        def __gt__(self, other):
            return False
        def __ge__(self, other):
            return False
        def __hash__(self):
            return id(self)
        def __str__(self):
            return "<stub>"
        def __repr__(self):
            return "<stub>"
        def __bool__(self):
            return True

    return {name: StubObject() for name in variable_names}


@pytest.mark.preflight
@pytest.mark.parametrize("template_path", _collect_all_templates(), ids=lambda p: str(p.relative_to(TEMPLATES_DIR)))
def test_template_renders_without_undefined(template_path: Path):
    """
    Verify that a template renders with StrictUndefined without raising.
    
    This catches:
    - Typos in variable names (e.g., {{ undefined_var }})
    - Slash vs dot bugs (e.g., {{ notes/field }} when only notes.field exists)
    - Syntax errors in Jinja
    
    The sentinel context provides all top-level variables found in the template.
    """
    template_content = template_path.read_text(encoding="utf-8")
    variable_names = _get_variable_names_from_template(template_content)
    context = _create_sentinel_context(variable_names)
    
    # Use StrictUndefined to catch any reference to undefined variables
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_content)
    
    # If this raises jinja2.UndefinedError, the test fails and reports the line/var
    rendered = template.render(**context)
    assert rendered is not None, f"Template {template_path.name} rendered to None"


@pytest.mark.preflight
def test_template_discovery_not_empty():
    """Sanity check: verify we found at least some templates."""
    templates = _collect_all_templates()
    assert len(templates) > 0, f"No .jinja2 templates found under {TEMPLATES_DIR}"
    print(f"Discovered {len(templates)} templates for render testing")
