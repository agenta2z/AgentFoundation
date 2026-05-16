"""Preflight: assert prompt templates contain the expected XML tag constants.

Catches silent regressions where a template tag is renamed without
updating the constant (or vice versa). Each test reads the actual
template file and asserts the tag constant appears as both opening
and closing XML tags.
"""

import os
import unittest
from pathlib import Path

from agent_foundation.common.inferencers.constants.prompt_tags import (
    TAG_PLAN_FOLLOWUP_ARTIFACT,
    TAG_PLAN_REVIEW_ARTIFACT,
    TAG_IMPL_FOLLOWUP_ARTIFACT,
    TAG_IMPL_REVIEW_ARTIFACT,
)

# Resolve prompt_templates root relative to the source tree.
_SRC = Path(__file__).resolve().parents[4] / "src"
_TEMPLATES = _SRC / "agent_foundation" / "resources" / "prompt_templates"


def _read_template(space: str, key: str) -> str:
    path = _TEMPLATES / space / "main" / f"{key}.jinja2"
    if not path.is_file():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


class TestPlanTemplateTags(unittest.TestCase):
    """Plan templates must contain the correct artifact XML tags."""

    def test_followup_has_prior_version_artifact_tag(self):
        content = _read_template("plan", "followup")
        self.assertIn(f"<{TAG_PLAN_FOLLOWUP_ARTIFACT}>", content)
        self.assertIn(f"</{TAG_PLAN_FOLLOWUP_ARTIFACT}>", content)

    def test_review_has_artifact_under_review_tag(self):
        content = _read_template("plan", "review")
        self.assertIn(f"<{TAG_PLAN_REVIEW_ARTIFACT}>", content)
        self.assertIn(f"</{TAG_PLAN_REVIEW_ARTIFACT}>", content)


class TestImplementationTemplateTags(unittest.TestCase):
    """Implementation templates must contain the correct artifact XML tags."""

    def test_followup_has_prior_implementation_tag(self):
        content = _read_template("implementation", "followup")
        self.assertIn(f"<{TAG_IMPL_FOLLOWUP_ARTIFACT}>", content)
        self.assertIn(f"</{TAG_IMPL_FOLLOWUP_ARTIFACT}>", content)

    def test_review_has_implementation_under_review_tag(self):
        content = _read_template("implementation", "review")
        self.assertIn(f"<{TAG_IMPL_REVIEW_ARTIFACT}>", content)
        self.assertIn(f"</{TAG_IMPL_REVIEW_ARTIFACT}>", content)


if __name__ == "__main__":
    unittest.main()
