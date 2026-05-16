"""XML tag constants used in prompt templates.

These tags form the structural envelope of rendered prompts. Defining them
as constants ensures consistency between templates and tests — a rename
in a template without updating these constants will be caught by the
preflight tests in ``test_prompt_tag_consistency.py``.

Naming convention:
  TAG_{SPACE}_{ROLE}_{PURPOSE}
  e.g. TAG_PLAN_FOLLOWUP_ARTIFACT = the artifact tag in plan/main/followup.jinja2
"""

# --- Plan templates (plan/main/) ---
TAG_PLAN_FOLLOWUP_ARTIFACT = "PriorVersionArtifact"
TAG_PLAN_REVIEW_ARTIFACT = "ArtifactUnderReview"

# --- Implementation templates (implementation/main/) ---
TAG_IMPL_FOLLOWUP_ARTIFACT = "PriorImplementation"
TAG_IMPL_REVIEW_ARTIFACT = "ImplementationUnderReview"
