"""Knowledge validation prompt for LLM-based quality checks."""

VALIDATION_PROMPT = """You are a knowledge validator performing quality checks.

## Content to Validate
Content: {content}
Domain: {domain}
Source: {source}
Created: {created_at}

## Checks to Perform: {checks_to_perform}

For each check, evaluate and provide a pass/fail with reasoning.

### Check Definitions

**correctness**: Is the content factually accurate? Look for errors, outdated facts, or misleading statements.

**authenticity**: Does the source seem credible? Is the content consistent with authoritative sources?

**consistency**: Does the content contradict itself or common knowledge in this domain?

**completeness**: Is critical context missing that would make this misleading if taken alone?

**staleness**: Does the content reference time-sensitive information that may be outdated?

**policy_compliance**: Does the content follow general content guidelines (no hate speech, no illegal advice, etc.)?

## Your Evaluation
Return JSON:
{{
  "passed": ["list of checks that passed"],
  "failed": ["list of checks that failed"],
  "issues": ["specific issues found, one per failed check"],
  "suggestions": ["how to fix each issue"],
  "overall_confidence": 0.0-1.0
}}
"""

VALIDATION_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 500,
}
