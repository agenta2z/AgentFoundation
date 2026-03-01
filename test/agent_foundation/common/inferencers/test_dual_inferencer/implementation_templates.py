

"""Implementation-phase prompt templates for DualInferencer testing.

Adapted from rankevolve/src/dual_agent/prompts/templates/ with variable
substitutions for DualInferencer:
    input              → original user request
    proposal           → cleaned response from previous agent (inside <Response> tags)
    issues             → pre-serialized JSON string of review issues
    reasoning          → reviewer's overall assessment
    counter_feedback   → serialized counter-feedback JSON (or None)
    enable_counter_feedback → bool (fix templates only)
    output_path        → concrete file path for writing output
    iteration          → current iteration (1-based) within the consensus attempt
    attempt            → current consensus attempt (1-based)
"""

IMPLEMENTATION_INITIAL_TEMPLATE = """\
{# Implementation Template — adapted for DualInferencer #}
# Implementation Assignment

## Context

**Original User Request:**
{{ input }}

---

## Your Task: Implement the Request

You are an expert software engineer. Your job is to:

1. **Implement the request** step by step
2. **Follow best practices** for code quality and testing
3. **Document any issues** you discover

## Implementation Guidelines

### Code Quality
- Follow existing code style and conventions in the codebase
- Add appropriate type hints and docstrings
- Write clean, maintainable code
- Handle edge cases and errors appropriately

### Testing
- Implement tests for key functionality
- Ensure tests cover the important paths
- Make sure existing tests still pass

## Output Requirements

1. **Write your implementation summary** to: `{{ output_path }}`

   The summary file should contain:
   - Files created/modified and what was changed
   - Testing status
   - Any potential concerns

2. **Wrap your final response** in `<Response>` tags:

<Response>
[Your natural language summary of what you implemented]
</Response>

---

**Example summary structure** (write this to the output file):
```markdown
# Implementation Summary

## 1. Files Created/Modified
- `path/to/file1.py`: [What was changed]
- `path/to/file2.py`: [What was changed]

## 2. Testing Status
- What tests were added/modified?
- Do all tests pass?

## 3. Potential Concerns
- Any areas that might need review?
- Any edge cases you're uncertain about?
```
"""

IMPLEMENTATION_REVIEW_TEMPLATE = """\
{# Implementation Review Template — adapted for DualInferencer #}
# Implementation Review Assignment

## Context

**Original User Request:**
{{ input }}

**Implementing Agent's Response:**
{{ proposal }}

{% if output_path %}
**Implementation Summary File:** `{{ output_path }}`
Read the summary file to review the implementation details.
{% endif %}

{% if counter_feedback %}
---

## Previous Counter-Feedback

The fixer agent has provided counter-feedback on your previous review. Consider their reasoning carefully:

<counter_feedback>
{{ counter_feedback }}
</counter_feedback>
{% endif %}

---

## Your Task: Critical Code Review with Deep Judgment

You are an expert code reviewer. Your job is to **carefully, thoroughly double-check with critical-thinking** whether this implementation:

1. **Follows the Plan**: Does it implement what was approved?
2. **Correct Logic**: Is the code logic sound and bug-free?
3. **Code Quality**: Is it well-structured, readable, and maintainable?
4. **Error Handling**: Are errors handled appropriately?
5. **Edge Cases**: Are edge cases considered?
6. **Testing**: Are there adequate tests?
7. **Performance**: Are there any performance concerns?
8. **Security**: Are there any security vulnerabilities?

## Deep Investigation Requirements

**You MUST make a deep, thorough and accurate investigation with critical thinking:**

- **Trace the logic**: Walk through the code mentally—does it do what it claims?
- **Check for bugs**: Look for off-by-one errors, null pointers, race conditions
- **Review error handling**: What happens when things go wrong?
- **Assess test coverage**: Are the important paths tested?
- **Consider edge cases**: What about empty inputs, large inputs, invalid inputs?
- **Verify types**: Are type annotations correct? Would Pyre/mypy pass?
- **Integration points**: Does this code integrate correctly with the existing codebase?
- **Resource management**: Are resources (files, connections, memory) properly managed?

{% if counter_feedback %}
## Counter-Feedback Consideration

The fixer has rejected some of your previous feedback with reasoning. You MUST:
- Carefully consider their counter-arguments
- Accept valid rejections (do NOT re-raise issues that were legitimately rejected)
- Only re-raise issues if you have NEW evidence or the fixer misunderstood
{% endif %}

## Issues to Flag

For each issue found, rate its severity:
- **CRITICAL**: Code has bugs or security issues that must be fixed
- **MAJOR**: Significant problems affecting functionality or maintainability
- **MINOR**: Small improvements for better code quality
- **COSMETIC**: Style or naming suggestions
- **NONE**: No issues found; the implementation is solid

## Verdict

Based on your review, determine your overall verdict:
- Set `"approved": true` if there are **no CRITICAL or MAJOR** issues (the implementation can proceed)
- Set `"approved": false` if there are **any CRITICAL or MAJOR** issues (the implementation needs revision)
- Set `"severity"` to the **highest** severity among all issues found, or `"NONE"` if there are no issues

## Response Format

Wrap your response in `<Response>` tags with a JSON review:

If the implementation is solid and has no issues (or only COSMETIC/MINOR ones):

<Response>
[Your natural language analysis of the implementation]

```json
{
  "approved": true,
  "severity": "NONE",
  "issues": [],
  "reasoning": "<overall assessment and reasoning for your verdict>",
  "tests_to_add": []
}
```
</Response>

If issues are found:

<Response>
[Your natural language analysis of the implementation]

```json
{
  "approved": false,
  "severity": "MAJOR",
  "issues": [
    {
      "severity": "<CRITICAL|MAJOR|MINOR|COSMETIC>",
      "category": "<correctness|logic|quality|error_handling|testing|performance|security>",
      "description": "<detailed description of the issue>",
      "location": "<file and line number or function name>",
      "suggestion": "<how to fix this issue>"
    }
  ],
  "reasoning": "<overall assessment and reasoning for your verdict>",
  "tests_to_add": ["<suggested tests to ensure correctness>"]
}
```
</Response>

Remember: Be thorough but fair. Your goal is to improve code quality, not find fault.
"""

IMPLEMENTATION_FOLLOWUP_TEMPLATE = """\
{# Implementation Refinement Template — adapted for DualInferencer #}
# Implementation Refinement Assignment

## Context

**Original User Request:**
{{ input }}

**Your Current Response:**
{{ proposal }}

{% if output_path %}
**Current Implementation Summary:** `{{ output_path }}`
Read the summary file to see the current implementation details.
{% endif %}

**Review Feedback to Address:**

{{ issues }}

**Reviewer's Overall Assessment:**
{{ reasoning }}

---

## Your Task: Critical Evaluation and Fix

You are an expert software engineer. Your job is to:

1. **Critically evaluate each feedback item** — The reviewer MIGHT be wrong
2. **Apply valid fixes** to the implementation
{% if enable_counter_feedback %}
3. **Produce counter-feedback** for invalid issues
{% endif %}

## Critical Thinking Process

For EACH issue in the feedback:

### Step 1: Understand the Issue
- What is the reviewer claiming is wrong?
- What evidence do they provide?

### Step 2: Verify the Claim
- Is this actually a bug or problem?
- Did the reviewer miss important context?
- Is the code behavior intentional?

### Step 3: Decide: Accept or Reject

**ACCEPT** if:
- The issue is a real bug or problem
- The suggested fix improves the code

**REJECT** if:
- The reviewer misunderstood the code or requirements
- The behavior is intentional and correct
- The suggested fix would cause other problems
- The issue is based on incorrect assumptions

---

## Output Requirements

1. **Write your improved implementation summary** to: `{{ output_path }}`
2. **Wrap your response** in `<Response>` tags:

<Response>
[Summary of changes you made to the implementation]

{% if enable_counter_feedback %}
```json
{
  "items": [
    {
      "issue_id": "<ID from feedback>",
      "accepted": <true|false>,
      "action_taken": "<what you did to fix it, if accepted>",
      "rejection_reason": "<why you rejected it, if rejected>"
    }
  ],
  "summary": "<brief summary of what was accepted/rejected>"
}
```
{% endif %}
</Response>

## Refinement Guidelines

1. **Address every issue**: Work through each issue systematically
2. **Explain your changes**: For each issue, briefly note how you addressed it
3. **Don't introduce new problems**: Be careful not to break things that were working
4. **Ask if unclear**: If any feedback is ambiguous, note your interpretation

## Critical Thinking Checklist

Before submitting your refined version:
- [ ] Have I addressed all CRITICAL issues? (required)
- [ ] Have I addressed all MAJOR issues? (required)
- [ ] Review all CRITICAL and MAJOR issues carefully.
- [ ] Have I considered MINOR issues? (recommended)
- [ ] Have I avoided introducing new bugs?
- [ ] Is my refinement internally consistent?

---

## Remember

- **Be thorough**: Don't blindly accept or reject. Verify each claim.
- **Be fair**: Give credit where due, but stand your ground when the feedback is wrong.
- **Be specific**: When rejecting, explain exactly why with evidence from the code.
"""
