

"""Plan-phase prompt templates for DualInferencer testing.

Adapted from rankevolve/src/dual_agent/prompts/templates/ with variable
substitutions for DualInferencer:
    input              → original user request
    proposal           → cleaned response from previous agent (inside <Response> tags)we
    issues             → pre-serialized JSON string of review issues
    reasoning          → reviewer's overall assessment
    counter_feedback   → serialized counter-feedback JSON (or None)
    enable_counter_feedback → bool (fix templates only)
    output_path        → concrete file path for writing output
    iteration          → current iteration (1-based) within the consensus attempt
    attempt            → current consensus attempt (1-based)
"""

PLAN_INITIAL_TEMPLATE = """\
You are tasked with creating a comprehensive plan for the following user request:

**Original User Request:**
<UserRequest>
{{ input }}
</UserRequest>

---

## Your Task: Create a Detailed Plan

Create a comprehensive, implementable plan that covers:

1. **High-Level Approach**: Overall strategy and architecture decisions
2. **Files to Create/Modify**: Specific files with their intended changes
3. **Key Implementation Steps**: Ordered list of concrete tasks
4. **Potential Risks and Mitigations**: What could go wrong and how to prevent it
5. **Testing Strategy**: How to verify the implementation works

## Deep Investigation Requirements

- You MUST thoroughly look into all code related to the user ask. You MUST conduct a deep, thorough, and accurate investigation with critical thinking.
- Don't assume. You MUST validate by reading actual code.

## Output Requirements

- Write your complete plan in markdown format to: `{{ output_path }}`
- The plan should be human readable and structured with details, that another human or agent could follow to implement later.

## Response Format

Wrap your response in `<Response>` tags:

<Response>
[Your natural language summary of what you have investigated and planned]
</Response>
"""

PLAN_REVIEW_TEMPLATE = """\
You are tasked with carefully and thoroughly reviewing a plan proposed by another agent for the following user request:

**Original User Request:**
<UserRequest>
{{ input }}
</UserRequest>

**Latest Proposed Plan By Another Agent:**
<ProposedPlan>
{{ main_response }}

The proposal is available at file path: `{{ output_path }}`
</ProposedPlan>

**Current Review Iteration Index**: {{ round_index }}

---

## Your Task: Critical Review

You are an expert code reviewer. Your job is to **carefully, thoroughly double-check with critical-thinking** whether this plan is:

1. **Complete**:
   * Does it cover all aspects of the user's request?
   * Are there dependencies or integrations the plan forgot?
2. **Correct**:
   * Are the assumptions about the codebase correct?
   * Are the proposed approaches technically sound?
   * Are there mis-handled edge cases?
3. **Fair**:
   * Is there hacky implementation when there is more elegant approach?
   * Is it over-engineering or making unnecessary breaking changes when there is a simpler but equally effective approach?

Above are only example checks. In general, check for any issues, problems or missing items.

## Deep Investigation Requirements

You MUST make a careful, deep, thorough and accurate investigation with critical thinking.

## Response Format

Wrap your response in `<Response>` tags with a JSON review.

For each issue found, assign an `issue_id` in the format `<round_index>-<issue_index>` (e.g. `2-1`, `2-2`, … for round 2) and rate its `severity`:
- **CRITICAL**: Blocks implementation; plan cannot proceed as-is
- **MAJOR**: Significant problem requiring substantial revision
- **MINOR**: Small improvements needed but plan is workable
- **COSMETIC**: Trivial style/naming suggestions only

You then judge the `overall_severity` of the proposed plan, and then judge if you `approve` the plan.
You approve the plan only if all remaining issues are trivial COSMETIC or there are no issues.

If the plan has no meaningful issues, return an empty `issues` array and set `approve: true`.

<Response>
[Your natural language analysis of the plan]
```json
{
  "issues": [
    {
      "issue_id": "<round_index>-<issue_index>",
      "severity": "<CRITICAL|MAJOR|MINOR|COSMETIC>",
      "description": "<detailed description of the issue>",
      "location": "<which section of the plan>",
      "suggestion": "<how to fix this issue>"
    }
  ],
  "reasoning": "<overall assessment and reasoning for your verdict>",
  "overall_severity": "<CRITICAL|MAJOR|MINOR|COSMETIC|NONE>",
  "approve": <true if no CRITICAL/MAJOR issues>
}
```
</Response>
"""

PLAN_FOLLOWUP_TEMPLATE = """\
You previously generated a plan for the following user request and received feedback from a reviewing agent.

**Original User Request:**
<UserRequest>
{{ input }}
</UserRequest>

**Your Current Proposed Plan:**
<ProposedPlan>
{{ main_response }}
</ProposedPlan>

**Reviewer Feedback:**
<ReviewerFeedback>
{{ reviewer_response }}
</ReviewerFeedback>

**Current Review Iteration Index**: {{ round_index }}

---

## Your Task: Evaluate Each Feedback Item & Refine the Plan

### Step 1: Critical Evaluation of Feedback

The reviewer may be wrong. For **each** issue raised:
- Verify the claim against the actual codebase — do not accept on faith.
- Accept if the issue is real and actionable.
- Reject with specific evidence if the reviewer is mistaken.

### Step 2: Refine the Plan

For each accepted issue, carefully revise the plan while ensuring:
- All CRITICAL and MAJOR issues are resolved (required).
- MINOR issues are addressed where practical (recommended).
- No new problems are introduced by your changes.
- The refined plan remains internally consistent.

## Deep Investigation Requirements

You MUST conduct a careful, deep, and accurate investigation of each feedback item with critical thinking. Validate claims by reading actual code — do not assume.

## Output Requirements

Write your improved plan in markdown format to: `{{ output_path }}`

Meanwhile, wrap a structured response in `<Response>` tags:

<Response>

[Concise summary of what changed and why]

```json
{
  "items": [
    {
      "issue_id": "<ID from feedback>",
      "accepted": true | false,
      "rationale": "<if accepted: what you fixed and how> | <if rejected: evidence why the reviewer is wrong>"
    }
  ],
  "summary": "<brief tally: N accepted, M rejected, with one-line reasoning>"
}

```
</Response>
"""
