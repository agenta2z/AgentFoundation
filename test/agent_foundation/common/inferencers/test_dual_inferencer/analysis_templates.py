

"""Analysis prompt templates for PlanThenImplementInferencer test script.

These templates are used by the analysis phase of the multi-iteration
refinement loop. The analyzer reviews implementation output and test/benchmark
results to decide whether another iteration is needed.

Templates are written to the workspace's prompt_templates/analysis/main/
directory and picked up by TemplateManager.
"""

ANALYSIS_INITIAL_TEMPLATE = """\
You are tasked with analyzing the results of iteration {{ round_index }} of a
multi-iteration plan-then-implement process.

**Analysis Input:**
<AnalysisInput>
{{ input }}
</AnalysisInput>

---

## Your Task

Review the implementation output, test results, and benchmark results above.
Determine whether the implementation meets the requirements or if further
iterations are needed.

Consider:
1. **Correctness**: Does the implementation correctly address the original request?
2. **Completeness**: Are all aspects covered?
3. **Test Results**: Do tests pass? Are there failures?
4. **Benchmark Results**: Do benchmarks meet performance expectations?
5. **Code Quality**: Is the code well-structured?

## Output Requirements

- Write your **full analysis** in markdown format to: `{{ output_path }}`
- Be thorough and reference specific findings.

## Response Format

Wrap a JSON decision object in `<Response>` tags:

<Response>
```json
{
  "should_continue": true/false,
  "summary": "Brief summary of findings",
  "next_iteration_request": "What to focus on in the next iteration (if continuing)"
}
```
</Response>

Set `should_continue` to `true` only if there are significant issues requiring another iteration.
"""

ANALYSIS_REVIEW_TEMPLATE = """\
You are reviewing an analysis produced by another agent.

**Analysis Input:**
<AnalysisInput>
{{ input }}
</AnalysisInput>

**Proposed Analysis:**
<ProposedAnalysis>
{{ main_response }}

The analysis is at: `{{ output_path }}`.
</ProposedAnalysis>

**Current Review Iteration Index**: {{ round_index }}

{% if counter_feedback %}
**Previous Counter-Feedback from Analyzer:**
<CounterFeedback>
{{ counter_feedback }}
</CounterFeedback>
{% endif %}

---

## Your Task: Review the Analysis

Check whether the analysis is:
1. **Accurate**: Are findings factually correct? Are results interpreted correctly?
2. **Complete**: Does it cover all aspects?
3. **Sound Decision**: Is `should_continue` justified by the evidence?
4. **Actionable**: Is `next_iteration_request` specific enough?

## Response Format

<Response>
[Your review]
```json
{
  "issues": [
    {
      "issue_id": "<round_index>-<issue_index>",
      "severity": "<CRITICAL|MAJOR|MINOR|COSMETIC>",
      "description": "<description>",
      "location": "<section>",
      "suggestion": "<fix>"
    }
  ],
  "reasoning": "<overall assessment>",
  "overall_severity": "<CRITICAL|MAJOR|MINOR|COSMETIC|NONE>",
  "approve": true/false
}
```
</Response>
"""

ANALYSIS_FOLLOWUP_TEMPLATE = """\
You previously generated an analysis and received reviewer feedback.

**Analysis Input:**
<AnalysisInput>
{{ input }}
</AnalysisInput>

**Your Current Analysis:**
<ProposedAnalysis>
{{ main_response }}
</ProposedAnalysis>

**Reviewer Feedback:**
<ReviewerFeedback>
{{ reviewer_response }}
</ReviewerFeedback>

**Current Review Iteration Index**: {{ round_index }}

---

## Your Task: Evaluate Feedback & Refine

For each issue: accept with fixes or reject with evidence.
Reconsider `should_continue` in light of feedback.

Write improved analysis to: `{{ output_path }}`

<Response>
[Summary of changes]
```json
{
  "items": [
    {
      "issue_id": "<ID>",
      "accepted": true/false,
      "rationale": "<reasoning>"
    }
  ],
  "summary": "<tally>"
}
```
</Response>
"""
