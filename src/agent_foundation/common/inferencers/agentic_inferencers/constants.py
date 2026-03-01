DEFAULT_PLACEHOLDER_INFERENCE_PROMPT = "prompt"
DEFAULT_PLACEHOLDER_INFERENCE_RESPONSE = "response"

DEFAULT_SELF_REFLECTION_PROMPT_TEMPLATE = """You are a smart agent helping users handle their asks.
You have been provided a prompt with the the user ask and all the instructions, and you made a response.

Here was the prompt provided to you.
<Prompt>
{{prompt}}
</Prompt>

Here was your response.
<Response>
{{response}}
</Response>

Have a deep Reflection on your above response as well as the reasoning process (if any). Check if there is any error or mistakes, and compose an ImprovedResponse. Following are common issues,
- Did not follow the NOTES (e.g. HUMAN ASSISTANT MINDSET, other mindset PRINCIPALS, and other NOTES on AvailableActions, Clarification, AlternativeActions, search action, search results, and specific action regulations)
- Did not perform sufficient reasoning/examination required by the NOTES
- Be very careful not to drop or deprioritize reasonable existing Actions when reviewing AlternativeAction. If you decide to explore more in parallel, append to the AlternativeActions.
- Be very careful not to trigger MakeAnswer action when only high-level summaries and snippets are available and missing critical details (e.g., search results, short overview paragraphs).
- Carefully check if you triggered redundant search actions that have been performed earlier.
- Carefully check the quantities are exact (e.g., when making order, customizing recipe, etc., and other scenarios when exact quantities are critical)
- Carefully check the time are exact (e.g., when making appointment, booking flights, shipping/delivery order, etc., and other scenarios when exact time are critical)
- Carefully check if Thought identifies multiple viable options but ImmediateNextActions does not include them; there should be good reason to drop additional options (such as explicit user preference is specified through Context or Conversation)

NOTES on Search results
- Check if you miss links like "more places", "more businesses", "see all reviews", etc. that can reveal more recommendations, reviews and ratings.
- YOU SHOULD still prioritize actions related to recommendations, reviews and ratings, including above "more places", "more businesses", "see all reviews", etc., even after you decide to have more parallel actions in AlternativeActions.

When there is no obvious error or mistakes, you can largely copy the original response as your ImprovedResponse.
You output follow this format:
<Reflection>
[your reflection and error checking on your response]
</Reflection>
<ImprovedResponse>
[compose your improved response; follow exactly the same format as your previous Response]
</ImprovedResponse>
"""

# DualInferencer placeholder keys
DEFAULT_PLACEHOLDER_DUAL_INPUT = "input"
DEFAULT_PLACEHOLDER_DUAL_PROPOSAL = "proposal"
DEFAULT_PLACEHOLDER_DUAL_ISSUES = "issues"
DEFAULT_PLACEHOLDER_DUAL_REASONING = "reasoning"
DEFAULT_PLACEHOLDER_DUAL_COUNTER_FEEDBACK = "counter_feedback"

DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE = """You are a critical reviewer. Review the following proposal against the original request.

<OriginalRequest>
{{ input }}
</OriginalRequest>

<Proposal>
{{ proposal }}
</Proposal>

{% if counter_feedback %}
<PreviousCounterFeedback>
{{ counter_feedback }}
</PreviousCounterFeedback>

The author has provided counter-feedback on your previous review. Consider their reasoning carefully.
Only re-raise issues if you have NEW evidence or the author misunderstood.
{% endif %}

Review the proposal thoroughly for completeness, correctness, feasibility, safety, and consistency.

Severity levels:
- CRITICAL: Blocks implementation; cannot proceed as-is
- MAJOR: Significant problem requiring substantial revision
- MINOR: Small improvements needed but plan is workable
- COSMETIC: Style/naming suggestions only
- NONE: No issues found

Verdict rules:
- Set "approved" to true if there are no CRITICAL or MAJOR issues
- Set "severity" to the highest severity among all issues, or "NONE" if there are no issues

Respond with structured JSON:

If the proposal is solid:
```json
{
  "approved": true,
  "severity": "NONE",
  "issues": [],
  "reasoning": "<overall assessment>"
}
```

If issues are found:
```json
{
  "approved": false,
  "severity": "MAJOR",
  "issues": [
    {
      "severity": "<CRITICAL|MAJOR|MINOR|COSMETIC>",
      "category": "<category>",
      "description": "<detailed description>",
      "location": "<where in the proposal>",
      "suggestion": "<how to fix>"
    }
  ],
  "reasoning": "<overall assessment>"
}
```"""

DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE = """You are an expert engineer. Address the reviewer's feedback on your proposal.

<OriginalRequest>
{{ input }}
</OriginalRequest>

<CurrentProposal>
{{ proposal }}
</CurrentProposal>

<ReviewIssues>
{{ issues }}
</ReviewIssues>

<ReviewerReasoning>
{{ reasoning }}
</ReviewerReasoning>

{% if enable_counter_feedback %}
For each issue, critically evaluate whether it is valid. Produce counter-feedback:
```json
{
  "items": [
    {
      "issue_id": "<ID>",
      "accepted": <true|false>,
      "action_taken": "<what you did, if accepted>",
      "rejection_reason": "<why rejected, if applicable>"
    }
  ],
  "summary": "<brief summary>"
}
```
{% endif %}

Then provide your complete improved proposal inside these tags:
<ImprovedProposal>
[your full improved proposal here]
</ImprovedProposal>"""

