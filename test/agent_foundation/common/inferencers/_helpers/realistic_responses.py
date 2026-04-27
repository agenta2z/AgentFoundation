"""Canned domain-appropriate response strings for Layer 1 tests.

These exercise parsing/routing logic in inferencers — they're more realistic
than 'mock_response_1' so tests catch parser regressions, not just call counts.
"""

# ---------------------------------------------------------------------------
# DualInferencer review responses
# ---------------------------------------------------------------------------

DUAL_REVIEW_RESPONSE_MAJOR = """## Review
Issues found: 2
Severity: MAJOR

1. Missing error handling in the API endpoint
2. SQL injection vulnerability in query builder
"""

DUAL_REVIEW_RESPONSE_COSMETIC = """## Review
Issues found: 1
Severity: COSMETIC

1. Variable name could be more descriptive
"""

DUAL_REVIEW_RESPONSE_APPROVED = """## Review
Severity: COSMETIC
Approved: true
"""

DUAL_FIX_RESPONSE = """I've addressed the review issues:
1. Added try/except around API call
2. Switched to parameterized queries
"""

DUAL_COUNTER_FEEDBACK = """## Counter-feedback
The reported issue #1 is intentional — the function is meant to fail
fast for invalid inputs.
"""


# ---------------------------------------------------------------------------
# BTA breakdown responses
# ---------------------------------------------------------------------------

BTA_BREAKDOWN_NUMBERED = """1. Research authentication best practices
2. Design the database schema
3. Implement the REST API endpoints
"""

BTA_BREAKDOWN_JSON = """[
  {"description": "Research auth", "task_type": "research"},
  {"description": "Design schema", "task_type": "design"},
  {"description": "Implement API", "task_type": "implementation"}
]"""

BTA_BREAKDOWN_WITH_TODOS = """[
  {"description": "Implement API", "todos": ["Read api.py", "Read models.py", "Read views.py"]}
]"""


# ---------------------------------------------------------------------------
# PTI plan + executor outputs
# ---------------------------------------------------------------------------

PTI_PLAN_OUTPUT = """## Implementation Plan

1. Create user model with validation
2. Implement CRUD endpoints
3. Add authentication middleware
"""

PTI_EXECUTOR_OUTPUT = """Implementation complete:
- Added User model in models/user.py
- Added CRUD endpoints in api/users.py
- Added auth middleware in middleware/auth.py
"""

PTI_ANALYSIS_CONTINUE = """## Analysis
should_continue: true
next_iteration_request: Add error handling and input validation
"""

PTI_ANALYSIS_DONE = """## Analysis
should_continue: false
The implementation meets all quality requirements.
"""


# ---------------------------------------------------------------------------
# ConversationalInferencer tool-call responses
# ---------------------------------------------------------------------------

CONV_TOOL_CALL_XML = """I'll start by setting the target path.
<tool_call>
<tool_name>set_target_path</tool_name>
<parameters>{"path": "/workspace/src"}</parameters>
</tool_call>"""

CONV_NO_TOOLS = "Done — no further action required."


# ---------------------------------------------------------------------------
# Aggregator responses
# ---------------------------------------------------------------------------

BTA_AGGREGATOR_OUTPUT = """## Synthesis
Combining all worker outputs into a unified design:

- Authentication: JWT with refresh tokens
- Database: PostgreSQL with normalized schema
- API: REST with OpenAPI spec
"""
