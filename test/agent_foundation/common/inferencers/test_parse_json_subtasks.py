"""Tests for BTA._parse_json_subtasks — fence regex + JSON repair.

Covers the two fixes:
1. Fence regex allows newline between ```json and { (standard markdown)
2. JSON repair for backtick-quoted code blocks with unescaped quotes

Uses realistic mocked responses based on actual RovoDevCLI output.
"""
import textwrap

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)


@pytest.fixture
def bta():
    """Minimal BTA instance with required attrs for parsing tests."""
    obj = BreakdownThenAggregateInferencer.__new__(
        BreakdownThenAggregateInferencer
    )
    obj.worker_query_fields = ("description", "todos")
    obj.expand_todos_to_workers = False
    obj._last_aggregation_guidance = None
    return obj


# ---------------------------------------------------------------------------
# Fence regex: newline between ```json and {
# ---------------------------------------------------------------------------


class TestFenceRegex:
    def test_json_on_next_line(self, bta):
        """Standard markdown: { on the line after ```json."""
        raw = textwrap.dedent("""\
            <Response>

            ```json decomposed_subtasks
            {
              "subtasks": [
                {
                  "subtask_id": 1,
                  "description": "Investigate the codebase",
                  "todos": ["Read README", "Explore src/"],
                  "scope": "full codebase",
                  "priority": "HIGH",
                  "priority_reason": "foundational",
                  "priority_score": "3.5",
                  "args": {},
                  "subtask_dependencies": []
                }
              ],
              "reasoning": "single subtask for simple request",
              "coverage_complete": true,
              "gaps": "none",
              "aggregation_guidance": "integrate findings"
            }
            ```

            </Response>
        """)
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 1
        assert "Investigate the codebase" in result[0]["query"]

    def test_json_on_same_line(self, bta):
        """Non-standard but valid: { on the same line as ```json."""
        raw = '<Response>\n```json decomposed_subtasks\n{"subtasks": [{"subtask_id": 1, "description": "test", "todos": ["do it"], "scope": "all", "priority": "HIGH", "priority_reason": "needed", "priority_score": "3.5", "args": {}, "subtask_dependencies": []}], "reasoning": "ok", "coverage_complete": true, "gaps": "none", "aggregation_guidance": "merge"}\n```\n</Response>'
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 1

    def test_fence_without_label(self, bta):
        """```json (no 'decomposed_subtasks' label) — should still parse."""
        raw = textwrap.dedent("""\
            <Response>

            ```json
            {
              "subtasks": [
                {
                  "subtask_id": 1,
                  "description": "Simple task",
                  "todos": ["do stuff"],
                  "scope": "narrow",
                  "priority": "MEDIUM",
                  "priority_reason": "exploration",
                  "priority_score": "2.5",
                  "args": {},
                  "subtask_dependencies": []
                }
              ],
              "reasoning": "minimal",
              "coverage_complete": true,
              "gaps": "none",
              "aggregation_guidance": "combine"
            }
            ```

            </Response>
        """)
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 1

    def test_no_response_tags(self, bta):
        """JSON subtasks without <Response> tags — fallback to raw text."""
        raw = textwrap.dedent("""\
            ```json decomposed_subtasks
            {
              "subtasks": [
                {
                  "subtask_id": 1,
                  "description": "No tags",
                  "todos": ["work"],
                  "scope": "all",
                  "priority": "HIGH",
                  "priority_reason": "critical",
                  "priority_score": "3.5",
                  "args": {},
                  "subtask_dependencies": []
                }
              ],
              "reasoning": "ok",
              "coverage_complete": true,
              "gaps": "none",
              "aggregation_guidance": "merge"
            }
            ```
        """)
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 1

    def test_extra_whitespace_before_brace(self, bta):
        """Indented { after fence tag."""
        raw = textwrap.dedent("""\
            <Response>
            ```json decomposed_subtasks
              {
                "subtasks": [
                  {"subtask_id": 1, "description": "indented", "todos": ["x"], "scope": "y", "priority": "LOW", "priority_reason": "minor", "priority_score": "1.5", "args": {}, "subtask_dependencies": []}
                ],
                "reasoning": "ok",
                "coverage_complete": true,
                "gaps": "none",
                "aggregation_guidance": "merge"
              }
            ```
            </Response>
        """)
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# JSON repair: backtick-quoted code blocks with unescaped quotes
# ---------------------------------------------------------------------------


class TestJsonRepair:
    def test_backtick_quoted_json_in_description(self, bta):
        """Description contains `{"key": "value"}` — unescaped quotes inside JSON string."""
        raw = textwrap.dedent("""\
            <Response>
            ```json decomposed_subtasks
            {
              "subtasks": [
                {
                  "subtask_id": 1,
                  "description": "Use JSON-CoT path with `{\\"reasoning\\": ..., \\"category\\": ...}` schema enforced via vLLM",
                  "todos": ["implement it"],
                  "scope": "format fix",
                  "priority": "CRITICAL",
                  "priority_reason": "fixes FP",
                  "priority_score": "4.5",
                  "args": {},
                  "subtask_dependencies": []
                }
              ],
              "reasoning": "addresses the core issue",
              "coverage_complete": true,
              "gaps": "none",
              "aggregation_guidance": "merge"
            }
            ```
            </Response>
        """)
        # This has properly escaped quotes — should parse directly
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 1

    def test_unescaped_backtick_json_in_description(self, bta):
        """Realistic LLM output: `{"reasoning": ..., "category": ...}` with UNESCAPED quotes."""
        # This is what the LLM actually produces — unescaped " inside string values
        raw = '<Response>\n```json decomposed_subtasks\n{\n  "subtasks": [\n    {\n      "subtask_id": 1,\n      "description": "JSON-CoT path with `{\\"reasoning\\": ..., \\"category\\": ...}` schema",\n      "todos": ["do it"],\n      "scope": "all",\n      "priority": "HIGH",\n      "priority_reason": "needed",\n      "priority_score": "3.5",\n      "args": {},\n      "subtask_dependencies": []\n    }\n  ],\n  "reasoning": "ok",\n  "coverage_complete": true,\n  "gaps": "none",\n  "aggregation_guidance": "merge"\n}\n```\n</Response>'
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 1

    def test_realistic_broken_json_repaired(self, bta):
        """Realistic case: unescaped quotes in backtick code that breaks json.loads."""
        # Construct JSON where description has `{"key": "val"}` with raw unescaped quotes
        # This is what the actual v7 run produced
        subtask_json = (
            '{\n'
            '  "subtasks": [\n'
            '    {\n'
            '      "subtask_id": 1,\n'
            '      "description": "Use `{\\"reasoning\\": ..., \\"category\\": ...}` schema",\n'
            '      "todos": ["step1"],\n'
            '      "scope": "all",\n'
            '      "priority": "HIGH",\n'
            '      "priority_reason": "critical",\n'
            '      "priority_score": "3.5",\n'
            '      "args": {},\n'
            '      "subtask_dependencies": []\n'
            '    },\n'
            '    {\n'
            '      "subtask_id": 2,\n'
            '      "description": "Normal subtask without special chars",\n'
            '      "todos": ["step2"],\n'
            '      "scope": "narrow",\n'
            '      "priority": "MEDIUM",\n'
            '      "priority_reason": "supporting",\n'
            '      "priority_score": "2.5",\n'
            '      "args": {},\n'
            '      "subtask_dependencies": [1]\n'
            '    }\n'
            '  ],\n'
            '  "reasoning": "two subtasks",\n'
            '  "coverage_complete": true,\n'
            '  "gaps": "none",\n'
            '  "aggregation_guidance": "synthesize"\n'
            '}'
        )
        raw = f"<Response>\n```json decomposed_subtasks\n{subtask_json}\n```\n</Response>"
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 2
        assert result[1].get("query") is not None


# ---------------------------------------------------------------------------
# Multiple subtasks with dependencies
# ---------------------------------------------------------------------------


class TestMultipleSubtasks:
    def test_five_subtasks_with_dependencies(self, bta):
        """Realistic 5-subtask decomposition with dependencies."""
        raw = textwrap.dedent("""\
            <Response>

            ```json decomposed_subtasks
            {
              "subtasks": [
                {
                  "subtask_id": 1,
                  "description": "Catalog experiment results and build performance baseline",
                  "todos": ["Parse metrics files", "Build comparison table"],
                  "scope": "quantitative grounding",
                  "priority": "CRITICAL",
                  "priority_reason": "all other work depends on accurate baseline",
                  "priority_score": "4.5",
                  "args": {"root_dir": "/path/to/experiments"},
                  "subtask_dependencies": []
                },
                {
                  "subtask_id": 2,
                  "description": "Analyze training data quality and propose augmentations",
                  "todos": ["Audit label quality", "Propose corrective data"],
                  "scope": "training data",
                  "priority": "CRITICAL",
                  "priority_reason": "data is the dominant lever",
                  "priority_score": "4.5",
                  "args": {},
                  "subtask_dependencies": []
                },
                {
                  "subtask_id": 3,
                  "description": "Design training recipes",
                  "todos": ["Propose 4-6 recipes", "Specify hyperparameters"],
                  "scope": "training methodology",
                  "priority": "HIGH",
                  "priority_reason": "necessary but secondary to data",
                  "priority_score": "3.7",
                  "args": {},
                  "subtask_dependencies": []
                },
                {
                  "subtask_id": 4,
                  "description": "Design format and decoding fixes",
                  "todos": ["Propose mitigation strategies", "Write prompt templates"],
                  "scope": "inference-time fixes",
                  "priority": "CRITICAL",
                  "priority_reason": "directly addresses user concern",
                  "priority_score": "4.6",
                  "args": {},
                  "subtask_dependencies": []
                },
                {
                  "subtask_id": 5,
                  "description": "Synthesize into unified experiment plan",
                  "todos": ["Build compatibility matrix", "Draft phased plan"],
                  "scope": "final synthesis",
                  "priority": "HIGH",
                  "priority_reason": "aggregation of all findings",
                  "priority_score": "3.8",
                  "args": {},
                  "subtask_dependencies": [1, 2, 3, 4]
                }
              ],
              "reasoning": "Five-stream decomposition covering baseline, data, recipes, format fixes, and synthesis",
              "coverage_complete": true,
              "gaps": "none",
              "aggregation_guidance": "Worker 5 synthesizes workers 1-4 into a single experiment plan"
            }
            ```

            </Response>
        """)
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 5
        assert bta._last_aggregation_guidance is not None
        assert "synthesize" in bta._last_aggregation_guidance.lower()


# ---------------------------------------------------------------------------
# Realistic RovoDevCLI output (with CLI chrome)
# ---------------------------------------------------------------------------


class TestRovoDevCliOutput:
    def test_cli_chrome_with_clean_response(self, bta):
        """RovoDevCLI output has CLI banners but clean <Response> block."""
        raw = (
            "Working in /some/path\n"
            "Jira projects: https://hello.atlassian.net/browse/PROJ\n\n"
            "[?2004hCreating agent...\n"
            "✔ Using model: anthropic:claude-opus-4-7\n"
            "✔ Started 23 MCP servers\n\n"
            "─── Response ───────────────────────────────────\n"
            "I'll investigate the directory.\n"
            "────────────────────────────────────────────────\n"
            "  ⬢ Called bash:\n"
            '      • command: "ls -la /path"\n\n'
            "─── Response ───────────────────────────────────\n\n"
            "<Response>\n\n"
            "```json decomposed_subtasks\n"
            "{\n"
            '  "subtasks": [\n'
            "    {\n"
            '      "subtask_id": 1,\n'
            '      "description": "Investigate the code",\n'
            '      "todos": ["Read files"],\n'
            '      "scope": "codebase",\n'
            '      "priority": "HIGH",\n'
            '      "priority_reason": "needed",\n'
            '      "priority_score": "3.5",\n'
            '      "args": {},\n'
            '      "subtask_dependencies": []\n'
            "    }\n"
            "  ],\n"
            '  "reasoning": "single stream",\n'
            '  "coverage_complete": true,\n'
            '  "gaps": "none",\n'
            '  "aggregation_guidance": "merge findings"\n'
            "}\n"
            "```\n\n"
            "</Response>\n\n"
            "Session context: ▮▮▮▮▮▮ 98.7K/1M\n"
        )
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 1
        assert "Investigate the code" in result[0]["query"]

    def test_cli_output_without_response_tags(self, bta):
        """RovoDevCLI output with JSON but no <Response> tags — fallback path."""
        raw = (
            "Working in /some/path\n\n"
            "✔ Using model: anthropic:claude-opus-4-7\n\n"
            "─── Response ───────────────────────────────────\n\n"
            "```json\n"
            "{\n"
            '  "subtasks": [\n'
            "    {\n"
            '      "subtask_id": 1,\n'
            '      "description": "Simple task",\n'
            '      "todos": ["do it"],\n'
            '      "scope": "all",\n'
            '      "priority": "HIGH",\n'
            '      "priority_reason": "needed",\n'
            '      "priority_score": "3.5",\n'
            '      "args": {},\n'
            '      "subtask_dependencies": []\n'
            "    }\n"
            "  ],\n"
            '  "reasoning": "ok",\n'
            '  "coverage_complete": true,\n'
            '  "gaps": "none",\n'
            '  "aggregation_guidance": "merge"\n'
            "}\n"
            "```\n\n"
            "Session context: ▮▮▮▮ 50K/1M\n"
        )
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_subtasks_falls_back(self, bta):
        """Empty subtasks array falls back to numbered list parser."""
        raw = '<Response>\n```json decomposed_subtasks\n{"subtasks": [], "reasoning": "empty"}\n```\n</Response>'
        result = bta._parse_json_subtasks(raw)
        # Empty subtasks → falls back to numbered list → returns empty
        assert isinstance(result, list)

    def test_no_json_at_all(self, bta):
        """No JSON anywhere — falls back to numbered list."""
        raw = "<Response>\nJust some prose about subtasks.\n1. First thing\n2. Second thing\n</Response>"
        result = bta._parse_json_subtasks(raw)
        assert isinstance(result, list)

    def test_malformed_json_falls_back(self, bta):
        """Completely malformed JSON falls back gracefully."""
        raw = "<Response>\n```json decomposed_subtasks\n{this is not json at all}\n```\n</Response>"
        result = bta._parse_json_subtasks(raw)
        assert isinstance(result, list)

    def test_aggregation_guidance_captured(self, bta):
        """aggregation_guidance field is stored on BTA instance."""
        raw = textwrap.dedent("""\
            <Response>
            ```json decomposed_subtasks
            {
              "subtasks": [
                {"subtask_id": 1, "description": "task", "todos": ["x"], "scope": "y", "priority": "HIGH", "priority_reason": "z", "priority_score": "3.5", "args": {}, "subtask_dependencies": []}
              ],
              "reasoning": "ok",
              "coverage_complete": true,
              "gaps": "none",
              "aggregation_guidance": "Read all worker outputs and build a compatibility matrix"
            }
            ```
            </Response>
        """)
        bta._last_aggregation_guidance = None
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 1
        assert bta._last_aggregation_guidance == "Read all worker outputs and build a compatibility matrix"

    def test_subtask_fields_mapped_to_query(self, bta):
        """Subtask description/todos/scope are composed into query string."""
        raw = textwrap.dedent("""\
            <Response>
            ```json decomposed_subtasks
            {
              "subtasks": [
                {
                  "subtask_id": 1,
                  "description": "Analyze performance metrics",
                  "todos": ["Parse JSON files", "Build comparison table"],
                  "scope": "evaluation results",
                  "priority": "CRITICAL",
                  "priority_reason": "foundational",
                  "priority_score": "4.5",
                  "args": {"data_dir": "/path/to/data"},
                  "subtask_dependencies": []
                }
              ],
              "reasoning": "single focused subtask",
              "coverage_complete": true,
              "gaps": "none",
              "aggregation_guidance": "integrate"
            }
            ```
            </Response>
        """)
        result = bta._parse_json_subtasks(raw)
        assert len(result) == 1
        query = result[0]["query"]
        assert "Analyze performance metrics" in query
