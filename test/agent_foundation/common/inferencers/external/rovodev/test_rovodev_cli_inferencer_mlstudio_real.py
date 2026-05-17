"""Real integration test: RovoDevCliInferencer deep-dives ML Studio codebase.

Validates that RovoDevCliInferencer — using the built-in Atlassian MCP tools
(Bitbucket file browsing, Confluence search, bash/grep over local workspace) —
can autonomously produce a detailed, structured report on the ML Studio
monorepo without any manual cloning or code reading by the test author.

This test proves the capability demonstrated interactively in the RovoDev
session on 2026-04-27: generating a comprehensive report covering:
  - Team namespaces in modules/ (54+ teams)
  - Shared libraries in libraries/ (40+ packages)
  - Workflow YAML descriptors (500+ files)
  - Admin/orchestration engine
  - CI/CD pipeline architecture
  - AI integration highlights

Prerequisites:
    - ``acli`` installed and authenticated (``acli auth login``)
    - Network access to Atlassian Bitbucket (for MCP tools)
    - ML Studio repo cloned locally at MLSTUDIO_LOCAL_PATH (or accessible via
      Bitbucket MCP)

Run with::

    PYTHONPATH=src:../RichPythonUtils/src python3 -m pytest \\
        test/agent_foundation/common/inferencers/external/rovodev/test_rovodev_cli_inferencer_mlstudio_real.py \\
        -vs -m integration

    # To override the ML Studio working dir:
    MLSTUDIO_LOCAL_PATH=/your/path/to/ml-studio pytest ... -vs -m integration
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
    RovoDevCliInferencer,
)

# ---------------------------------------------------------------------------
# Pytest marks: skip entire module if acli is not installed / not on PATH
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not shutil.which("acli"),
        reason="acli not installed or not in PATH — run: brew install atlassian-cli",
    ),
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 900  # 15 minutes — deep-dive tasks can take time

# Local path to the ML Studio checkout. Falls back to the path used in the
# interactive RovoDev session (workspace atlassian_packages/ml-studio).
_DEFAULT_MLSTUDIO_PATH = os.path.expanduser(
    "~/MyProjects/atlassian_packages/ml-studio"
)
MLSTUDIO_LOCAL_PATH: str = os.environ.get(
    "MLSTUDIO_LOCAL_PATH", _DEFAULT_MLSTUDIO_PATH
)

# Bitbucket workspace + repo slug for MCP-based browsing
BITBUCKET_WORKSPACE = "atlassian"
BITBUCKET_REPO = "ml-studio"

# JSON output schema that structures the agent's report into typed fields.
# This forces the agent to produce machine-readable output that we can assert
# on precisely, rather than grepping free-form markdown.
REPORT_OUTPUT_SCHEMA = json.dumps({
    "type": "object",
    "required": [
        "title",
        "platform_summary",
        "module_team_namespaces",
        "library_team_namespaces",
        "workflow_areas",
        "platform_sdk_highlights",
        "cicd_pipeline_highlights",
        "ai_integration_highlights",
        "total_module_teams_count",
        "total_library_packages_count",
        "total_workflow_yaml_files_estimate",
    ],
    "properties": {
        "title": {
            "type": "string",
            "description": "Short title for the report",
        },
        "platform_summary": {
            "type": "string",
            "description": "2-3 sentence description of what ML Studio is",
        },
        "module_team_namespaces": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of team namespace names found under modules/src/",
        },
        "library_team_namespaces": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of team namespace names found under libraries/src/",
        },
        "workflow_areas": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of workflow areas found under workflows/src/",
        },
        "platform_sdk_highlights": {
            "type": "string",
            "description": "Key capabilities of the ml-studio-sdk library",
        },
        "cicd_pipeline_highlights": {
            "type": "string",
            "description": "Key pipeline steps and features in bitbucket-pipelines.yml",
        },
        "ai_integration_highlights": {
            "type": "string",
            "description": "AI/LLM integration highlights (AI Gateway, fine-tuning, agents, etc.)",
        },
        "total_module_teams_count": {
            "type": "integer",
            "description": "Total number of team namespaces in modules/src/",
        },
        "total_library_packages_count": {
            "type": "integer",
            "description": "Total number of individual library packages in libraries/src/",
        },
        "total_workflow_yaml_files_estimate": {
            "type": "integer",
            "description": "Estimated total number of workflow YAML files in workflows/src/",
        },
    },
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inferencer(working_dir: str, output_file: str | None = None) -> RovoDevCliInferencer:
    """Create a RovoDevCliInferencer pointed at the ML Studio working dir."""
    return RovoDevCliInferencer(
        target_path=working_dir,
        output_file=output_file,
        output_schema=REPORT_OUTPUT_SCHEMA,
        idle_timeout_seconds=DEFAULT_TIMEOUT,
        tool_use_idle_timeout_seconds=DEFAULT_TIMEOUT,
        yolo=True,          # skip tool confirmations for programmatic use
        enable_legacy=True,  # legacy mode: reliable --output-file capture
    )


def _parse_report(result) -> dict:
    """Extract and parse the JSON report from the inferencer result.

    Tries multiple sources in order of reliability:
      1. ``raw_output`` — the clean captured output file content (most reliable
         when output_file is used; async path also stores JSON here)
      2. ``output`` directly — works when output is already clean JSON
      3. Extract first ``{ ... }`` block from ``output`` — fallback for rich
         terminal output that embeds JSON amongst ANSI/tool traces
    """
    # 1. Try raw_output first — cleanest source (output_file content or async capture)
    raw = getattr(result, "raw_output", None) or ""
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try extracting JSON block from raw_output
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass

    # 2. Try output directly
    output = result.output or ""
    if output:
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass

        # 3. Extract first { ... } block from output (rich terminal stream fallback)
        # Use rfind from the last "}" to handle nested JSON correctly
        start = output.find("{")
        end = output.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(output[start:end])
            except json.JSONDecodeError:
                pass

    pytest.fail(
        f"Could not parse JSON from inferencer output.\n"
        f"raw_output ({len(raw)} chars): {raw[:500]}\n"
        f"output ({len(output)} chars): {output[:500]}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRovoDevMLStudioDeepDive:
    """Test that RovoDevCliInferencer can autonomously deep-dive ML Studio.

    Uses the local ML Studio checkout as ``working_dir`` so the agent can:
      1. Use bash/ls/grep tools to explore the local filesystem
      2. Use Atlassian MCP Bitbucket tools to browse remote file contents
      3. Use Confluence/Jira search tools for additional context

    The structured JSON ``output_schema`` forces a typed report we can assert
    on precisely.
    """

    @pytest.fixture
    def mlstudio_path(self) -> str:
        """Return the ML Studio local path, skip test if not available."""
        path = Path(MLSTUDIO_LOCAL_PATH)
        if not path.exists():
            pytest.skip(
                f"ML Studio local path not found: {MLSTUDIO_LOCAL_PATH}\n"
                f"Set env var MLSTUDIO_LOCAL_PATH to override."
            )
        return str(path)

    @pytest.fixture
    def output_file(self, tmp_path) -> str:
        """Temp file to capture clean agent output."""
        return str(tmp_path / "mlstudio_report.json")

    def test_mlstudio_report_structure(self, mlstudio_path, output_file):
        """Agent produces a valid structured JSON report for ML Studio.

        This is the primary integration test: verifies that RovoDevCliInferencer,
        given the ML Studio working directory, can autonomously explore the codebase
        and return a fully populated JSON report with correct structure.
        """
        inferencer = _make_inferencer(mlstudio_path, output_file=output_file)

        prompt = (
            "You are analyzing the ML Studio monorepo (atlassian/ml-studio on Bitbucket). "
            "Your job is to produce a comprehensive deep-dive report by exploring the codebase. "
            "\n\n"
            "Please explore the following areas using your available tools (bash, ls, grep, "
            "Bitbucket MCP file browsing, etc.):\n"
            "1. List all team namespaces under modules/src/ (there should be 50+)\n"
            "2. List all team namespaces under libraries/src/ (there should be 15+)\n"
            "3. List all workflow areas under workflows/src/\n"
            "4. Read the ml-studio-sdk library under libraries/src/ml_platform/ml-studio-sdk/ "
            "and summarize its key capabilities\n"
            "5. Read bitbucket-pipelines.yml and summarize the CI/CD pipeline architecture\n"
            "6. Identify AI/LLM integration highlights (AI Gateway, fine-tuning workflows, "
            "agentic patterns, MCP tools, etc.)\n"
            "7. Count total module team namespaces, total library packages, and estimate "
            "total workflow YAML files\n"
            "\n"
            "Return your findings as a structured JSON report matching the output schema."
        )

        result = inferencer(prompt)

        assert result.success, (
            f"Inferencer failed.\n"
            f"stderr: {result.stderr}\n"
            f"stdout (first 1000 chars): {str(result.raw_output)[:1000]}"
        )

        report = _parse_report(result)

        # --- Structural checks ---
        assert "title" in report, "Report must have a title"
        assert "platform_summary" in report, "Report must have platform_summary"
        assert len(report.get("platform_summary", "")) > 50, \
            "platform_summary should be a meaningful description (>50 chars)"

        # --- Module teams ---
        module_teams = report.get("module_team_namespaces", [])
        assert isinstance(module_teams, list), "module_team_namespaces must be a list"
        assert len(module_teams) >= 20, (
            f"Expected at least 20 module team namespaces, got {len(module_teams)}.\n"
            f"Teams found: {module_teams}"
        )
        # Spot-check a few well-known team namespaces
        module_teams_lower = [t.lower() for t in module_teams]
        for expected in ["search_relevance", "confluence_ai", "jira_ai", "loom_ai", "ml_platform"]:
            assert any(expected in t for t in module_teams_lower), (
                f"Expected to find '{expected}' in module team namespaces.\n"
                f"Teams found: {module_teams}"
            )

        # --- Library teams ---
        lib_teams = report.get("library_team_namespaces", [])
        assert isinstance(lib_teams, list), "library_team_namespaces must be a list"
        assert len(lib_teams) >= 8, (
            f"Expected at least 8 library team namespaces, got {len(lib_teams)}.\n"
            f"Library teams found: {lib_teams}"
        )
        lib_teams_lower = [t.lower() for t in lib_teams]
        for expected in ["ml_platform", "confluence_ai", "search_relevance"]:
            assert any(expected in t for t in lib_teams_lower), (
                f"Expected '{expected}' in library team namespaces.\n"
                f"Library teams: {lib_teams}"
            )

        # --- Workflow areas ---
        workflow_areas = report.get("workflow_areas", [])
        assert isinstance(workflow_areas, list), "workflow_areas must be a list"
        assert len(workflow_areas) >= 10, (
            f"Expected at least 10 workflow areas, got {len(workflow_areas)}.\n"
            f"Workflow areas: {workflow_areas}"
        )

        # --- SDK highlights ---
        sdk_highlights = report.get("platform_sdk_highlights", "")
        assert len(sdk_highlights) > 50, \
            "platform_sdk_highlights should be meaningful (>50 chars)"
        # The ml-studio-sdk provides MLflow, ASAP, secrets, data_classification, Tecton
        sdk_lower = sdk_highlights.lower()
        assert any(kw in sdk_lower for kw in ["mlflow", "asap", "secret", "classification", "tecton", "databricks"]), (
            f"SDK highlights should mention key capabilities.\nGot: {sdk_highlights}"
        )

        # --- CI/CD highlights ---
        cicd_highlights = report.get("cicd_pipeline_highlights", "")
        assert len(cicd_highlights) > 50, \
            "cicd_pipeline_highlights should be meaningful (>50 chars)"

        # --- AI integration ---
        ai_highlights = report.get("ai_integration_highlights", "")
        assert len(ai_highlights) > 50, \
            "ai_integration_highlights should be meaningful (>50 chars)"

        # --- Counts ---
        total_module_teams = report.get("total_module_teams_count", 0)
        assert total_module_teams >= 20, (
            f"Expected >=20 module teams, agent reported {total_module_teams}"
        )

        total_lib_packages = report.get("total_library_packages_count", 0)
        assert total_lib_packages >= 10, (
            f"Expected >=10 library packages, agent reported {total_lib_packages}"
        )

        total_workflow_yamls = report.get("total_workflow_yaml_files_estimate", 0)
        assert total_workflow_yamls >= 50, (
            f"Expected >=50 workflow YAML files, agent reported {total_workflow_yamls}"
        )

        # --- Log the full report for human inspection ---
        print("\n" + "=" * 70)
        print("ML Studio Deep-Dive Report (RovoDevCliInferencer)")
        print("=" * 70)
        print(json.dumps(report, indent=2))
        print("=" * 70)

    def test_mlstudio_module_teams_completeness(self, mlstudio_path, output_file):
        """Agent can enumerate all 50+ module team namespaces specifically.

        A focused test that asks only about modules/src structure to verify
        the agent can enumerate a large flat directory reliably.
        """
        inferencer = _make_inferencer(mlstudio_path, output_file=output_file)

        prompt = (
            "List every team namespace directory directly under modules/src/ in "
            "the ML Studio monorepo. Use bash or ls tools to enumerate them. "
            "Return a JSON array of all namespace names, with a count. "
            "There should be approximately 50+ namespaces."
        )

        # Use a simpler output schema for this focused test
        inferencer.output_schema = json.dumps({
            "type": "object",
            "required": ["namespaces", "count"],
            "properties": {
                "namespaces": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All team namespace names under modules/src/",
                },
                "count": {
                    "type": "integer",
                    "description": "Total count of namespaces",
                },
            },
        })

        result = inferencer(prompt)
        assert result.success, (
            f"Inferencer failed: {result.stderr}\nOutput: {result.output[:500]}"
        )

        report = _parse_report(result)
        namespaces = report.get("namespaces", [])
        count = report.get("count", 0)

        assert len(namespaces) >= 20, (
            f"Expected >=20 module namespaces enumerated, got {len(namespaces)}.\n"
            f"Namespaces: {namespaces}"
        )
        assert count >= 20, (
            f"Expected count >= 20, got {count}"
        )

        # Verify known namespaces are present
        ns_lower = [n.lower() for n in namespaces]
        for expected in [
            "search_relevance", "confluence_ai", "jira_ai",
            "loom_ai", "knowledge_graph", "ml_platform", "core_ml",
            "canary_analysis", "csm_ai", "devai_autoreview",
        ]:
            assert any(expected in n for n in ns_lower), (
                f"Expected namespace '{expected}' not found.\n"
                f"Namespaces found: {namespaces}"
            )

        print(f"\n✅ Enumerated {len(namespaces)} module team namespaces")
        print(f"   Count reported by agent: {count}")

    def test_mlstudio_library_packages_discovery(self, mlstudio_path, output_file):
        """Agent can discover and categorize all shared libraries.

        Verifies the agent can traverse the nested libraries/src/<team>/<lib>
        structure and identify all independently versioned Python packages.
        """
        inferencer = _make_inferencer(mlstudio_path, output_file=output_file)

        prompt = (
            "Explore libraries/src/ in the ML Studio monorepo. "
            "For each team namespace under libraries/src/, list the individual "
            "library packages (each has its own pyproject.toml). "
            "Return a JSON object mapping each team namespace to its list of library package names."
        )

        inferencer.output_schema = json.dumps({
            "type": "object",
            "required": ["libraries_by_team", "total_packages"],
            "properties": {
                "libraries_by_team": {
                    "type": "object",
                    "description": "Map of team_namespace -> [library_package_names]",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "total_packages": {
                    "type": "integer",
                    "description": "Total number of library packages discovered",
                },
            },
        })

        result = inferencer(prompt)
        assert result.success, (
            f"Inferencer failed: {result.stderr}\nOutput: {result.output[:500]}"
        )

        report = _parse_report(result)
        libs_by_team = report.get("libraries_by_team", {})
        total_packages = report.get("total_packages", 0)

        assert len(libs_by_team) >= 5, (
            f"Expected >=5 team namespaces in libraries, got {len(libs_by_team)}.\n"
            f"Teams found: {list(libs_by_team.keys())}"
        )
        assert total_packages >= 10, (
            f"Expected >=10 total library packages, got {total_packages}"
        )

        # ml_platform should have multiple SDKs
        ml_platform_libs = libs_by_team.get("ml_platform", [])
        assert len(ml_platform_libs) >= 2, (
            f"ml_platform should have multiple SDKs, got: {ml_platform_libs}"
        )

        # The ml-studio-sdk should appear somewhere under ml_platform
        ml_platform_lower = [lib.lower() for lib in ml_platform_libs]
        assert any("studio" in lib or "sdk" in lib for lib in ml_platform_lower), (
            f"ml-studio-sdk not found under ml_platform.\n"
            f"ml_platform libs: {ml_platform_libs}"
        )

        print(f"\n✅ Discovered {total_packages} library packages across {len(libs_by_team)} teams")
        for team, libs in sorted(libs_by_team.items()):
            print(f"   {team}: {libs}")

    @pytest.mark.asyncio
    async def test_mlstudio_async_report(self, mlstudio_path, output_file):
        """Async version: same deep-dive but via ainfer().

        Verifies the async path works correctly for long-running deep-dive tasks.
        """
        inferencer = _make_inferencer(mlstudio_path, output_file=output_file)

        prompt = (
            "Provide a brief overview of ML Studio: what is it, what are the "
            "main top-level directories and their purposes, and name at least "
            "5 team namespaces from modules/src/. "
            "Return structured JSON with fields: summary, top_level_dirs, sample_module_teams."
        )

        inferencer.output_schema = json.dumps({
            "type": "object",
            "required": ["summary", "top_level_dirs", "sample_module_teams"],
            "properties": {
                "summary": {"type": "string"},
                "top_level_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "sample_module_teams": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 5,
                },
            },
        })

        result = await inferencer.ainfer(prompt)

        assert result.success, (
            f"Async inferencer failed: {result.stderr}\nOutput: {result.output[:500]}"
        )

        report = _parse_report(result)

        assert len(report.get("summary", "")) > 30, \
            "Async report summary should be meaningful"
        assert len(report.get("top_level_dirs", [])) >= 3, \
            "Should identify at least 3 top-level dirs (modules, libraries, workflows)"
        assert len(report.get("sample_module_teams", [])) >= 5, \
            "Should name at least 5 module team namespaces"

        print(f"\n✅ Async report: {report.get('summary', '')[:200]}")
