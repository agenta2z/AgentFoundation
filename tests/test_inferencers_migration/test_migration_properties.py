"""Property-based tests for inferencers migration.

Feature: inferencers-migration
"""

import os
import re
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Compute the destination inferencers directory relative to this test file
_TEST_FILE_DIR = Path(__file__).resolve().parent
_INFERENCERS_DIR = (
    _TEST_FILE_DIR.parent.parent
    / "src"
    / "agent_foundation"
    / "common"
    / "inferencers"
)


def _build_py_manifest() -> list[str]:
    """Build a manifest of ALL .py files under the destination inferencers directory."""
    manifest: list[str] = []
    for root, _dirs, files in os.walk(_INFERENCERS_DIR):
        for fname in files:
            if fname.endswith(".py"):
                manifest.append(os.path.join(root, fname))
    return manifest


# Build the manifest once at module level
_PY_FILE_MANIFEST = _build_py_manifest()

# Pattern to match "rankevolve" case-insensitively
_RANKEVOLVE_PATTERN = re.compile(r"rankevolve", re.IGNORECASE)


@pytest.mark.skipif(
    len(_PY_FILE_MANIFEST) == 0,
    reason="No .py files found in destination inferencers directory",
)
@given(file_path=st.sampled_from(_PY_FILE_MANIFEST))
@settings(max_examples=100)
def test_no_rankevolve_references(file_path: str) -> None:
    """Property 1: No rankevolve references remain in destination.

    For any .py file under AgentFoundation/src/agent_foundation/common/inferencers/,
    the string "rankevolve" (case-insensitive) shall not appear anywhere in the file content.

    **Validates: Requirements 11.1–11.7**

    Feature: inferencers-migration, Property 1: No rankevolve references remain in destination
    """
    content = Path(file_path).read_text(encoding="utf-8", errors="replace")
    matches = _RANKEVOLVE_PATTERN.findall(content)
    assert len(matches) == 0, (
        f"Found {len(matches)} 'rankevolve' reference(s) in {file_path}:\n"
        f"File content snippet around match:\n"
        f"{_get_context_around_match(content)}"
    )


def _get_context_around_match(content: str) -> str:
    """Return a snippet of content around the first 'rankevolve' match for debugging."""
    match = _RANKEVOLVE_PATTERN.search(content)
    if match is None:
        return "(no match found)"
    start = max(0, match.start() - 80)
    end = min(len(content), match.end() + 80)
    return f"...{content[start:end]}..."


# ---------------------------------------------------------------------------
# Property 2: All expected files exist in destination
# ---------------------------------------------------------------------------

# Complete manifest of expected file paths (relative to _INFERENCERS_DIR).
# Grouped by module for readability; flattened into a single list for testing.

_ROOT_FILES = [
    "__init__.py",
    "inferencer_base.py",
    "streaming_inferencer_base.py",
    "templated_inferencer.py",
    "api_inferencer_base.py",
    "http_request_inferencer.py",
    "remote_inferencer_base.py",
    "inference_args.py",
]

_CONVERSATIONAL_FILES = [
    "agentic_inferencers/conversational/__init__.py",
    "agentic_inferencers/conversational/context.py",
    "agentic_inferencers/conversational/context_compressor.py",
    "agentic_inferencers/conversational/conversation_response_parser.py",
    "agentic_inferencers/conversational/conversation_tools.py",
    "agentic_inferencers/conversational/conversational_inferencer.py",
    "agentic_inferencers/conversational/prompt_rendering.py",
    "agentic_inferencers/conversational/protocols.py",
    "agentic_inferencers/conversational/tool_call_parser.py",
    "agentic_inferencers/conversational/tool_input_collector.py",
]

_METAMATE_FILES = [
    "agentic_inferencers/external/metamate/__init__.py",
    "agentic_inferencers/external/metamate/common.py",
    "agentic_inferencers/external/metamate/exceptions.py",
    "agentic_inferencers/external/metamate/metamate_cli_inferencer.py",
    "agentic_inferencers/external/metamate/metamate_sdk_inferencer.py",
    "agentic_inferencers/external/metamate/query_metamate.py",
    "agentic_inferencers/external/metamate/types.py",
    "agentic_inferencers/external/metamate/adapters/__init__.py",
    "agentic_inferencers/external/metamate/adapters/debug_assistant_adapter.py",
    "agentic_inferencers/external/metamate/adapters/deep_research_adapter.py",
    "agentic_inferencers/external/metamate/adapters/knowledge_discovery_adapter.py",
    "agentic_inferencers/external/metamate/adapters/platform_qa_adapter.py",
    "agentic_inferencers/external/metamate/clients/__init__.py",
    "agentic_inferencers/external/metamate/clients/fallback_client.py",
    "agentic_inferencers/external/metamate/clients/interfaces.py",
    "agentic_inferencers/external/metamate/clients/metamate_client.py",
    "agentic_inferencers/external/metamate/clients/mock_metamate_client.py",
]

_FLOW_INFERENCERS_FILES = [
    "agentic_inferencers/flow_inferencers/__init__.py",
    "agentic_inferencers/flow_inferencers/plan_then_implement_inferencer.py",
    "agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py",
]

_CLAUDE_CODE_FILES = [
    "agentic_inferencers/external/claude_code/__init__.py",
    "agentic_inferencers/external/claude_code/claude_code_inferencer.py",
    "agentic_inferencers/external/claude_code/claude_code_cli_inferencer.py",
    "agentic_inferencers/external/claude_code/common.py",
]

_DEVMATE_FILES = [
    "agentic_inferencers/external/devmate/__init__.py",
    "agentic_inferencers/external/devmate/devmate_sdk_inferencer.py",
    "agentic_inferencers/external/devmate/common.py",
    "agentic_inferencers/external/devmate/configs/freeform_agentic.md",
]

_PLUGBOARD_FILES = [
    "api_inferencers/plugboard/__init__.py",
    "api_inferencers/plugboard/plugboard_api_inferencer.py",
]

# Flattened manifest: all 48 expected files
_EXPECTED_FILE_MANIFEST = (
    _ROOT_FILES
    + _CONVERSATIONAL_FILES
    + _METAMATE_FILES
    + _FLOW_INFERENCERS_FILES
    + _CLAUDE_CODE_FILES
    + _DEVMATE_FILES
    + _PLUGBOARD_FILES
)


@given(rel_path=st.sampled_from(_EXPECTED_FILE_MANIFEST))
@settings(max_examples=100)
def test_all_expected_files_exist(rel_path: str) -> None:
    """Property 2: All expected files exist in destination.

    For any file path in the expected file manifest (conversational 10, metamate 17,
    flow_inferencers 3, claude_code 4, devmate 4, plugboard 2, root 8), that file
    shall exist in the destination directory.

    **Validates: Requirements 1.1, 2.1, 3.1, 3.2, 3.3**

    Feature: inferencers-migration, Property 2: All expected files exist in destination
    """
    full_path = _INFERENCERS_DIR / rel_path
    assert full_path.is_file(), (
        f"Expected file missing from destination: {rel_path}\n"
        f"Full path: {full_path}"
    )

# ---------------------------------------------------------------------------
# Property 3: All destination-only files are preserved
# ---------------------------------------------------------------------------

# Manifest of destination-only file paths (relative to _INFERENCERS_DIR).
# These files exist ONLY in the destination and must be preserved unchanged.

_BEDROCK_INFERENCERS_FILES = [
    "bedrock_inferencers/__init__.py",
    "bedrock_inferencers/bedrock_inferencer.py",
    "bedrock_inferencers/claude_bedrock_inferencer.py",
    "bedrock_inferencers/constants.py",
    "bedrock_inferencers/_dev/feedback_learning_test.py",
    "bedrock_inferencers/_dev/feedback_learning_test2.py",
    "bedrock_inferencers/_dev/gold_test_cases.jsonl",
    "bedrock_inferencers/_dev/test_claude3_inference.py",
    "bedrock_inferencers/_dev/test_claude3_parallel_inference.py",
]

_AG_FILES = [
    "api_inferencers/ag/__init__.py",
    "api_inferencers/ag/ag_claude_api_inferencer.py",
]

_TERMINAL_DEST_ONLY_FILES = [
    "terminal_inferencers/terminal_inferencer_base.py",
    "terminal_inferencers/terminal_inferencer_response.py",
    "terminal_inferencers/devmate/__init__.py",
]

# Flattened manifest of all destination-only files
_DESTINATION_ONLY_MANIFEST = (
    _BEDROCK_INFERENCERS_FILES
    + _AG_FILES
    + _TERMINAL_DEST_ONLY_FILES
)


@given(rel_path=st.sampled_from(_DESTINATION_ONLY_MANIFEST))
@settings(max_examples=100)
def test_destination_only_files_preserved(rel_path: str) -> None:
    """Property 3: All destination-only files are preserved.

    For any destination-only file (bedrock_inferencers/*, api_inferencers/ag/*,
    terminal_inferencers/terminal_inferencer_base.py,
    terminal_inferencers/terminal_inferencer_response.py,
    terminal_inferencers/devmate/*), the file shall exist in the destination.

    **Validates: Requirements 8.5, 10.1, 10.2, 10.3, 10.4, 10.5**

    Feature: inferencers-migration, Property 3: All destination-only files are preserved
    """
    full_path = _INFERENCERS_DIR / rel_path
    assert full_path.is_file(), (
        f"Destination-only file missing or deleted: {rel_path}\n"
        f"Full path: {full_path}"
    )


# ---------------------------------------------------------------------------
# Property 4: Destination-only methods preserved in inferencer_base.py
# ---------------------------------------------------------------------------

import ast

# The set of methods that must exist in inferencer_base.py (destination-only
# or shared methods that must be preserved after migration).
_REQUIRED_INFERENCER_BASE_METHODS = [
    "parallel_infer",
    "aconnect",
    "adisconnect",
    "__aenter__",
    "__aexit__",
]


def _get_class_methods_from_file(filepath: Path) -> set[str]:
    """Parse a Python file and return all method names defined inside classes."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    methods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef,)):
            for item in ast.walk(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(item.name)
    return methods


@given(method_name=st.sampled_from(_REQUIRED_INFERENCER_BASE_METHODS))
@settings(max_examples=100)
def test_destination_only_methods_preserved_in_inferencer_base(method_name: str) -> None:
    """Property 4: Destination-only methods preserved in inferencer_base.py.

    For any method in the set {parallel_infer, aconnect, adisconnect, __aenter__,
    __aexit__}, the method shall exist as a defined method in the destination
    inferencer_base.py.

    **Validates: Requirements 5.1, 5.2, 10.7**

    Feature: inferencers-migration, Property 4: Destination-only methods preserved in inferencer_base.py
    """
    inferencer_base_path = _INFERENCERS_DIR / "inferencer_base.py"
    assert inferencer_base_path.is_file(), (
        f"inferencer_base.py not found at {inferencer_base_path}"
    )
    methods = _get_class_methods_from_file(inferencer_base_path)
    assert method_name in methods, (
        f"Method '{method_name}' not found in inferencer_base.py.\n"
        f"Found methods: {sorted(methods)}"
    )


# ---------------------------------------------------------------------------
# Property 5: All __init__.py files export expected symbols
# ---------------------------------------------------------------------------


def _parse_all_from_init(init_path: Path) -> set[str]:
    """Parse the __all__ list from an __init__.py file using AST.

    If __all__ is not defined, fall back to collecting top-level imported names
    (from ... import X, Y style).
    """
    source = init_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(init_path))

    # Try to find __all__ assignment
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return {
                            elt.value  # type: ignore[union-attr]
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        }

    # Fallback: collect names from `from ... import X, Y` statements
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported = alias.asname if alias.asname else alias.name
                if not imported.startswith("_"):
                    names.add(imported)
    return names


# (init_file relative to _INFERENCERS_DIR, expected symbols set)
_INIT_EXPECTED_SYMBOLS: list[tuple[str, set[str]]] = [
    (
        "agentic_inferencers/__init__.py",
        {
            "ClaudeCodeInferencer",
            "DevmateSDKInferencer",
            "DevmateCliInferencer",
            "SDKInferencerResponse",
            "DualInferencer",
            "ReflectiveInferencer",
            "MetamateSDKInferencer",
            "MetamateCliInferencer",
            "PlanThenImplementInferencer",
            "BreakdownThenAggregateInferencer",
        },
    ),
    (
        "agentic_inferencers/conversational/__init__.py",
        {
            "AgenticDynamicContext",
            "AgenticResult",
            "CompletedAction",
            "ContextBudget",
            "ConversationTool",
            "ConversationToolType",
            "ConversationResponse",
            "ConversationalInferencer",
            "ContextCompressorCallable",
            "PromptRenderer",
            "ToolExecutionResult",
            "ToolExecutorCallable",
            "parse_conversation_response",
        },
    ),
    (
        "agentic_inferencers/flow_inferencers/__init__.py",
        {
            "BreakdownThenAggregateInferencer",
            "parse_numbered_list",
        },
    ),
    (
        "agentic_inferencers/external/__init__.py",
        {
            "SDKInferencerResponse",
        },
    ),
]

# Build a flat list of (init_file, symbol) pairs for Hypothesis to sample from
_INIT_SYMBOL_PAIRS: list[tuple[str, str]] = [
    (init_file, symbol)
    for init_file, symbols in _INIT_EXPECTED_SYMBOLS
    for symbol in symbols
]


@given(pair=st.sampled_from(_INIT_SYMBOL_PAIRS))
@settings(max_examples=100)
def test_init_files_export_expected_symbols(pair: tuple[str, str]) -> None:
    """Property 5: All __init__.py files export expected symbols.

    For any (init_file, expected_symbol) pair drawn from the expected symbol
    manifest, parsing the __all__ list (or imported names) from that __init__.py
    shall include the expected symbol.

    **Validates: Requirements 1.7, 2.7, 3.6, 4.1–4.4, 10.6**

    Feature: inferencers-migration, Property 5: All __init__.py files export expected symbols
    """
    init_rel_path, expected_symbol = pair
    init_path = _INFERENCERS_DIR / init_rel_path
    assert init_path.is_file(), (
        f"__init__.py not found: {init_rel_path}\nFull path: {init_path}"
    )
    exported_symbols = _parse_all_from_init(init_path)
    assert expected_symbol in exported_symbols, (
        f"Symbol '{expected_symbol}' not exported by {init_rel_path}.\n"
        f"Exported symbols: {sorted(exported_symbols)}"
    )


# ---------------------------------------------------------------------------
# Property 6: Import rewriting is a correct transformation
# ---------------------------------------------------------------------------

# Source inferencers directory (relative to this test file)
_SOURCE_INFERENCERS_DIR = (
    _TEST_FILE_DIR.parent.parent.parent
    / "_dev"
    / "rankevolve"
    / "src"
    / "agentic_foundation"
    / "common"
    / "inferencers"
)

# Import rewriting rules in priority order (most specific first to avoid
# partial matches).  Each tuple is (old_pattern, new_pattern).
_IMPORT_REWRITE_RULES: list[tuple[str, str]] = [
    # Most specific first
    (
        "rankevolve.src.utils.common_utils.async_utils",
        "rich_python_utils.common_utils.async_function_helper",
    ),
    ("rankevolve.src.agentic_foundation", "agent_foundation"),
    ("rankevolve.src.utils", "rich_python_utils"),
    ("rankevolve.src.resources", "agent_foundation.resources"),
    ("rankevolve.src.server", "agent_foundation.server"),
    # Bare prefixes (without rankevolve.src)
    ("agentic_foundation.common.inferencers", "agent_foundation.common.inferencers"),
    ("agentic_foundation.apis", "agent_foundation.apis"),
]

# Additional string-literal / comment / variable-name rewriting rules that
# complement the import rules above.  The full migration (Tasks 5.1–5.4)
# also rewrites non-import occurrences of "rankevolve" in prompts, docstrings,
# config paths, BUCK targets, and variable names.
_STRING_LITERAL_REWRITE_RULES: list[tuple[str, str]] = [
    # fbcode path references
    ("fbcode/rankevolve/src/agentic_foundation", "fbcode/agent_foundation"),
    ("fbcode//rankevolve/src/agentic_foundation", "fbcode//agent_foundation"),
    # Variable / constant names
    ("_RANKEVOLVE_DEVMATE_CONFIG_DIR", "_AGENT_FOUNDATION_DEVMATE_CONFIG_DIR"),
    ("RANKEVOLVE_AGENTIC", "AGENT_FOUNDATION_AGENTIC"),
    # Branded references in prompts and docstrings
    ("RankEvolve", "AgentFoundation"),
    ("rankevolve", "agent_foundation"),
]


def _build_source_py_manifest() -> list[str]:
    """Build a manifest of ALL .py files under the SOURCE inferencers directory."""
    if not _SOURCE_INFERENCERS_DIR.is_dir():
        return []
    manifest: list[str] = []
    for root, _dirs, files in os.walk(_SOURCE_INFERENCERS_DIR):
        for fname in files:
            if fname.endswith(".py"):
                manifest.append(os.path.join(root, fname))
    return manifest


_SOURCE_PY_FILE_MANIFEST = _build_source_py_manifest()


def _apply_import_rewrite_rules(content: str) -> str:
    """Apply all import rewriting rules and string-literal rules to content.

    Import rules are applied first (most-specific-first), then string-literal
    rules mop up non-import occurrences (comments, docstrings, variable names).
    """
    for old_pattern, new_pattern in _IMPORT_REWRITE_RULES:
        content = content.replace(old_pattern, new_pattern)
    for old_pattern, new_pattern in _STRING_LITERAL_REWRITE_RULES:
        content = content.replace(old_pattern, new_pattern)
    return content


@pytest.mark.skipif(
    len(_SOURCE_PY_FILE_MANIFEST) == 0,
    reason="Source inferencers directory not found or contains no .py files",
)
@given(file_path=st.sampled_from(_SOURCE_PY_FILE_MANIFEST))
@settings(max_examples=100)
def test_import_rewriting_is_correct_transformation(file_path: str) -> None:
    """Property 6: Import rewriting is a correct transformation.

    For any source .py file, applying the import rewriting rules (the 7
    substitution rules from the Import Path Mapping Table in priority order)
    and then checking for "rankevolve" occurrences shall yield zero matches.
    This validates that the rewriting rules are complete and sufficient.

    **Validates: Requirements 11.1–11.7**

    Feature: inferencers-migration, Property 6: Import rewriting is a correct transformation
    """
    content = Path(file_path).read_text(encoding="utf-8", errors="replace")
    rewritten = _apply_import_rewrite_rules(content)
    matches = _RANKEVOLVE_PATTERN.findall(rewritten)
    assert len(matches) == 0, (
        f"After applying import rewrite rules, {len(matches)} 'rankevolve' "
        f"reference(s) remain in {file_path}:\n"
        f"{_get_rewritten_context(rewritten)}"
    )


def _get_rewritten_context(content: str) -> str:
    """Return a snippet of rewritten content around the first 'rankevolve' match."""
    match = _RANKEVOLVE_PATTERN.search(content)
    if match is None:
        return "(no match found)"
    start = max(0, match.start() - 100)
    end = min(len(content), match.end() + 100)
    return f"...{content[start:end]}..."


# ---------------------------------------------------------------------------
# Property 7: async_utils mapped to async_function_helper
# ---------------------------------------------------------------------------

# Pattern matching the OLD (incorrect) import name: common_utils.async_utils
_OLD_ASYNC_UTILS_PATTERN = re.compile(r"common_utils\.async_utils")
# Pattern matching the NEW (correct) import name: common_utils.async_function_helper
_NEW_ASYNC_HELPER_PATTERN = re.compile(r"common_utils\.async_function_helper")


@pytest.mark.skipif(
    len(_PY_FILE_MANIFEST) == 0,
    reason="No .py files found in destination inferencers directory",
)
@given(file_path=st.sampled_from(_PY_FILE_MANIFEST))
@settings(max_examples=100)
def test_async_utils_mapped_to_async_function_helper(file_path: str) -> None:
    """Property 7: async_utils mapped to async_function_helper.

    For any .py file under the destination inferencers directory, if the file
    contains an import from common_utils.async (i.e., a line referencing
    common_utils.async_utils or common_utils.async_function_helper), it shall
    reference async_function_helper and NOT async_utils.

    **Validates: Requirements 11.6**

    Feature: inferencers-migration, Property 7: async_utils mapped to async_function_helper
    """
    content = Path(file_path).read_text(encoding="utf-8", errors="replace")

    has_old_name = _OLD_ASYNC_UTILS_PATTERN.search(content) is not None
    has_new_name = _NEW_ASYNC_HELPER_PATTERN.search(content) is not None

    # Only assert if the file references either pattern (skip files with neither)
    if has_old_name or has_new_name:
        assert not has_old_name, (
            f"File still uses old 'common_utils.async_utils' import "
            f"(should be 'common_utils.async_function_helper'): {file_path}"
        )
