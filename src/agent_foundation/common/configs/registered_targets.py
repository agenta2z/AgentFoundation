"""Central registry of all instantiable targets in AgentFoundation.

Import this module once at startup to populate the alias registry.
Uses ``register_alias()`` (string-only) to avoid importing every domain class
and its transitive dependencies at registration time.
"""

from rich_python_utils.config_utils import register_alias

_P = "agent_foundation"

# --- Inferencers ---
register_alias(
    "ClaudeAPI",
    f"{_P}.common.inferencers.api_inferencers.claude_api_inferencer.ClaudeApiInferencer",
    "inferencer",
)
register_alias(
    "AgClaudeAPI",
    f"{_P}.common.inferencers.api_inferencers.ag.ag_claude_api_inferencer.AgClaudeApiInferencer",
    "inferencer",
)
register_alias(
    "AgOpenAIAPI",
    f"{_P}.common.inferencers.api_inferencers.ag.ag_openai_api_inferencer.AgOpenAIApiInferencer",
    "inferencer",
)
register_alias(
    "AgGeminiAPI",
    f"{_P}.common.inferencers.api_inferencers.ag.ag_gemini_api_inferencer.AgGeminiApiInferencer",
    "inferencer",
)
register_alias(
    "RovoDevCLI",
    f"{_P}.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer.RovoDevCliInferencer",
    "inferencer",
)
register_alias(
    "ClaudeCodeCLI",
    f"{_P}.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer.ClaudeCodeCliInferencer",
    "inferencer",
)
register_alias(
    "ClaudeCodeSDK",
    f"{_P}.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer.ClaudeCodeSdkInferencer",
    "inferencer",
)
register_alias(
    "CodexCLI",
    f"{_P}.common.inferencers.agentic_inferencers.external.codex.codex_cli_inferencer.CodexCliInferencer",
    "inferencer",
)
register_alias(
    "CodexSDK",
    f"{_P}.common.inferencers.agentic_inferencers.external.codex.codex_sdk_inferencer.CodexSdkInferencer",
    "inferencer",
)
register_alias(
    "Conversational",
    f"{_P}.common.inferencers.agentic_inferencers.conversational"
    ".conversational_inferencer.ConversationalInferencer",
    "inferencer",
    alternatives=[
        f"{_P}.common.inferencers.agentic_inferencers.conversational"
        ".flow_node_adapter.ConversationalFlowNodeAdapter",
    ],
)
register_alias(
    "Dual",
    f"{_P}.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer.DualInferencer",
    "inferencer",
)
register_alias(
    "BTA",
    f"{_P}.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer.BreakdownThenAggregateInferencer",
    "inferencer",
)
register_alias(
    "MultiFlow",
    f"{_P}.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer.MultiFlowInferencer",
    "inferencer",
)
register_alias(
    "MultiFlowDual",
    f"{_P}.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer.MultiFlowDualInferencer",
    "inferencer",
)
register_alias(
    "PTI",
    f"{_P}.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer.PlanThenImplementInferencer",
    "inferencer",
)
register_alias(
    "RovoChat",
    f"{_P}.common.inferencers.agentic_inferencers.external.rovochat.rovochat_inferencer.RovoChatInferencer",
    "inferencer",
)
register_alias(
    "Metamate",
    f"{_P}.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer.MetamateSDKInferencer",
    "inferencer",
)
register_alias(
    "Devmate",
    f"{_P}.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer.DevmateCliInferencer",
    "inferencer",
)

# --- Template Management ---
register_alias(
    "TemplateManager",
    "rich_python_utils.string_utils.formatting.template_manager.template_manager.TemplateManager",
    "config",
)

# --- Conflict-Aware Prompt Builder ---
register_alias(
    "ConflictAwarePromptBuilder",
    f"{_P}.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer.make_conflict_aware_prompt_builder",
    "config",
)

# --- Upstream-Injecting Aggregator Prompt Builder ---
# For BTA aggregators whose wrapper template uses a separate
# ``{{ upstream_artifacts }}`` slot for peer outputs, distinct from the
# ``{{ input }}`` slot for the original BTA query.
register_alias(
    "UpstreamInjectingAggregatorPromptBuilder",
    f"{_P}.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer.make_upstream_injecting_aggregator_prompt_builder",
    "config",
)

# --- MultiFlow / MFDual tag parsers ---
# YAML resolves `_target_: WinnerParser` to the factory function, which
# is then called with no args to return the actual parser callable. The
# parser becomes the value of the inferencer's `winner_parser`,
# `end_condition`, or `response_parser` field at construction time.
register_alias(
    "WinnerParser",
    f"{_P}.common.inferencers.flow_parsers.make_winner_parser",
    "config",
)
register_alias(
    "DecisionStopParser",
    f"{_P}.common.inferencers.flow_parsers.make_decision_stop_parser",
    "config",
)
register_alias(
    "FinalPlanParser",
    f"{_P}.common.inferencers.flow_parsers.make_finalplan_parser",
    "config",
)
register_alias(
    "RankingParser",
    f"{_P}.common.inferencers.flow_parsers.make_ranking_parser",
    "config",
)

# --- Workspace ---
register_alias(
    "InferencerWorkspace",
    "agent_foundation.common.inferencers.inferencer_workspace.InferencerWorkspace",
    "config",
)

# --- Mock inferencers (for /mock_task developer tool) ---
register_alias(
    "MockBreakdownInferencer",
    f"{_P}.common.inferencers.mock_inferencers.mock_bta_components.MockBreakdownInferencer",
    "inferencer",
)
register_alias(
    "MockWorker",
    f"{_P}.common.inferencers.mock_inferencers.mock_bta_components.MockWorker",
    "inferencer",
)
register_alias(
    "MockAggregator",
    f"{_P}.common.inferencers.mock_inferencers.mock_bta_components.MockAggregator",
    "inferencer",
)

# --- Config objects ---
register_alias(
    "ContextBudget",
    f"{_P}.common.inferencers.agentic_inferencers.conversational.context.ContextBudget",
    "config",
)
register_alias(
    "LlmInferenceArgs",
    f"{_P}.common.inferencers.inference_args.CommonLlmInferenceArgs",
    "config",
)
register_alias(
    "ConsensusConfig",
    f"{_P}.common.inferencers.agentic_inferencers.common.ConsensusConfig",
    "config",
)

# --- Agents ---
# Most Agent fields are callables/protocols — YAML provides the structural
# skeleton, factories.py injects callable collaborators post-construction.
register_alias(
    "Agent",
    f"{_P}.agents.agent.Agent",
    "agent",
)
