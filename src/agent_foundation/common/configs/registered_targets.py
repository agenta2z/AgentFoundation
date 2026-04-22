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
    "Conversational",
    f"{_P}.common.inferencers.agentic_inferencers.conversational.conversational_inferencer.ConversationalInferencer",
    "inferencer",
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
    "RovoChat",
    f"{_P}.common.inferencers.agentic_inferencers.external.rovochat.rovochat_inferencer.RovoChatInferencer",
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
