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
