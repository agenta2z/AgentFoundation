---
# AgentFoundation agentic freeform config.
#
# Extends the built-in freeform config (inherits all default tools) and
# properly declares ``model_name``, ``max_iterations``, and
# ``max_output_tokens`` as template variables so they are substituted
# into the YAML frontmatter instead of being silently ignored.
#
# Usage (CLI):
#   devmate run fbcode/agent_foundation/.../configs/freeform_agentic \
#       prompt="..." model_name="claude-opus-4.6" max_iterations=200
#
# Usage (Python - DevmateCliInferencer / DevmateSDKInferencer):
#   inferencer = DevmateCliInferencer(
#       config_name=DevmateConfig.AGENT_FOUNDATION_AGENTIC,
#       model_name="claude-opus-4.6",
#   )
#
# @param prompt string The task prompt.
# @param model_name string Model to use (default: claude-opus-4.6).
# @param max_iterations int Max agent iterations (default: 200).
# @param max_output_tokens int Max LLM output tokens (default: 64000).
# @param thinking_budget_tokens int Extended thinking budget (default: 10000).
# @param enable_shell bool Enable shell/command execution (default: true).

extends: 'freeform.md'

orchestrator:
  model_name: ${{ model_name:str = claude-opus-4.6 }}
  max_iterations: ${{ max_iterations:int = 200 }}
  max_time_mins: 60
  max_total_tokens: 10000000
  create_commit: false
  backup_commit: false

# NOTE: max_output_tokens controls the ORCHESTRATOR LLM's output limit.
# However, devmate's edit tool uses a SEPARATE patchgen LLM with its own
# hardcoded max_tokens=8192 (in devai/config/patchgen.py). This means large
# edits (e.g., writing full documentation files) will be truncated at ~8192
# tokens regardless of this setting. See:
#   docs/dev/issues/devmate/patchgen_max_tokens_8192.md
llm:
  max_output_tokens: ${{ max_output_tokens:int = 64000 }}
  thinking_budget_tokens: ${{ thinking_budget_tokens:int = 10000 }}
  # temperature must be 1 when extended thinking is enabled (Anthropic API requirement)
  temperature: 1

mcp_servers:
  tools:
    # write_file: direct file write with content — NO patchgen LLM call, NO 8192
    # token limit. Use this for creating/overwriting files with large content
    # (e.g., documentation). The content is limited only by max_output_tokens above.
    write_file:
      llm_enabled: true
      config:
        allow_paths_outside_repository: true
    # str_replace_edit: deterministic search/replace — NO patchgen LLM call.
    # Use this for targeted edits instead of the default "edit" tool (which
    # goes through patchgen and is capped at 8192 output tokens).
    str_replace_edit:
      llm_enabled: true
    execute_command:
      tool_name: shell
      llm_enabled: ${{ enable_shell:bool = true }}
      config:
        enable_preapproved_commands: true
        timeout_seconds: 1800
    search_files:
      config:
        allow_paths_outside_repository: true
    # --- Additional tools (not enabled by default in devmate) ---
    # File browsing / discovery
    list_files:
      llm_enabled: true
    read_directory:
      llm_enabled: true
    glob:
      llm_enabled: true
    find_file:
      llm_enabled: true
    create_file:
      llm_enabled: true
    # Code search (semantic)
    search_class:
      llm_enabled: true
    search_method:
      llm_enabled: true
    search_method_in_class:
      llm_enabled: true
    # History / diff
    read_file_history:
      llm_enabled: true
    get_local_changes:
      llm_enabled: true
    # Reasoning / planning
    think:
      llm_enabled: true
    sequential_thinking:
      llm_enabled: true
    write_todo_list:
      llm_enabled: true
    # Batch editing — multiple search/replace in one call, NO patchgen
    str_replace_multi_edits:
      llm_enabled: true
    # Knowledge — load content from URLs, diffs, tasks, SEVs, pastes
    knowledge_load_vsc:
      llm_enabled: true
    # Knowledge — search Meta internal knowledge base
    knowledge_search_vsc:
      llm_enabled: true
---

${{ prompt:str }}
