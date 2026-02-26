"""
Example: Integrating UI Inferencer Selection with Web Agent Service

This example shows how to modify the web agent service to use
the inferencer selected in the UI debugger's Settings tab.

The key idea is that the agent factory can call app.get_session_inferencer(session_id)
to get the UI-selected inferencer and use it when creating agents.
"""

# In your WebAgent service's task.py or agent creation code:

def create_agent_factory_with_ui_inferencer_selection(app):
    """
    Create an agent factory that respects the UI's inferencer selection.

    Args:
        app: DashInteractiveAppWithLogs instance

    Returns:
        Factory function that creates agents with UI-selected inferencers
    """
    def agent_factory_with_ui_selection(session_id):
        """Create agent with inferencer selected in UI."""
        from science_modeling_tools.agents.prompt_based_agents.prompt_based_planning_agent import (
            PromptBasedActionPlanningAgent
        )
        from science_modeling_tools.common.inferencers.api_inferencers.claude_api_inferencer import (
            ClaudeApiInferencer
        )

        # Get the UI-selected inferencer for this session
        # If user selected something in Settings tab, use that
        # Otherwise, fall back to default
        ui_selected_inferencer = app.get_session_inferencer(session_id)

        if ui_selected_inferencer is not None:
            # Use the UI-selected inferencer
            reasoner = ui_selected_inferencer
            print(f"Using UI-selected inferencer: {reasoner.__class__.__name__}")
        else:
            # Fall back to default inferencer
            reasoner = ClaudeApiInferencer(
                max_retry=3,
                default_inference_args={'connect_timeout': 20, 'response_timeout': 120}
            )
            print("Using default ClaudeApiInferencer")

        # Create your agent with the selected reasoner
        planning_agent = PromptBasedActionPlanningAgent(
            default_prompt_template=load_prompt_template('planning_agent_prompt_template'),
            # ... other config ...
            reasoner=reasoner,  # Use the UI-selected or default inferencer
            # ... rest of config ...
        )

        return planning_agent

    return agent_factory_with_ui_selection


# Usage in your Flask app initialization:

def initialize_webagent_with_ui_debugger():
    """Initialize web agent service with UI debugger."""
    from science_modeling_tools.ui.dash_interactive.dash_interactive_app_with_logs import (
        DashInteractiveAppWithLogs
    )

    # Create the debugger UI
    app = DashInteractiveAppWithLogs(
        title="Web Agent Debugger",
        port=8050,
        debug=True
    )

    # Set the agent factory that uses UI-selected inferencers
    factory = create_agent_factory_with_ui_inferencer_selection(app)
    app.set_agent_factory(factory)

    # Run the app
    print("Web Agent Debugger UI available at http://localhost:8050")
    print("Use Settings tab to select MockClarificationInferencer or ClaudeApiInferencer")
    app.run()


# Alternative: If you want to use inferencer mode (no agents, just direct inferencer calls):

def initialize_simple_inferencer_debugger():
    """Initialize debugger for testing inferencers directly (no full agent)."""
    from science_modeling_tools.ui.dash_interactive.dash_interactive_app_with_logs import (
        DashInteractiveAppWithLogs
    )

    app = DashInteractiveAppWithLogs(
        title="Inferencer Debugger",
        port=8050,
        debug=True
    )

    # Use inferencer mode - calls inferencers directly without agent wrapper
    app.set_inferencer_mode()

    # Now users can:
    # 1. Go to Settings tab
    # 2. Select "Mock: Clarification Inferencer" to test clarification flows
    # 3. Select "Claude API Inferencer (Sonnet)" to test real API calls
    # 4. Click "Apply Changes"
    # 5. Type messages and see responses

    app.run()


if __name__ == '__main__':
    # Choose one:

    # Option 1: Full web agent with UI inferencer selection
    # initialize_webagent_with_ui_debugger()

    # Option 2: Simple inferencer testing mode
    initialize_simple_inferencer_debugger()
