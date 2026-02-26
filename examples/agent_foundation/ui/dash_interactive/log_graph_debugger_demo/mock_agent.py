"""
Mock agent components for testing and demonstrations.

This module provides mock implementations of agent components including:
- MockInteractive: Mock interactive interface
- ComplexAgentReasoner: Reasoner that creates parallel branching scenarios
- BranchedAgentReasoner: Reasoner for branched agents
- TrackingActor: Mock actor that tracks executed actions
- ComplexAgent: Agent subclass with specific implementation for demos
"""

from attr import attrs, attrib

from science_modeling_tools.agents.agent import Agent
from science_modeling_tools.agents.agent_response import AgentResponse, AgentAction
from science_modeling_tools.agents.agent_state import AgentStateItem
from science_modeling_tools.ui.interactive_base import InteractiveBase, InteractionFlags


@attrs
class MockInteractive(InteractiveBase):
    """
    Mock interactive interface for the agent.
    
    Inherits from InteractiveBase to ensure proper interface compliance.
    Useful for testing and demonstrations where no real user interaction is needed.
    """
    inputs: list = attrib(factory=list)
    input_index: int = attrib(default=0, init=False)
    responses: list = attrib(factory=list, init=False)

    def get_input(self):
        """Get the next input from the pre-configured inputs list."""
        if self.input_index < len(self.inputs):
            result = self.inputs[self.input_index]
            self.input_index += 1
            return result
        return "continue"

    def _send_response(self, response, flag: InteractionFlags = InteractionFlags.TurnCompleted):
        """Store the response in the responses list."""
        self.responses.append(response)
    
    def reset_input(self, flag: InteractionFlags):
        """Reset input state (no-op for mock)."""
        pass


class ComplexAgentReasoner:
    """
    Reasoner for a complex agent that demonstrates parallel branching.

    Execution Flow:
    1. Search for information (sequential)
    2. Analyze in 3 parallel branches (each branch runs a separate agent)
       - Each branched agent does its analysis and completes
       - Results are merged by summary node
    3. Write final report based on merged analysis (sequential)
    4. Complete

    Note: The parallel branches use BranchedAgentReasoner which completes
    immediately after executing their action. The main agent then takes the
    merged results and writes the final report.
    """
    def __init__(self):
        self.call_count = 0

    def __call__(self, reasoner_input, reasoner_config):
        self.call_count += 1

        if self.call_count == 1:
            # Iteration 1: Search for information
            return AgentResponse(
                instant_response="Searching for information...",
                next_actions=[
                    [AgentAction(type="SearchAction", target="quantum_computing")]
                ]
            )
        elif self.call_count == 2:
            # Iteration 2: Parallel analysis (3 branches)
            # Each action spawns a branched agent that analyzes its specific aspect
            return AgentResponse(
                instant_response="Analyzing results in parallel (3 branches)...",
                next_actions=[
                    [
                        AgentAction(type="AnalyzeAlgorithms", target="algorithms"),
                        AgentAction(type="AnalyzeHardware", target="hardware"),
                        AgentAction(type="AnalyzeUseCases", target="use_cases")
                    ]
                ]
            )
        elif self.call_count <= 5:
            # Calls 3-5: Branched agents (one per parallel branch)
            # Each branched agent completes immediately after executing its action
            # This prevents branched agents from creating their own WorkGraphs
            return AgentResponse(
                instant_response="Branch analysis complete",
                next_actions=[]  # Empty = branched agent completes
            )
        elif self.call_count == 6:
            # Iteration 3 of main agent: Write final report based on merged analysis results
            return AgentResponse(
                instant_response="Writing final report from merged analysis...",
                next_actions=[
                    [AgentAction(type="WriteReport", target="final_report")]
                ]
            )
        else:
            # Iteration 4 of main agent (call 7+): Complete
            return AgentResponse(
                instant_response="Task completed!",
                next_actions=[]  # Empty = agent completes
            )


class BranchedAgentReasoner:
    """
    Reasoner for branched agents - completes immediately after initial action.
    
    This reasoner is used by the branched agents created during parallel execution.
    Each branched agent:
    1. Executes its assigned action (e.g., AnalyzeAlgorithms)
    2. Returns results immediately without further reasoning
    3. Results are collected and merged by the summary node
    
    The main agent then continues with the merged results.
    """
    def __init__(self):
        self.call_count = 0

    def __call__(self, reasoner_input, reasoner_config):
        self.call_count += 1
        # Branched agents complete immediately after their action executes
        # The action itself (e.g., AnalyzeAlgorithms) has already been executed
        # by the time this reasoner is called
        return AgentResponse(
            instant_response="Branch analysis complete",
            next_actions=[]  # No further actions - complete immediately
        )


class TrackingActor:
    """Mock actor that tracks executed actions."""
    def __init__(self):
        self.executed_actions = []

    def __call__(self, action_type=None, action_target=None, action_args=None, **kwargs):
        """
        Actor receives action attributes with 'action_' prefix.
        This matches the Agent's default actor_args_transformation.
        """
        action_desc = f"{action_type}({action_target})"
        self.executed_actions.append(action_desc)
        return {"result": f"Executed {action_desc}", "status": "success"}


class ComplexAgent(Agent):
    """Agent subclass with specific implementation for demo."""

    def _parse_raw_response(self, raw_response):
        """Parse the raw response into AgentResponse and state."""
        if isinstance(raw_response, AgentResponse):
            # Create a simple agent state
            agent_state = AgentStateItem()
            return raw_response, agent_state
        return AgentResponse(instant_response=str(raw_response), next_actions=[]), AgentStateItem()
