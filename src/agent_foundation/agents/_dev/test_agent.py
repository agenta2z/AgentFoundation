from agent_foundation.agents.agent import Agent
from agent_foundation.ui.terminal_interactive import TerminalInteractive

agent = Agent(
    user_profile="static profile",
    context="session context",
    reasoner=lambda x: f"Processed {x}",
    interactive=TerminalInteractive()
)
agent.start()
