from attr import attrs

from agent_foundation.agents.prompt_based_agents.prompt_based_action_agent import PromptBasedActionAgent
from rich_python_utils.common_utils import has_single_key, iter_


@attrs
class PromptBasedSummaryActionAgent(PromptBasedActionAgent):
    def _parse_instant_response(self, instant_response):
        answers = []

        for response in iter_(instant_response):
            if has_single_key(response, 'Response'):
                response = response['Response']
                answers.append(response['Answer'])

        return f"<html><body>{'\n\n'.join(answers)}</body></html>"
