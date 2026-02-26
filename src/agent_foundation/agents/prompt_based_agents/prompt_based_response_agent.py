from attr import attrs

from science_modeling_tools.agents.agent_actor import AgentActionResult
from science_modeling_tools.agents.prompt_based_agents.prompt_based_action_agent import PromptBasedActionAgent
from rich_python_utils.common_utils import has_single_key, iter_


@attrs
class PromptBasedResponseActionAgent(PromptBasedActionAgent):
    def _parse_instant_response(self, instant_response):
        if isinstance(instant_response, str):
            return instant_response
        else:
            answers = []

            for response in iter_(instant_response):
                if has_single_key(response, 'Response'):
                    response = response['Response']
                    answers.append(response['Answer'])

            return f"<html><body>{'\n\n'.join(answers)}</body></html>"

    def _get_agent_results(self, trigger_action, trigger_action_results, new_states):
        if self.states:
            last_agent_state = self.states[-1]
            return AgentActionResult(
                summary=last_agent_state.response.instant_response,
                details=self._get_action_result_string(trigger_action_results),
                source=trigger_action_results.source,
                action=trigger_action
            )
