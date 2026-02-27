from enum import StrEnum
from typing import Union, Any, Iterable, List

from attr import attrs, attrib

from agent_foundation.agents.agent_response import AgentResponse, AgentAction
from agent_foundation.common.inferencers.agentic_inferencers.common import InferencerResponse
from rich_python_utils.common_utils import get_


class AgentTaskStatusFlags(StrEnum):
    Completed = 'Completed'
    Ongoing = 'Ongoing'
    Pending = 'Pending'


@attrs
class AgentStateItem:
    new_task: bool = attrib(default=True)
    task_status: Union[AgentTaskStatusFlags, str, Any] = attrib(default=None)
    task_status_description: str = attrib(default=None)
    task_status_description_extended: str = attrib(default=None)

    last_action_source: str = attrib(default=None)
    last_action_type: Any = attrib(default=None)
    last_anchor_action_type: Any = attrib(default=None)
    user_input: Any = attrib(default=None)
    reasoner_input: str = attrib(default=None)
    raw_response: Union[str, InferencerResponse, Any] = attrib(default=None)
    response: Union[str, AgentResponse] = attrib(default=None)
    action_results: Any = attrib(default=None)
    task_label: str = attrib(default=None)

    def __str__(self):
        return self.task_status.__str__()

    def __hash__(self):
        return self.task_status.__hash__()

    def __cmp__(self, other):
        return self.task_status.__cmp__(other)


@attrs
class AgentStates:
    _states: list = attrib(factory=list)
    last_action: Any = attrib(default=None)
    last_action_source: Any = attrib(default=None)
    last_anchor_action: Any = attrib(default=None)
    last_action_type: Any = attrib(default=None)
    last_anchor_action_type: Any = attrib(default=None)

    def append(self, state_item):
        self._states.append(state_item)

    def __getitem__(self, index):
        return self._states[index]

    def __len__(self):
        return len(self._states)

    def __delitem__(self, index):
        del self._states[index]

    def __iter__(self):
        yield from self._states

    def __copy__(self):
        new_states = AgentStates()
        new_states._states = self._states.copy()
        new_states.last_action = self.last_action
        new_states.last_action_type = self.last_action_type
        new_states.last_anchor_action = self.last_anchor_action
        new_states.last_anchor_action_type = self.last_anchor_action_type
        return new_states

    @staticmethod
    def _resolve_action_type(action):
        if isinstance(action, AgentAction):
            return action.type
        elif isinstance(action, str):
            return action
        else:
            return get_(action, key1='type', key2='action_type', default=action)


    @staticmethod
    def _resolve_action_source(action):
        if isinstance(action, AgentAction):
            return action.source
        else:
            return get_(action, key1='source', key2='action_source', default=None)


    def set_last_action(self, action, anchor_actions_types: Iterable = None):
        last_action_type = self._resolve_action_type(action)
        if anchor_actions_types and last_action_type in anchor_actions_types:
            self.last_anchor_action = action
            self.last_anchor_action_type = last_action_type
        self.last_action = action
        self.last_action_type = last_action_type

        action_source = self._resolve_action_source(action)
        if action_source:
            self.last_action_source = action_source

