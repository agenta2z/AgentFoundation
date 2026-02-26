from enum import StrEnum
from typing import Dict, Any, Iterable, Union, Mapping

from attr import attrs, attrib
from rich_python_utils.io_utils.json_io import artifact_field

class AgentResponseFormat(StrEnum):
    XML = 'xml'
    JSON = 'json'
    Other = 'other'

@attrs
class AgentAction:
    reasoning: str = attrib(default=None)
    target: str = attrib(default=None)
    type: str = attrib(default=None)
    is_follow_up: bool = attrib(default=False)
    memory_target: str = attrib(default=None)
    args: Dict[str, Any] = attrib(default=None)
    source: Any = attrib(default=None)
    result: Any = attrib(default=None)

@attrs
class AgentResponse:
    raw_response: Any = attrib(default=None)
    instant_response: str = attrib(default=None)
    instant_learnings: Mapping = attrib(default=None)
    next_actions: Iterable[Iterable[Union[str, AgentAction]]] = attrib(default=None)

    def __str__(self):
        return self.instant_response
