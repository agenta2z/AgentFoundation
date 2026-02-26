from typing import Dict, Any, Iterable, Union, Callable, Mapping, Sequence

from attr import attrs, attrib

from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_utils.iter_helper import in__


@attrs
class AgentActor(Debuggable):
    actor: Callable[[Any], Any] = attrib()
    target_action_type: Union[str, Iterable[str]] = attrib(default=None)

    def get_actor_input(
            self,
            action_results: Sequence,
            task_input: Any,
            action_type: str,
            action_target: str = None,
            action_args: Mapping = None,
            attachments: Sequence = None
    ):
        raise NotImplementedError

    @property
    def source(self):
        return self.actor.source

    def __call__(
            self,
            action_results: Sequence,
            task_input: Any,
            action_type: str,
            action_target: str = None,
            action_args: Mapping = None,
            attachments: Sequence = None
    ):
        # if not isinstance(action_results, Sequence):
        #     raise TypeError(f"'action_results' must be a sequence, got {type(action_results)}")

        if not in__(action_type, self.target_action_type):
            raise ValueError(
                f"action_type '{action_type}' cannot match target_action_type '{self.target_action_type}'"
            )

        actor_input = self.get_actor_input(
            action_results=action_results,
            task_input=task_input,
            action_type=action_type,
            action_target=action_target,
            action_args=action_args,
            attachments=attachments
        )

        return self.actor(actor_input)





@attrs
class AgentActionResult:
    summary: Any = attrib()
    details: Any = attrib(default=None)
    action: Any = attrib(default=None)
    anchor_action: Any = attrib(default=None)
    source: str = attrib(default=None)
    task_label: str = attrib(default=None)
    last_action_response: Any = attrib(default=None)

    def __str__(self):
        return str(self.summary)