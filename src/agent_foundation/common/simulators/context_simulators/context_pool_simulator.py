from collections import defaultdict
from typing import Mapping, List, Any, Union

from attr import attrs, attrib
import random

from agent_foundation.common.simulators.context_simulators.context_simulator_base import ContextSimulatorBase
from rich_python_utils.io_utils.json_io import iter_all_json_objs_from_all_sub_dirs, DEFAULT_JSON_FILE_PATTERN


def read_context_pool(input_path: str, file_pattern=DEFAULT_JSON_FILE_PATTERN) -> Mapping[str, List[str]]:
    context_pool = defaultdict(list)
    for jobj in iter_all_json_objs_from_all_sub_dirs(input_path, pattern=file_pattern):
        context_name = jobj['name']
        context_content = jobj['content']
        context_pool[context_name].append(context_content)
    return context_pool


@attrs
class ContextPoolSimulator(ContextSimulatorBase):
    context_pool: Union[str, Mapping[str, List[Any]]] = attrib()
    enabled_contexts: List[str] = attrib(default=None)

    def __attrs_post_init__(self):
        if isinstance(self.context_pool, str):
            self.context_pool = read_context_pool(self.context_pool)

        if not self.enabled_contexts:
            self.enabled_contexts = list(self.context_pool.keys())

    def __call__(self, data, existing_context) -> Mapping[str, Any]:
        simulated_contexts = {}
        for context_name in self.enabled_contexts:
            simulated_contexts[context_name] = random.choice(self.context_pool[context_name])
        return simulated_contexts
