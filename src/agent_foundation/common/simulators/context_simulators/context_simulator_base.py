from abc import ABC
from typing import Any, Mapping

from attr import attrs


@attrs
class ContextSimulatorBase(ABC):
    def __call__(self, data, existing_context) -> Mapping[str, Any]:
        raise NotImplementedError
