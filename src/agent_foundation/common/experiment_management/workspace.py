import glob
from functools import partial
from os import path
from typing import List, Union, Iterable, Mapping

from attr import attrs, attrib

from agent_foundation.common.experiment_management.constants import (
    SPACE_TYPE_ARTIFACTS,
    SPACE_TYPE_DATA,
    SPACE_TYPE_EVALUATION,
    SUBSPACE_PROMPTS,
    SUBSPACE_CONFIGS
)
from rich_python_utils.common_utils import iter_
from rich_python_utils.path_utils.path_join import join_
from rich_python_utils.path_utils.common import ensure_parent_dir_existence


SUBSPACE_NAME_AND_VERSIONS = Iterable[
    Union[
        str,
        Mapping[str, Iterable[str]]
    ]
]

SUBSPACE_TYPE_AND_STRUCTURE = Mapping[
    str,
    SUBSPACE_NAME_AND_VERSIONS
]


@attrs
class Workspace:
    space: str = attrib()
    root: str = attrib(default=None)
    subspaces: Union[
        SUBSPACE_TYPE_AND_STRUCTURE,
        Iterable[
            Union[
                str,
                SUBSPACE_TYPE_AND_STRUCTURE
            ]
        ]
    ] = attrib(default=(SPACE_TYPE_ARTIFACTS, SPACE_TYPE_DATA, SPACE_TYPE_EVALUATION), converter=tuple)
    auto_create_parent_dir: bool = attrib(default=True)
    verbose: bool = attrib(default=True)

    def __attrs_post_init__(self):
        if not self.root:
            self.root = path.expanduser('~')
        for space in iter_(self.subspaces):
            if isinstance(space, str):
                setattr(self, f'get_{space}_path', partial(self._get_path, subspace_type=space))
            elif isinstance(space, Mapping):
                for space_type, subspaces in space.items():
                    setattr(self, f'get_{space_type}_path', partial(self._get_path, subspace_type=space_type))
                    for subspace in subspaces:
                        if isinstance(subspace, str):
                            setattr(self, f'get_{space_type}_{subspace}_path', partial(self._get_path, subspace_type=space_type, subspace=subspace))
                        elif isinstance(subspace, Mapping):
                            for subspace_name, versions in subspace.items():
                                setattr(self, f'get_{space_type}_{subspace_name}_path', partial(self._get_path, subspace_type=space_type, subspace=subspace_name))
                                for version in versions:
                                    setattr(self, f'get_{space_type}_{subspace_name}_{version}_path', partial(self._get_path, subspace_type=space_type, subspace=subspace_name, version=version))

    def _get_path(self, subspace_type: str, subspace: str = None, version: str = None, target: str = None) -> Union[str, List[str]]:
        _path = join_(self.root, self.space, subspace_type, subspace, version, target)
        if '*' in subspace_type or '?' in subspace_type:
            raise ValueError(f"Cannot specify pattern for workspace type; got '{subspace_type}'")
        if subspace and ('*' in subspace or '?' in subspace):
            raise ValueError(f"Cannot specify pattern for subspace; got '{subspace}'")

        if (
                (version and ('*' in version or '?' in version))
                or (target and ('*' in target or '?' in target))
        ):
            return glob.glob(_path)
        else:
            if self.auto_create_parent_dir:
                ensure_parent_dir_existence(_path, verbose=self.verbose)
            return _path


@attrs
class EvaluationWorkspace(Workspace):
    artifacts_subspace_name: str = attrib(default=SPACE_TYPE_ARTIFACTS)
    artifacts_configs_dirname: str = attrib(default=SUBSPACE_CONFIGS)
    artifacts_configs_types: Iterable[str] = attrib(default=[])

    data_subspace_name: str = attrib(default=SPACE_TYPE_DATA)
    data_subspace_versions: SUBSPACE_NAME_AND_VERSIONS = attrib(default=[])
    evaluation_subspace_name: str = attrib(default=SPACE_TYPE_EVALUATION)
    evaluation_subspace_versions: SUBSPACE_NAME_AND_VERSIONS = attrib(default=[])

    def __attrs_post_init__(self):
        self.subspaces = {
            self.artifacts_subspace_name: [
                *iter_(self.subspaces[self.artifacts_subspace_name]),
                {self.artifacts_configs_dirname: self.artifacts_configs_types}
            ],
            self.data_subspace_name: self.data_subspace_versions,
            self.evaluation_subspace_name: self.evaluation_subspace_versions
        }
        super(EvaluationWorkspace, self).__attrs_post_init__()


@attrs
class LlmEvaluationWorkspace(EvaluationWorkspace):
    artifacts_prompts_dirname: str = attrib(default=SUBSPACE_PROMPTS)
    artifacts_prompts_types: Iterable[str] = attrib(default=[])

    def __attrs_post_init__(self):
        self.subspaces = {
            self.artifacts_subspace_name: {self.artifacts_prompts_dirname: self.artifacts_prompts_types}
        }
        super(LlmEvaluationWorkspace, self).__attrs_post_init__()
