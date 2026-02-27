import warnings
from typing import Callable, List, Iterable, Union, Mapping, Iterator, Sequence

from agent_foundation.common.experiment_management.constants import DIRNAME_SOURCE_DATA, DIRNAME_SAMPLED_DATA
from rich_python_utils.io_utils.json_io import iter_all_json_objs_from_all_sub_dirs, write_json_objs
from os import path

from rich_python_utils.path_utils.common import paths_in_same_directory, get_directory_if_paths_in_same_directory


def get_source_data_path(input_path_dataspace: Union[str, Sequence[str]], source_data_dir_name: str = DIRNAME_SOURCE_DATA):
    if not isinstance(input_path_dataspace, str):
        if len(input_path_dataspace) == 1:
            _input_path_data = input_path_dataspace[0]
        else:
            _input_path_data = get_directory_if_paths_in_same_directory(input_path_dataspace)
            if not _input_path_data:
                raise ValueError(f"Unable to obtain a unique root dir path from '{input_path_dataspace}'")
    else:
        _input_path_data = input_path_dataspace

    if isinstance(_input_path_data, str):
        if path.isfile(_input_path_data):
            _input_path_data = path.dirname(_input_path_data)

        if path.basename(_input_path_data) == source_data_dir_name:
            return _input_path_data
        else:
            return path.join(_input_path_data, source_data_dir_name)


def solve_dataspace_path(
        input_path_dataspace: str,
        data_iter: Callable[[str], Iterator] = iter_all_json_objs_from_all_sub_dirs,
        sample: Union[str, float, Mapping[str, Callable[[Iterator], Iterator]]] = None,
        data_writer: Callable[[Iterable, str], None] = write_json_objs,
        source_data_dir_name: str = DIRNAME_SOURCE_DATA
) -> Union[str, Mapping[str, str]]:
    """
    Determines the appropriate input data path based on the provided data space and sampling requirements.

    Args:
        input_path_dataspace (str): The root directory of the data space.
        data_iter (Callable[[str], Iterator], optional): A function that iterates over all data objects in the source directory. Defaults to `iter_all_json_objs_from_all_sub_dirs`.
        sample (Union[str, float, Mapping[str, Callable[[Iterator], Iterator]]], optional): Defines the sampling strategy:
            - If `None` or `1`, returns the source data path.
            - If `str`, it returns the path to the specific sample directory.
            - If `float`, samples the given percentage of data and writes it to the sample directory.
            - If `Mapping`, applies the corresponding sampling function for each key-value pair in the dictionary.
        data_writer (Callable[[Iterable, str], None], optional): A function to write the sampled data to disk. Required if `sample` is a float or `Mapping`. Defaults to `write_json_objs`.

    Returns:
        Union[str, Mapping[str, str]]: The path to the source data or sampled data, depending on the `sample` parameter.
        Returns a dictionary of paths if multiple samples are generated.

    Raises:
        FileNotFoundError: If the specified sample path does not exist.
        ValueError: If `data_iter` or `data_writer` is required but not provided.

    Examples:
        >>> solve_dataspace_path('/data', sample=None)
        '/data/source_data'

        >>> solve_dataspace_path('/data', sample='10')
        '/data/sampled_data/10'

        >>> solve_dataspace_path('/data', sample=0.2)
        '/data/sampled_data/20'

        >>> solve_dataspace_path('/data', sample={'train': lambda it: it, 'test': lambda it: it})
        {'train': '/data/sampled_data/train', 'test': '/data/sampled_data/test'}
    """
    input_source_data = get_source_data_path(
        input_path_dataspace=input_path_dataspace,
        source_data_dir_name=source_data_dir_name
    )

    if isinstance(input_path_dataspace, str) and path.isdir(input_path_dataspace):
        _input_path_dataspace = input_source_data
    else:
        _input_path_dataspace = input_path_dataspace

    if not path.exists(input_source_data):
        warnings.warn(
            f"Provided input path '{input_path_dataspace}' is not a managed data space. Use the original input path."
        )
        return input_path_dataspace

    try:
        sample = float(sample)
    except:
        pass

    if not sample or sample == 1:
        return _input_path_dataspace
    else:
        input_path_sample_data_root = path.join(path.dirname(input_source_data), DIRNAME_SAMPLED_DATA)
        if isinstance(sample, str):
            input_path_sample_data = path.join(input_path_sample_data_root, sample)
            if not path.exists(input_path_sample_data):
                raise FileNotFoundError(f"sample data '{sample}' does not exist under data space '{input_path_dataspace}")
            return input_path_sample_data
        elif isinstance(sample, float):
            input_path_sample_data = path.join(input_path_sample_data_root, str(sample * 100))
            if path.exists(input_path_sample_data):
                return input_path_sample_data
            else:
                if data_iter is None:
                    raise ValueError("'data_iter' must be provided to for data sampling")
                all_data = list(data_iter(_input_path_dataspace))
                import random
                data_sample = random.sample(all_data, int(len(all_data) * sample))
                if data_writer is None:
                    raise ValueError("'data_writer' must be provided to save the sample data")
                data_writer(data_sample, input_path_sample_data)
                return input_path_sample_data
        elif isinstance(sample, Mapping):
            output_samples = {}
            if data_iter is None:
                raise ValueError("'data_iter' must be provided to for data sampling")
            for sample_name, sampler in sample.items():
                input_path_sample_data = path.join(input_path_sample_data_root, sample_name)
                if not path.exists(input_path_sample_data):
                    data_writer(
                        sampler(data_iter(_input_path_dataspace)),
                        input_path_sample_data
                    )
                    output_samples[sample_name] = input_path_sample_data
            if len(output_samples) == 1:
                return next(iter(output_samples.values()))
            else:
                return output_samples
