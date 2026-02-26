from copy import copy
from typing import Callable, Mapping, Union, Any, Iterable, Tuple, List, Dict

from rich_python_utils.common_utils import iter_, explode_map, add_key_prefix_suffix, get_relevant_named_args

SCORE_TYPE = Union[float, int, bool]
TAG_TYPE = Union[int, bool, str]
SCORING_TARGET_TYPE = Union[Mapping[str, Any], Tuple]
SCORING_TARGET_EXTRACTOR_TYPE = Callable[[Any], Mapping[str, SCORING_TARGET_TYPE]]
SCORER_TYPE = Callable[..., Mapping[str, SCORE_TYPE]]
TAG_EXTRACTOR_TYPE = Callable[[Any], Mapping[str, TAG_TYPE]]
SCORE_AGGREGATOR_TYPE = Callable[[Iterable[SCORE_TYPE]], SCORE_TYPE]


def score_datapoint(
        data_point: Any,
        preprocessors: Iterable[SCORING_TARGET_EXTRACTOR_TYPE],
        scorers: Mapping[str, SCORER_TYPE],
        explode: Mapping[str, List[str]] = None,
        associate_scoring_results_with_datapoint: Union[bool, str] = False,
        add_preprocessing_results_to_datapoint: Union[bool, Callable] = False,
        add_scoring_results_to_datapoint: Union[bool, Callable] = False
) -> Mapping:
    """
    Computes a metric (or metrics) for a given data point, and optionally returns associated tags.

    Args:
        data_point (Any): The data point to compute metrics for.
        preprocessors (Iterable[SCORING_TARGET_EXTRACTOR_TYPE]): Iterable of functions to extract the prediction and reference (ground truth) from the data point.
        scorers (Mapping[str, SCORER_TYPE]): Mapping of metric names to functions that compute the metric(s).
        explode (Mapping[str, List[str]], optional): Mapping of scorer target names to lists of keys for exploding the scorer target. Defaults to None.
        add_preprocessing_results_to_datapoint (Union[bool, Callable], optional): If True, add preprocessing results to the data point. If callable, it will be called with data_point and scoring_target. Defaults to False.
        add_scoring_results_to_datapoint (Union[bool, Callable], optional): If True, add scoring results to the data point. If callable, it will be called with data_point and all_scoring_results. Defaults to False.

    Returns:
        Mapping[str, Any]: Computed metric(s) and associated tags.

    Examples:
        >>> def mock_extractor(data_point):
        ...     return {'accuracy_scorer': {'prediction': data_point['prediction'], 'reference': data_point['reference']}}
        >>> def mock_scorer(prediction, reference):
        ...     return {'accuracy': prediction == reference}
        >>> scorers = {'accuracy_scorer': mock_scorer}
        >>> preprocessors = [mock_extractor]

        >>> data_point = {'prediction': 1, 'reference': 1}
        >>> score_datapoint(data_point, preprocessors, scorers)
        {'accuracy_scorer': {'accuracy_scorer.accuracy': True}}

        >>> data_point = {'prediction': 0, 'reference': 1}
        >>> score_datapoint(data_point, preprocessors, scorers)
        {'accuracy_scorer': {'accuracy_scorer.accuracy': False}}

        >>> data_point = {'prediction': 0, 'reference': [0, 1, 2]}
        >>> score_datapoint(data_point, preprocessors, scorers, explode={'accuracy_scorer': ['reference']})
        {'accuracy_scorer': [{'accuracy_scorer.accuracy': True}, {'accuracy_scorer.accuracy': False}, {'accuracy_scorer.accuracy': False}]}
    """
    if associate_scoring_results_with_datapoint is True:
        associate_scoring_results_with_datapoint = '_data'
    scoring_targets = {}

    for preprocessor in preprocessors:
        extracts = preprocessor(data_point)
        if extracts:
            scoring_targets.update(extracts)

    if not scoring_targets:
        return {}

    if add_preprocessing_results_to_datapoint is True:
        data_point.update(scoring_targets)
    elif callable(add_preprocessing_results_to_datapoint):
        add_preprocessing_results_to_datapoint(data_point, scoring_targets)

    all_scoring_results = {}
    if associate_scoring_results_with_datapoint:
        all_scoring_results_no_datapoint = {}
    else:
        all_scoring_results_no_datapoint = all_scoring_results

    for scorer_target_name, scorer in scorers.items():
        if scorer_target_name in scoring_targets:
            score_target = scoring_targets[scorer_target_name]
            has_explosion = False
            if explode and scorer_target_name in explode:
                explode_keys = explode[scorer_target_name]
                if explode_keys:
                    score_target = tuple(explode_map(score_target, explode_keys))
                    has_explosion = True
            if has_explosion:
                scoring_results = []
                scoring_results_no_datapoint = []
                for score_target_item in score_target:
                    if isinstance(score_target_item, Mapping):
                        scoring_results_item = add_key_prefix_suffix(scorer(**get_relevant_named_args(scorer, **score_target_item)), prefix=scorer_target_name, sep='.')
                    elif isinstance(score_target_item, Tuple):
                        scoring_results_item = add_key_prefix_suffix(scorer(*score_target_item), prefix=scorer_target_name, sep='.')
                    else:
                        raise ValueError(f"Unsupported scoring target '{score_target_item}'")
                    scoring_result = {**scoring_results_item, **score_target_item}

                    if associate_scoring_results_with_datapoint:
                        scoring_results_no_datapoint.append(copy(scoring_result))
                        scoring_result[associate_scoring_results_with_datapoint] = data_point
                    scoring_results.append(scoring_result)
            else:
                if isinstance(score_target, Mapping):
                    scoring_results = add_key_prefix_suffix(scorer(**get_relevant_named_args(scorer, **score_target)), prefix=scorer_target_name, sep='.')
                elif isinstance(score_target, Tuple):
                    scoring_results = add_key_prefix_suffix(scorer(*score_target), prefix=scorer_target_name, sep='.')
                else:
                    raise ValueError(f"Unsupported scoring target '{score_target}'")
                scoring_results = {**scoring_results, **score_target}
                if associate_scoring_results_with_datapoint:
                    scoring_results_no_datapoint = copy(scoring_results)
                    scoring_results[associate_scoring_results_with_datapoint] = data_point

            all_scoring_results[scorer_target_name] = scoring_results
            if associate_scoring_results_with_datapoint:
                all_scoring_results_no_datapoint[scorer_target_name] = scoring_results_no_datapoint

    if add_scoring_results_to_datapoint is True:
        data_point.update(all_scoring_results_no_datapoint)
    elif callable(add_scoring_results_to_datapoint):
        add_scoring_results_to_datapoint(data_point, all_scoring_results_no_datapoint)

    return all_scoring_results


def score_precision_and_recall_by_hits(
        prediction: Union[
            Any, List[Any], List[List[Any]],
            Mapping[str, Union[Any, List[Any]]]
        ],
        reference: Union[
            Any, List[Any],
            Mapping[str, Any]
        ],
        comparer: Callable[[Any, Any], float] = None,
        k=1,
        undergrab_threshold: float = 1.0,
        overgrab_threshold: float = 0.0
) -> Dict[str, float]:
    """
    Calculates precision and recall metrics for predictions against references.

    Args:
        prediction (Union[Any, List[Any], List[List[Any]]]): The prediction or list of predictions.
        reference (Union[Any, List[Any]]): The reference or list of references (ground truth).
        comparer (Callable[[Any, Any], float], optional): Function to compare prediction and reference items, returning a score between 0 and 1. Defaults to None.
        k (int, optional): Value of k for calculating precision@k and recall@k. Defaults to 1.
        undergrab_threshold (float, optional): The threshold below which a reference is considered undergrabbed at k. Defaults to 1.0.
        overgrab_threshold (float, optional): The threshold above which a prediction is considered overgrabbed at k. Defaults to 0.0.

    Returns:
        Dict[str, float]: Dictionary containing precision, recall, precision@k, and recall@k.

    Raises:
        ValueError: If comparer score is not between 0 and 1.

    Examples:
        >>> score_precision_and_recall_by_hits(3, 3)
        {'precision': 1.0, 'recall': 1.0, 'precision@1': 1.0, 'recall@1': 1.0, 'overgrab': [], 'undergrab': [], 'overgrab@1': [], 'undergrab@1': []}

        >>> score_precision_and_recall_by_hits([1, 2, 3], [2, 3, 4])
        {'precision': 0.6666666666666666, 'recall': 0.6666666666666666, 'precision@1': 0.0, 'recall@1': 0.0, 'overgrab': [1], 'undergrab': [4], 'overgrab@1': [1], 'undergrab@1': [2, 3, 4]}

        >>> score_precision_and_recall_by_hits([[1, 2], [3, 4]], [2, 4])
        {'precision': 1.0, 'recall': 1.0, 'precision@1': 1.0, 'recall@1': 0.5, 'overgrab': [], 'undergrab': [], 'overgrab@1': [], 'undergrab@1': [4]}

        >>> def custom_comparer(a, b):
        ...     return 1.0 if a == b else 0.0
        >>> score_precision_and_recall_by_hits([1, 2, 3], [2, 3, 4], comparer=custom_comparer)
        {'precision': 0.6666666666666666, 'recall': 0.6666666666666666, 'precision@1': 0.0, 'recall@1': 0.0, 'overgrab': [1], 'undergrab': [4], 'overgrab@1': [1], 'undergrab@1': [2, 3, 4]}

        >>> def fuzzy_comparer(a, b):
        ...     return 1.0 if a in b else 0.0
        >>> score_precision_and_recall_by_hits([1, 2, 3], [(1, 2), (2, 3), (3, 4)], comparer=fuzzy_comparer, k=2)
        {'precision': 1.0, 'recall': 1.0, 'precision@1': 1.0, 'precision@2': 1.0, 'recall@1': 0.3333333333333333, 'recall@2': 0.6666666666666666, 'overgrab': [], 'undergrab': [], 'overgrab@1': [], 'overgrab@2': [], 'undergrab@1': [(2, 3), (3, 4)], 'undergrab@2': [(3, 4)]}

        >>> # Test case for precision@2 and recall@2
        >>> score_precision_and_recall_by_hits([1, 2, 3], [1, 2, 4], k=2)
        {'precision': 0.6666666666666666, 'recall': 0.6666666666666666, 'precision@1': 1.0, 'precision@2': 1.0, 'recall@1': 0.3333333333333333, 'recall@2': 0.6666666666666666, 'overgrab': [3], 'undergrab': [4], 'overgrab@1': [], 'overgrab@2': [], 'undergrab@1': [2, 4], 'undergrab@2': [4]}

        >>> # Test case for undergrab
        >>> score_precision_and_recall_by_hits([1, 2], [1, 2, 3])
        {'precision': 1.0, 'recall': 0.6666666666666666, 'precision@1': 1.0, 'recall@1': 0.3333333333333333, 'overgrab': [], 'undergrab': [3], 'overgrab@1': [], 'undergrab@1': [2, 3]}

        >>> # Test case for overgrab
        >>> score_precision_and_recall_by_hits([1, 2, 3, 4], [1, 2, 3])
        {'precision': 0.75, 'recall': 1.0, 'precision@1': 1.0, 'recall@1': 0.3333333333333333, 'overgrab': [4], 'undergrab': [], 'overgrab@1': [], 'undergrab@1': [2, 3]}

        >>> # Test case for undergrab@2
        >>> score_precision_and_recall_by_hits([1, 2, 4], [1, 2, 3], k=2)
        {'precision': 0.6666666666666666, 'recall': 0.6666666666666666, 'precision@1': 1.0, 'precision@2': 1.0, 'recall@1': 0.3333333333333333, 'recall@2': 0.6666666666666666, 'overgrab': [4], 'undergrab': [3], 'overgrab@1': [], 'overgrab@2': [], 'undergrab@1': [2, 3], 'undergrab@2': [3]}

        >>> # Test case for overgrab@2
        >>> score_precision_and_recall_by_hits([1, 3, 4], [1, 2, 3], k=2)
        {'precision': 0.6666666666666666, 'recall': 0.6666666666666666, 'precision@1': 1.0, 'precision@2': 1.0, 'recall@1': 0.3333333333333333, 'recall@2': 0.6666666666666666, 'overgrab': [4], 'undergrab': [2], 'overgrab@1': [], 'overgrab@2': [], 'undergrab@1': [2, 3], 'undergrab@2': [2]}

        >>> # Test case with keys - undergrab
        >>> score_precision_and_recall_by_hits({'a': 1, 'b': 2}, {'a': 1, 'b': 2, 'c': 3})
        {'precision': 1.0, 'recall': 0.6666666666666666, 'precision@1': 1.0, 'recall@1': 0.3333333333333333, 'overgrab': [], 'undergrab': ['c'], 'overgrab@1': [], 'undergrab@1': ['b', 'c']}

        >>> # Test case with keys - overgrab
        >>> score_precision_and_recall_by_hits({'a': 1, 'b': 2, 'd': 4}, {'a': 1, 'b': 2, 'c': 3})
        {'precision': 0.6666666666666666, 'recall': 0.6666666666666666, 'precision@1': 1.0, 'recall@1': 0.3333333333333333, 'overgrab': ['d'], 'undergrab': ['c'], 'overgrab@1': [], 'undergrab@1': ['b', 'c']}
    """

    # region process arguments

    if not reference:
        # If `reference` is None or empty, then we are unable to compute the hits and metrics.
        # In this case return `None` for both precision and recall.
        return {
            'precision': None,
            'recall': None,
        }

    prediction_labels = reference_labels = None

    if isinstance(reference, Mapping):
        reference_labels = list(reference.keys())
        reference = list(reference.values())
    if not isinstance(reference, List):
        reference = [reference]
    len_reference = len(reference)
    if not reference_labels:
        reference_labels = reference

    if isinstance(prediction, Mapping):
        prediction_labels = list(prediction.keys())
        prediction = list(prediction.values())
    if prediction is not None and not isinstance(prediction, List):
        prediction = [prediction]
    len_prediction = len(prediction)
    if not prediction_labels:
        prediction_labels = prediction

    if not prediction:
        precision_at_k = {f'precision@{_k}': 0 for _k in range(1, k + 1)}

        recall_at_k = {f'recall@{_k}': 0 for _k in range(1, k + 1)}

        return {
            'precision': 0,
            'recall': 0,
            **precision_at_k,
            **recall_at_k,
            'overgrab': [],
            'undergrab': (
                    reference_labels or list(range(len_reference))
            )
        }

    # endregion

    # region compute hits
    hits_prediction = [0] * len_prediction
    hits_reference = [0] * len_reference
    recall_matrix = [[0] * len_reference for _ in range(len_prediction)]
    recall_at_k_matrix = [[0] * len_reference for _ in range(len_prediction)]

    if comparer is None:
        for i, pred in enumerate(prediction):
            for pred_item in iter_(pred):
                for j, ref in enumerate(reference):
                    if pred_item == ref:
                        hits_prediction[i] = 1
                        hits_reference[j] = 1
                        recall_matrix[i][j] = 1
                        recall_at_k_matrix[i][j] = 1
                    elif i != 0 and recall_at_k_matrix[i - 1][j]:
                        recall_at_k_matrix[i][j] = 1
    else:
        for i, pred in enumerate(prediction):
            for pred_item in iter_(pred):
                for j, ref in enumerate(reference):
                    compare_score = comparer(pred_item, ref)
                    if not (0 <= compare_score <= 1):
                        raise ValueError(f"Comparer score must be between 0 and 1; got {compare_score}")
                    hits_prediction[i] = max(compare_score, hits_prediction[i])
                    hits_reference[j] = max(compare_score, hits_reference[j])
                    recall_matrix[i][j] = compare_score
                    if i == 0:
                        recall_at_k_matrix[0][j] = compare_score
                    else:
                        recall_at_k_matrix[i][j] = max(compare_score, recall_at_k_matrix[i - 1][j])
    # endregion

    # region aggregate metrics
    precision = sum(hits_prediction) / len_prediction if prediction else 0
    overgrab = [prediction_labels[i] for i in range(len_prediction) if not hits_prediction[i]]
    recall = sum(hits_reference) / len_reference if reference else 0
    undergrab = [reference_labels[i] for i in range(len_reference) if not hits_reference[i]]

    precision_at_k = {
        f'precision@{_k}': sum(hits_prediction[:_k]) / _k if _k < len_prediction else precision
        for _k in range(1, k + 1)
    }

    recall_at_k = {
        f'recall@{_k}': sum(recall_at_k_matrix[_k - 1]) / len_reference if _k < len_prediction else recall
        for _k in range(1, k + 1)
    }

    undergrab_at_k = {
        f'undergrab@{_k}': (
            [
                reference_labels[i] for i in range(len_reference)
                if recall_at_k_matrix[_k - 1][i] < undergrab_threshold
            ] if _k < len_prediction else undergrab
        )
        for _k in range(1, k + 1)
    }

    overgrab_at_k = {
        f'overgrab@{_k}': (
            [
                prediction_labels[i] for i in range(min(_k, len_prediction))
                if all(recall_matrix[i][j] <= overgrab_threshold for j in range(len_reference))
            ] if _k < len_prediction else overgrab
        )
        for _k in range(1, k + 1)
    }

    # endregion

    return {
        'precision': precision,
        'recall': recall,
        **precision_at_k,
        **recall_at_k,
        'overgrab': overgrab,
        'undergrab': undergrab,
        **overgrab_at_k,
        **undergrab_at_k
    }
