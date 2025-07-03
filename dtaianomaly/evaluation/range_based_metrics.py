import abc
from collections.abc import Callable

import numpy as np

from dtaianomaly.evaluation._common import FBetaBase, make_intervals
from dtaianomaly.evaluation.metrics import BinaryMetric

_IntervalType = tuple[int, int]
_DeltaType = str | Callable[[int, int], float]
_GammaType = str | Callable[[int], float]


def _interval_overlap(a: _IntervalType, b: _IntervalType) -> _IntervalType | None:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return (start, end) if start < end else None


def _omega(
    anomaly_range: _IntervalType,
    overlap_set: _IntervalType | None,
    delta: _DeltaType,
) -> float:
    # Figure 2.a
    if overlap_set is None:
        return 0
    my_value = 0
    max_value = 0
    anomaly_length = anomaly_range[1] - anomaly_range[0]
    for i in range(1, anomaly_length + 1):
        bias = _delta(delta, i, anomaly_length)
        max_value += bias
        if overlap_set[0] <= anomaly_range[0] + i - 1 < overlap_set[1]:
            my_value += bias
    return my_value / max_value


def _delta(delta: _DeltaType, i: int, anomaly_length: int) -> float:
    # Figure 2.b
    if delta == "flat":
        return 1
    elif delta == "front":
        return anomaly_length - i + 1
    elif delta == "back":
        return i
    elif delta == "middle":
        return i if i <= anomaly_length / 2 else anomaly_length - i + 1
    else:  # Custom method
        return delta(i, anomaly_length)


def _gamma(gamma: _GammaType, nb_overlapping_intervals: int) -> float:
    if gamma == "one":
        return 1
    elif gamma == "reciprocal":
        return 1 / nb_overlapping_intervals
    else:  # Custom method
        return gamma(nb_overlapping_intervals)


def _existence_reward(
    interval: _IntervalType, other_intervals: list[_IntervalType]
) -> float:
    # Equation (5)
    for other_interval in other_intervals:
        if _interval_overlap(interval, other_interval) is not None:
            return 1
    return 0


def _overlap_reward(
    interval: _IntervalType,
    other_intervals: list[_IntervalType],
    delta: _DeltaType,
    gamma: _GammaType,
) -> float:
    # Equation (6)
    return _cardinality_factor(interval, other_intervals, gamma) * sum(
        [
            _omega(interval, _interval_overlap(interval, other_interval), delta)
            for other_interval in other_intervals
        ]
    )


def _cardinality_factor(
    interval: _IntervalType, other_intervals: list[_IntervalType], gamma: _GammaType
) -> float:
    # Equation (7)
    nb_overlapping_intervals = 0
    for other_interval in other_intervals:
        if _interval_overlap(interval, other_interval) is not None:
            nb_overlapping_intervals += 1
    return (
        1 if nb_overlapping_intervals <= 1 else _gamma(gamma, nb_overlapping_intervals)
    )


def _precision_interval(
    interval: _IntervalType,
    ground_truth_intervals: list[_IntervalType],
    delta: _DeltaType,
    gamma: _GammaType,
) -> float:
    # Equation (9)
    return _overlap_reward(interval, ground_truth_intervals, delta, gamma)


def _recall_interval(
    interval: _IntervalType,
    predicted_intervals: list[_IntervalType],
    alpha: float,
    delta: _DeltaType,
    gamma: _GammaType,
) -> float:
    # Equation (4)
    return alpha * _existence_reward(interval, predicted_intervals) + (
        1 - alpha
    ) * _overlap_reward(interval, predicted_intervals, delta, gamma)


class RangeBasedMetricBasePrecision(BinaryMetric, abc.ABC):
    delta: _DeltaType
    gamma: _GammaType

    def __init__(self, delta: _DeltaType = "flat", gamma: _GammaType = "reciprocal"):
        if isinstance(delta, str):
            if delta not in ["flat", "front", "back", "middle"]:
                raise ValueError(
                    f"Only predefined `delta` values are ['flat', 'front', 'back', 'middle'], received: '{delta}'"
                )
        elif callable(delta):
            try:
                delta(0, 10)
            except TypeError:
                raise TypeError(
                    "If 'delta' is a custom method, it should be of the form '(int, int) -> float'"
                )
        else:
            raise TypeError(f"`delta` should be a string or a callable")

        if isinstance(gamma, str):
            if gamma not in ["one", "reciprocal"]:
                raise ValueError(
                    f"Only predefined `gamma` values are ['one', 'reciprocal'], received: '{gamma}'"
                )
        elif callable(gamma):
            try:
                gamma(2)
            except TypeError:
                raise TypeError(
                    "If 'gamma' is a custom method, it should be of the form 'int -> float'"
                )
        else:
            raise TypeError(f"`gamma` should be a string or a callable")

        self.delta = delta
        self.gamma = gamma

    def _precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Add 1 to ends, because make_intervals returns closed intervals while the code here assumes half-open intervals
        gt_starts, gt_ends = make_intervals(y_true)
        pred_starts, pred_ends = make_intervals(y_pred)

        ground_truth_intervals = list(zip(gt_starts, gt_ends + 1))
        precision_T = [
            _precision_interval(
                interval, ground_truth_intervals, self.delta, self.gamma
            )
            for interval in zip(pred_starts, pred_ends + 1)
        ]

        return sum(precision_T) / pred_starts.shape[0]


class RangeBasedMetricBasePrecisionRecall(RangeBasedMetricBasePrecision, abc.ABC):
    alpha: float

    def __init__(
        self,
        alpha: float = 0.5,
        delta: _DeltaType = "flat",
        gamma: _GammaType = "reciprocal",
    ):
        super().__init__(delta, gamma)

        if not isinstance(alpha, (float, int)) or isinstance(alpha, bool):
            raise TypeError("`alpha` should be numeric")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("`alpha` should be at least 0 and at most 1")

        self.alpha = alpha

    def _recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Add 1 to ends, because make_intervals returns closed intervals while the code here assumes half-open intervals
        gt_starts, gt_ends = make_intervals(y_true)
        pred_starts, pred_ends = make_intervals(y_pred)

        predicted_intervals = list(zip(pred_starts, pred_ends + 1))
        recall_T = [
            _recall_interval(
                interval, predicted_intervals, self.alpha, self.delta, self.gamma
            )
            for interval in zip(gt_starts, gt_ends + 1)
        ]

        return sum(recall_T) / gt_starts.shape[0]


class RangeBasedPrecision(RangeBasedMetricBasePrecision):
    """
    Computes the range-based precision score :cite:`tatbul2018precision`.

    The range-based precision computes a precision-score for each predicted
    anomalous range and then takes the average over all ranges. This precision-score
    consists of two parts: (1) the amount of overlap between the predicted range
    and the ground truth ranges, and (2) whether the predicted range overlaps with
    only one or multiple ground truth ranges. These components can be computed
    independently, and are multiplied to get a final precision-score for the range.

    Parameters
    ----------
    delta: str or callable, default='flat'
        Bias for the position of the predicted anomaly in the ground truth anomalous
        range. Valid options are:

        - ``'flat'``: Equal bias towards all positions in the ground truth anomalous range.
        - ``'front'``: Predictions that are near the front of the ground truth anomaly (i.e. early detection) have a higher weight.
        - ``'back'``: Predictions that are near the end of the ground truth anomaly (i.e. late detection) have a higher weight.
        - ``'middle'``: Predictions that are near the center of the ground truth anomaly have a higher weight.
        - Callable: A custom function to include positional bias, which takes as input two integers (a position within the anomalous range, and the total length of that range) and returns a float (the weight of that position).

    gamma: str or callable, default='reciprocal'
        Penalization approach for detecting multiple ranges with a single range. Valid options are:

        - ``'one'``: Fragmented detection should not be penalized.
        - ``'reciprocal'``: Weight fragmented detection of :math:´N´ ranges with as single range by a factor of :math:´1/N´.
        - Callable: A custom function to penalize fragmented detection, which takes as input an integer (the number of detected ranges) and returns a float (the penalization factor).
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        return self._precision(y_true, y_pred)


class RangeBasedRecall(RangeBasedMetricBasePrecisionRecall):
    """
    Computes the range-based recall score :cite:`tatbul2018precision`.

    The range-based recall computes a recall-score for each ground truth
    anomalous range and then takes the average over all ranges. This recall-score
    consists of three parts: (1) the amount of overlap between the ground truth range
    and the predicted ranges, (2) whether the ground truth range overlaps with
    only one or multiple predicted ranges, and (3) whether the final ground truth
    range is detected at all. Components (1) and (2) are computed independently
    and multiplied, of which the result is combined with component (3) through
    a convex combination to get a final recall-score for the ground truth range.

    Parameters
    ----------
    alpha: float, default=0.5
        The importance of detecting the events (even if it is only a single detected point)
        compared to detecting a large portion of the ground truth events. Should be at least 0
        and at most 1.

    delta: str or callable, default='flat'
        Bias for the position of the predicted anomaly in the ground truth anomalous
        range. Valid options are:

        - ``'flat'``: Equal bias towards all positions in the ground truth anomalous range.
        - ``'front'``: Predictions that are near the front of the ground truth anomaly (i.e. early detection) have a higher weight.
        - ``'back'``: Predictions that are near the end of the ground truth anomaly (i.e. late detection) have a higher weight.
        - ``'middle'``: Predictions that are near the center of the ground truth anomaly have a higher weight.
        - Callable: A custom function to include positional bias, which takes as input two integers (a position within the anomalous range, and the total length of that range) and returns a float (the weight of that position).

    gamma: str or callable, default='reciprocal'
        Penalization approach for detecting multiple ranges with a single range. Valid options are:

        - ``'one'``: Fragmented detection should not be penalized.
        - ``'reciprocal'``: Weight fragmented detection of :math:´N´ ranges with as single range by a factor of :math:´1/N´.
        - Callable: A custom function to penalize fragmented detection, which takes as input an integer (the number of detected ranges) and returns a float (the penalization factor).
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        return self._recall(y_true, y_pred)


class RangeBasedFBeta(RangeBasedMetricBasePrecisionRecall, FBetaBase):
    """
    Computes the range-based :math:`F_\\beta` score :cite:`tatbul2018precision`.

    The range-based :math:`F_\\beta`-score equals the harmonic mean of the range-based
    precision and range-based recall. The metrics take into account three parts: (1) the
    amount of overlap between the ground truth ranges and the predicted ranges, (2) whether
    there is fragmented detection or not, and (3) whether the ground truth ranges are
    detected at all.

    Parameters
    ----------
    beta: int, float, default=1
        Desired beta parameter.
    alpha: float, default=0.5
        The importance of detecting the events (even if it is only a single detected point)
        compared to detecting a large portion of the ground truth events. Should be at least 0
        and at most 1.

    delta: str or callable, default='flat'
        Bias for the position of the predicted anomaly in the ground truth anomalous
        range. Valid options are:

        - ``'flat'``: Equal bias towards all positions in the ground truth anomalous range.
        - ``'front'``: Predictions that are near the front of the ground truth anomaly (i.e. early detection) have a higher weight.
        - ``'back'``: Predictions that are near the end of the ground truth anomaly (i.e. late detection) have a higher weight.
        - ``'middle'``: Predictions that are near the center of the ground truth anomaly have a higher weight.
        - Callable: A custom function to include positional bias, which takes as input two integers (a position within the anomalous range, and the total length of that range) and returns a float (the weight of that position).

    gamma: str or callable, default='reciprocal'
        Penalization approach for detecting multiple ranges with a single range. Valid options are:

        - ``'one'``: Fragmented detection should not be penalized.
        - ``'reciprocal'``: Weight fragmented detection of :math:´N´ ranges with as single range by a factor of :math:´1/N´.
        - Callable: A custom function to penalize fragmented detection, which takes as input an integer (the number of detected ranges) and returns a float (the penalization factor).

    See also
    --------
    RangeBasedPrecision: Compute the range-based precision score.
    RangeBasedRecall: Compute the range-based recall score.
    """

    def __init__(
        self,
        beta: (float, int) = 1.0,
        alpha: float = 0.5,
        delta: _DeltaType = "flat",
        gamma: _GammaType = "reciprocal",
    ):
        RangeBasedMetricBasePrecisionRecall.__init__(self, alpha, delta, gamma)
        FBetaBase.__init__(self, beta)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        return self._f_score(
            precision=self._precision(y_true, y_pred),
            recall=self._recall(y_true, y_pred),
        )
