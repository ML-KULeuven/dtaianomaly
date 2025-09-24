import abc
from typing import Literal

import numpy as np

from dtaianomaly.evaluation._BinaryMetric import BinaryMetric
from dtaianomaly.evaluation._FBetaMixin import FBetaMixin
from dtaianomaly.type_validation import FloatAttribute, LiteralAttribute
from dtaianomaly.utils import make_intervals

__all__ = ["RangeBasedPrecision", "RangeBasedFBeta", "RangeBasedRecall"]

_IntervalType = tuple[int, int]
_DeltaType = Literal["flat", "front", "back", "middle"]
_GammaType = Literal["one", "reciprocal"]


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
        raise ValueError(f"Invalid value for delta given: '{delta}")


def _gamma(gamma: _GammaType, nb_overlapping_intervals: int) -> float:
    if gamma == "one":
        return 1
    elif gamma == "reciprocal":
        return 1 / nb_overlapping_intervals
    else:  # Custom method
        raise ValueError(f"Invalid value for gamma given: '{gamma}")


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
    attribute_validation = {
        "delta": LiteralAttribute("flat", "front", "back", "middle"),
        "gamma": LiteralAttribute("one", "reciprocal"),
    }

    def __init__(self, delta: _DeltaType = "flat", gamma: _GammaType = "reciprocal"):
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

    attribute_validation = {
        "alpha": FloatAttribute(0.0, 1.0),
    }

    def __init__(
        self,
        alpha: float = 0.5,
        delta: _DeltaType = "flat",
        gamma: _GammaType = "reciprocal",
    ):
        super().__init__(delta, gamma)
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
    delta: str, default='flat'
        Bias for the position of the predicted anomaly in the ground truth anomalous
        range. Valid options are:

        - ``'flat'``: Equal bias towards all positions in the ground truth anomalous range.
        - ``'front'``: Predictions that are near the front of the ground truth anomaly (i.e. early detection) have a higher weight.
        - ``'back'``: Predictions that are near the end of the ground truth anomaly (i.e. late detection) have a higher weight.
        - ``'middle'``: Predictions that are near the center of the ground truth anomaly have a higher weight.

    gamma: str, default='reciprocal'
        Penalization approach for detecting multiple ranges with a single range. Valid options are:

        - ``'one'``: Fragmented detection should not be penalized.
        - ``'reciprocal'``: Weight fragmented detection of :math:´N´ ranges with as single range by a factor of :math:´1/N´.

    See Also
    --------
    RangeBasedPrecision: Compute the range-based precision score.
    RangeBasedRecall: Compute the range-based recall score.

    Examples
    --------
    >>> from dtaianomaly.evaluation import RangeBasedPrecision
    >>> metric = RangeBasedPrecision()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)  # doctest: +ELLIPSIS
    0.333...

    Warnings
    --------
    Note that, while tuning a metric to some domain is beneficial in practical applications,
    this flexibility makes it difficult for a large-scale, general-purpose evaluation of
    multiple anomaly detectors, as you can optimize the metric for a specific application.
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

    delta: str, default='flat'
        Bias for the position of the predicted anomaly in the ground truth anomalous
        range. Valid options are:

        - ``'flat'``: Equal bias towards all positions in the ground truth anomalous range.
        - ``'front'``: Predictions that are near the front of the ground truth anomaly (i.e. early detection) have a higher weight.
        - ``'back'``: Predictions that are near the end of the ground truth anomaly (i.e. late detection) have a higher weight.
        - ``'middle'``: Predictions that are near the center of the ground truth anomaly have a higher weight.

    gamma: str, default='reciprocal'
        Penalization approach for detecting multiple ranges with a single range. Valid options are:

        - ``'one'``: Fragmented detection should not be penalized.
        - ``'reciprocal'``: Weight fragmented detection of :math:´N´ ranges with as single range by a factor of :math:´1/N´.

    See Also
    --------
    RangeBasedPrecision: Compute the range-based precision score.
    RangeBasedFBeta: Compute the range-based :math:`F_\\beta` score.

    Examples
    --------
    >>> from dtaianomaly.evaluation import RangeBasedRecall
    >>> metric = RangeBasedRecall()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)
    1.0

    Warnings
    --------
    Note that, while tuning a metric to some domain is beneficial in practical applications,
    this flexibility makes it difficult for a large-scale, general-purpose evaluation of
    multiple anomaly detectors, as you can optimize the metric for a specific application.
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        return self._recall(y_true, y_pred)


class RangeBasedFBeta(RangeBasedMetricBasePrecisionRecall, FBetaMixin):
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

    delta: str, default='flat'
        Bias for the position of the predicted anomaly in the ground truth anomalous
        range. Valid options are:

        - ``'flat'``: Equal bias towards all positions in the ground truth anomalous range.
        - ``'front'``: Predictions that are near the front of the ground truth anomaly (i.e. early detection) have a higher weight.
        - ``'back'``: Predictions that are near the end of the ground truth anomaly (i.e. late detection) have a higher weight.
        - ``'middle'``: Predictions that are near the center of the ground truth anomaly have a higher weight.

    gamma: str, default='reciprocal'
        Penalization approach for detecting multiple ranges with a single range. Valid options are:

        - ``'one'``: Fragmented detection should not be penalized.
        - ``'reciprocal'``: Weight fragmented detection of :math:´N´ ranges with as single range by a factor of :math:´1/N´.

    See Also
    --------
    RangeBasedPrecision: Compute the range-based precision score.
    RangeBasedRecall: Compute the range-based recall score.

    Examples
    --------
    >>> from dtaianomaly.evaluation import RangeBasedFBeta
    >>> metric = RangeBasedFBeta()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)
    0.5

    Warnings
    --------
    Note that, while tuning a metric to some domain is beneficial in practical applications,
    this flexibility makes it difficult for a large-scale, general-purpose evaluation of
    multiple anomaly detectors, as you can optimize the metric for a specific application.
    """

    def __init__(
        self,
        beta: int | float = 1.0,
        alpha: float = 0.5,
        delta: _DeltaType = "flat",
        gamma: _GammaType = "reciprocal",
    ):
        RangeBasedMetricBasePrecisionRecall.__init__(self, alpha, delta, gamma)
        FBetaMixin.__init__(self, beta)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        return self._f_score(
            precision=self._precision(y_true, y_pred),
            recall=self._recall(y_true, y_pred),
        )
