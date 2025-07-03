from typing import TypeVar

from dtaianomaly.evaluation import BinaryMetric, Metric, ProbaMetric, ThresholdMetric
from dtaianomaly.thresholding import Thresholding

T = TypeVar("T")


def convert_to_proba_metrics(
    metrics: list[Metric], thresholds: list[Thresholding]
) -> list[ProbaMetric]:
    """The given lists are assumed to be non-empty."""
    proba_metrics = []
    for metric in metrics:
        if isinstance(metric, BinaryMetric):
            proba_metrics.extend(
                ThresholdMetric(thresholder=threshold, metric=metric)
                for threshold in thresholds
            )
        elif isinstance(metric, ProbaMetric):
            proba_metrics.append(metric)
    return proba_metrics


def convert_to_list(value: T | list[T]) -> list[T]:
    """If a list is given, it is assumed to be non-empty."""
    if not isinstance(value, list):
        return [
            value,
        ]
    return value
