from typing import Dict, List, Union, TypeVar

from dtaianomaly.evaluation import Metric, ProbaMetric, BinaryMetric, ThresholdMetric
from dtaianomaly.thresholding import Thresholding
from dtaianomaly.preprocessing import Preprocessor
from dtaianomaly.anomaly_detection import BaseDetector
from dtaianomaly.pipeline import EvaluationPipeline


T = TypeVar("T")


def build_pipelines(
    preprocessors: List[Preprocessor],
    detectors: List[BaseDetector],
    metrics: List[ProbaMetric],
) -> List[EvaluationPipeline]:
    """The given lists are assumed to be non-empty."""
    return [
        EvaluationPipeline(
            preprocessor=preprocessor, detector=detector, metrics=metrics
        )
        for preprocessor in preprocessors
        for detector in detectors
    ]


def convert_to_proba_metrics(
    metrics: List[Metric], thresholds: List[Thresholding]
) -> List[ProbaMetric]:
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


def convert_to_list(value: Union[T, List[T]]) -> List[T]:
    """If a list is given, it is assumed to be non-empty."""
    if not isinstance(value, list):
        return [
            value,
        ]
    return value
