"""
This module contains functionality to evaluate performance of an
anomaly detector. It can be imported as follows:

>>> from dtaianomaly import evaluation

Custom evaluation metrics can be implemented by extending :py:class:`~dtaianomaly.evaluation.Metric` or
:py:class:`~dtaianomaly.evaluation.ProbaMetric`. The former expects predicted "decisions" (anomaly or not),
the latter predicted "scores" (more or less anomalous). This distinction is important for later use in
a :py:class:`~dtaianomaly.workflow.Worfklow`.
"""

from .metrics import Metric, BinaryMetric, ProbaMetric, ThresholdMetric
from .simple_binary_metrics import Precision, Recall, FBeta
from .simple_proba_metrics import AreaUnderROC, AreaUnderPR
from .point_adjusted_binary_metrics import (
    PointAdjustedPrecision,
    PointAdjustedRecall,
    PointAdjustedFBeta,
)
from .BestThresholdMetric import BestThresholdMetric

__all__ = [
    "Metric",
    "BinaryMetric",
    "ProbaMetric",
    "ThresholdMetric",
    "Precision",
    "Recall",
    "FBeta",
    "AreaUnderPR",
    "AreaUnderROC",
    "PointAdjustedPrecision",
    "PointAdjustedRecall",
    "PointAdjustedFBeta",
    "BestThresholdMetric",
]
