"""
This module contains functionality to evaluate performance of an
anomaly detector. It can be imported as follows:

>>> from dtaianomaly import evaluation

Custom evaluation metrics can be implemented by extending :py:class:`~dtaianomaly.evaluation.BinaryMetric` or
:py:class:`~dtaianomaly.evaluation.ProbaMetric`. The former expects predicted "decisions" (anomaly or not),
the latter predicted "scores" (more or less anomalous). This distinction is important for later use in
a :py:class:`~dtaianomaly.workflow.Worfklow`.
"""

from .BestThresholdMetric import BestThresholdMetric
from .metrics import BinaryMetric, Metric, ProbaMetric, ThresholdMetric
from .point_adjusted_binary_metrics import (
    PointAdjustedFBeta,
    PointAdjustedPrecision,
    PointAdjustedRecall,
)
from .simple_binary_metrics import FBeta, Precision, Recall
from .simple_proba_metrics import AreaUnderPR, AreaUnderROC
from .volume_under_surface import (
    RangeAreaUnderPR,
    RangeAreaUnderROC,
    VolumeUnderPR,
    VolumeUnderROC,
)

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
    "RangeAreaUnderROC",
    "RangeAreaUnderPR",
    "VolumeUnderROC",
    "VolumeUnderPR",
]
