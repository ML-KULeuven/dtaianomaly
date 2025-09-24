"""
This module contains functionality to evaluate performance of an
anomaly detector. It can be imported as follows:

>>> from dtaianomaly import evaluation

Custom evaluation metrics can be implemented by extending :py:class:`~dtaianomaly.evaluation.BinaryMetric` or
:py:class:`~dtaianomaly.evaluation.ProbaMetric`. The former expects predicted "decisions" (anomaly or not),
the latter predicted "scores" (more or less anomalous). This distinction is important for later use in
a :py:class:`~dtaianomaly.workflow.Worfklow`.
"""

from ._affiliation_metrics import (
    AffiliationFBeta,
    AffiliationPrecision,
    AffiliationRecall,
)
from ._area_under_the_curve_metrics import AreaUnderPR, AreaUnderROC
from ._BestThresholdMetric import BestThresholdMetric
from ._BinaryMetric import BinaryMetric
from ._classification_based_metrics import FBeta, Precision, Recall
from ._event_wise_metrics import EventWiseFBeta, EventWisePrecision, EventWiseRecall
from ._FBetaMixin import FBetaMixin
from ._Metric import Metric
from ._point_adjusted_metrics import (
    PointAdjustedFBeta,
    PointAdjustedPrecision,
    PointAdjustedRecall,
)
from ._ProbaMetric import ProbaMetric
from ._range_based_metrics import RangeBasedFBeta, RangeBasedPrecision, RangeBasedRecall
from ._ThresholdMetric import ThresholdMetric
from ._UCRScore import UCRScore
from ._volume_under_surface_metrics import (
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
    "EventWisePrecision",
    "EventWiseRecall",
    "EventWiseFBeta",
    "AffiliationPrecision",
    "AffiliationRecall",
    "AffiliationFBeta",
    "RangeBasedPrecision",
    "RangeBasedRecall",
    "RangeBasedFBeta",
    "UCRScore",
    "FBetaMixin",
]
