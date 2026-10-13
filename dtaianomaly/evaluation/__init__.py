"""
This module contains functionality to evaluate performance of an
anomaly detector. It can be imported as follows:

>>> from dtaianomaly import evaluation

Custom evaluation metrics can be implemented by extending :py:class:`~dtaianomaly.evaluation.BinaryMetric` or
:py:class:`~dtaianomaly.evaluation.ProbaMetric`. The former expects predicted "decisions" (anomaly or not),
the latter predicted "scores" (more or less anomalous). This distinction is important for later use in
a :py:class:`~dtaianomaly.workflow.Worfklow`.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    from ._range_based_metrics import (
        RangeBasedFBeta,
        RangeBasedPrecision,
        RangeBasedRecall,
    )
    from ._ThresholdMetric import ThresholdMetric
    from ._UCRScore import UCRScore
    from ._volume_under_surface_metrics import (
        RangeAreaUnderPR,
        RangeAreaUnderROC,
        VolumeUnderPR,
        VolumeUnderROC,
    )

_LAZY_IMPORTS = {
    "AffiliationFBeta": "._affiliation_metrics",
    "AffiliationPrecision": "._affiliation_metrics",
    "AffiliationRecall": "._affiliation_metrics",
    "AreaUnderPR": "._area_under_the_curve_metrics",
    "AreaUnderROC": "._area_under_the_curve_metrics",
    "BestThresholdMetric": "._BestThresholdMetric",
    "BinaryMetric": "._BinaryMetric",
    "FBeta": "._classification_based_metrics",
    "Precision": "._classification_based_metrics",
    "Recall": "._classification_based_metrics",
    "EventWiseFBeta": "._event_wise_metrics",
    "EventWisePrecision": "._event_wise_metrics",
    "EventWiseRecall": "._event_wise_metrics",
    "FBetaMixin": "._FBetaMixin",
    "Metric": "._Metric",
    "PointAdjustedFBeta": "._point_adjusted_metrics",
    "PointAdjustedPrecision": "._point_adjusted_metrics",
    "PointAdjustedRecall": "._point_adjusted_metrics",
    "ProbaMetric": "._ProbaMetric",
    "RangeBasedFBeta": "._range_based_metrics",
    "RangeBasedPrecision": "._range_based_metrics",
    "RangeBasedRecall": "._range_based_metrics",
    "ThresholdMetric": "._ThresholdMetric",
    "UCRScore": "._UCRScore",
    "RangeAreaUnderPR": "._volume_under_surface_metrics",
    "RangeAreaUnderROC": "._volume_under_surface_metrics",
    "VolumeUnderPR": "._volume_under_surface_metrics",
    "VolumeUnderROC": "._volume_under_surface_metrics",
}

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


def __getattr__(name):
    # Load public objects on first access (PEP 562), so importing this module
    # does not import heavy dependencies of objects that are never used.
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
