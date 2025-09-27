"""
Anomaly detectors typically predict continous anomaly scores: *How anomalous is the observation?*
In many applications, a hard decision is required: *Is the observation an anomaly?* The continuous
scores can be converted to discrete scores through thresholding. This module contains different
methods to threshold the scores, and can be imported as follows:

>>> from dtaianomaly import thresholding

Custom thresholders can be implemented by extending the base :py:class:`dtaianomaly.thresholding.Thresholding` class.
"""

from ._ContaminationRateThreshold import ContaminationRateThreshold
from ._FixedCutoffThreshold import FixedCutoffThreshold
from ._Thresholding import Thresholding
from ._TopNThreshold import TopNThreshold

__all__ = [
    "Thresholding",
    "FixedCutoffThreshold",
    "ContaminationRateThreshold",
    "TopNThreshold",
]
