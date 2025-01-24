"""
This module contains thresholding functionality. It can be imported as follows:

>>> from dtaianomaly import thresholding

Thresholding is required to convert raw anomaly scores from a detector,
obtained via the :py:meth:`dtaianomaly.anomaly_detection.BaseDetector.decision_function`,
to binary predictions (anomaly or not).

Custom thresholders can be implemented by extending the base :py:class:`dtaianomaly.thresholding.Thresholding` class.
"""

from .thresholding import Thresholding, FixedCutoff, ContaminationRate, TopN

__all__ = ["Thresholding", "FixedCutoff", "ContaminationRate", "TopN"]
