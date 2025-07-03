"""
This module contains functionality to combine preprocessing and anomaly detection
 in a single wrapped object.

>>> from dtaianomaly import pipeline

Users are not expected to extend the base Pipeline objects as they are
wrappers of underlying ``dtaianomaly`` objects. Custom functionality is
better achieved by implementing the :py:class:`dtaianomaly.preprocessing.Preprocessor`,
:py:class:`dtaianomaly.anomaly_detection.BaseDetector` or
:py:class:`dtaianomaly.evaluation.Metric` objects.
"""

from .Pipeline import Pipeline

__all__ = ["Pipeline"]
