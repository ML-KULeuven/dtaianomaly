
"""
This module contains functionality to detect anomalies. It can be imported 
as follows:

>>> from dtaianomaly import anomaly_detection

We refer to the `documentation <https://dtaianomaly.readthedocs.io/en/stable/getting_started/anomaly_detection.html>`_
for more information regarding detecting anomalies using ``dtaianomaly``.
"""

from .BaseDetector import BaseDetector, load_detector
from .windowing_utils import sliding_window, reverse_sliding_window

from .baselines import AlwaysNormal, AlwaysAnomalous, RandomDetector
from .IsolationForest import IsolationForest
from .LocalOutlierFactor import LocalOutlierFactor
from .MatrixProfileDetector import MatrixProfileDetector

__all__ = [
    'BaseDetector',
    'load_detector',
    'sliding_window',
    'reverse_sliding_window',
    'AlwaysNormal',
    'AlwaysAnomalous',
    'RandomDetector',
    'MatrixProfileDetector',
    'IsolationForest',
    'LocalOutlierFactor'
]
