"""
This module contains functionality to detect anomalies. It can be imported 
as follows:

>>> from dtaianomaly import anomaly_detection

We refer to the `documentation <https://dtaianomaly.readthedocs.io/en/stable/getting_started/anomaly_detection.html>`_
for more information regarding detecting anomalies using ``dtaianomaly``.
"""

from .BaseDetector import BaseDetector, Supervision, load_detector
from .windowing_utils import (
    sliding_window,
    reverse_sliding_window,
    check_is_valid_window_size,
    compute_window_size,
)

from .baselines import AlwaysNormal, AlwaysAnomalous, RandomDetector
from .PyODAnomalyDetector import PyODAnomalyDetector
from .ClusterBasedLocalOutlierFactor import ClusterBasedLocalOutlierFactor
from .CopulaBasedOutlierDetector import CopulaBasedOutlierDetector
from .HistogramBasedOutlierScore import HistogramBasedOutlierScore
from .IsolationForest import IsolationForest
from .KernelPrincipalComponentAnalysis import KernelPrincipalComponentAnalysis
from .KMeansAnomalyDetector import KMeansAnomalyDetector
from .KNearestNeighbors import KNearestNeighbors
from .LocalOutlierFactor import LocalOutlierFactor
from .MatrixProfileDetector import MatrixProfileDetector
from .MedianMethod import MedianMethod
from .OneClassSupportVectorMachine import OneClassSupportVectorMachine
from .PrincipalComponentAnalysis import PrincipalComponentAnalysis
from .RobustPrincipalComponentAnalysis import RobustPrincipalComponentAnalysis

__all__ = [
    # Base
    "BaseDetector",
    "Supervision",
    "load_detector",
    # Sliding window
    "sliding_window",
    "reverse_sliding_window",
    "check_is_valid_window_size",
    "compute_window_size",
    # Baselines
    "AlwaysNormal",
    "AlwaysAnomalous",
    "RandomDetector",
    # Detectors
    "ClusterBasedLocalOutlierFactor",
    "CopulaBasedOutlierDetector",
    "HistogramBasedOutlierScore",
    "IsolationForest",
    "KernelPrincipalComponentAnalysis",
    "KMeansAnomalyDetector",
    "KNearestNeighbors",
    "LocalOutlierFactor",
    "MatrixProfileDetector",
    "MedianMethod",
    "OneClassSupportVectorMachine",
    "PrincipalComponentAnalysis",
    "PyODAnomalyDetector",
    "RobustPrincipalComponentAnalysis",
]
