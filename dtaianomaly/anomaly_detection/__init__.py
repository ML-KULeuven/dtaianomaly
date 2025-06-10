"""
This module contains functionality to detect anomalies. It can be imported
as follows:

>>> from dtaianomaly import anomaly_detection

We refer to the `documentation <https://dtaianomaly.readthedocs.io/en/stable/getting_started/anomaly_detection.html>`_
for more information regarding detecting anomalies using ``dtaianomaly``.
"""

from .AutoEncoder import AutoEncoder
from .BaseDetector import BaseDetector, Supervision, load_detector
from .baselines import AlwaysAnomalous, AlwaysNormal, RandomDetector
from .ClusterBasedLocalOutlierFactor import ClusterBasedLocalOutlierFactor
from .CopulaBasedOutlierDetector import CopulaBasedOutlierDetector
from .DWT_MLEAD import DWT_MLEAD
from .HistogramBasedOutlierScore import HistogramBasedOutlierScore
from .IsolationForest import IsolationForest
from .KernelPrincipalComponentAnalysis import KernelPrincipalComponentAnalysis
from .KMeansAnomalyDetector import KMeansAnomalyDetector
from .KNearestNeighbors import KNearestNeighbors
from .KShapeAnomalyDetector import KShapeAnomalyDetector
from .LocalOutlierFactor import LocalOutlierFactor
from .MatrixProfileDetector import MatrixProfileDetector
from .MedianMethod import MedianMethod
from .MultivariateDetector import MultivariateDetector
from .NeuralBaseDetector import NeuralBaseDetector
from .OneClassSupportVectorMachine import OneClassSupportVectorMachine
from .PrincipalComponentAnalysis import PrincipalComponentAnalysis
from .PyODAnomalyDetector import PyODAnomalyDetector
from .RobustPrincipalComponentAnalysis import RobustPrincipalComponentAnalysis
from .windowing_utils import (
    check_is_valid_window_size,
    compute_window_size,
    reverse_sliding_window,
    sliding_window,
)

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
    "KShapeAnomalyDetector",
    "LocalOutlierFactor",
    "MatrixProfileDetector",
    "MedianMethod",
    "OneClassSupportVectorMachine",
    "PrincipalComponentAnalysis",
    "PyODAnomalyDetector",
    "RobustPrincipalComponentAnalysis",
    "MultivariateDetector",
    "DWT_MLEAD",
    "NeuralBaseDetector",
    "AutoEncoder",
]
