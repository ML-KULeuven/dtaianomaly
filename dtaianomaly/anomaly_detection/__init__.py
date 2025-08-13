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
from .BaseNeuralDetector import BaseNeuralDetector
from .BaseNeuralDetector_utils import (
    BaseNeuralForecastingDetector,
    BaseNeuralReconstructionDetector,
    ForecastDataset,
    ReconstructionDataset,
    TimeSeriesDataset,
)
from .ClusterBasedLocalOutlierFactor import ClusterBasedLocalOutlierFactor
from .ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from .CopulaBasedOutlierDetector import CopulaBasedOutlierDetector
from .DWT_MLEAD import DWT_MLEAD
from .HistogramBasedOutlierScore import HistogramBasedOutlierScore
from .IsolationForest import IsolationForest
from .KernelPrincipalComponentAnalysis import KernelPrincipalComponentAnalysis
from .KMeansAnomalyDetector import KMeansAnomalyDetector
from .KNearestNeighbors import KNearestNeighbors
from .KShapeAnomalyDetector import KShapeAnomalyDetector
from .LocalOutlierFactor import LocalOutlierFactor
from .LocalPolynomialApproximation import LocalPolynomialApproximation
from .LongShortTermMemoryNetwork import LongShortTermMemoryNetwork
from .MatrixProfileDetector import MatrixProfileDetector
from .MedianMethod import MedianMethod
from .MultilayerPerceptron import MultilayerPerceptron
from .MultivariateDetector import MultivariateDetector
from .OneClassSupportVectorMachine import OneClassSupportVectorMachine
from .PrincipalComponentAnalysis import PrincipalComponentAnalysis
from .PyODAnomalyDetector import PyODAnomalyDetector
from .RobustPrincipalComponentAnalysis import RobustPrincipalComponentAnalysis
from .Transformer import Transformer
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
    "BaseNeuralDetector",
    "AutoEncoder",
    "MultilayerPerceptron",
    "ForecastDataset",
    "ReconstructionDataset",
    "TimeSeriesDataset",
    "BaseNeuralForecastingDetector",
    "BaseNeuralReconstructionDetector",
    "LongShortTermMemoryNetwork",
    "ConvolutionalNeuralNetwork",
    "Transformer",
    "LocalPolynomialApproximation",
]
