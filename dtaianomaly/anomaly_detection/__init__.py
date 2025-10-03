"""
This module contains functionality to detect anomalies. It can be imported
as follows:

>>> from dtaianomaly import anomaly_detection

We refer to the `documentation <https://dtaianomaly.readthedocs.io/en/stable/getting_started/anomaly_detection.html>`_
for more information regarding detecting anomalies using ``dtaianomaly``.

API cheatsheet
--------------

Below there is a quick overview of the most essential methods to
detect anomalies:

#. :py:meth:`~dtaianomaly.anomaly_detection.BaseDetector.fit`. Fit the anomaly
   detector. This method requires both an ``X`` (the time series) and ``y``
   (anomaly labels) parameter. However, in practice, most anomaly detectors
   will not use the ground truth labels. The parameter ``y`` is present for
   API consistency and is not required.

#. :py:meth:`~dtaianomaly.anomaly_detection.BaseDetector.decision_function`.
   Compute the decision scores of an observation being an anomaly for a given
   time series ``X``. Returns an array with an entry for each observation in
   the time series. Note that this score is not normalized, and depends on
   the specific anomaly detector. However, for all detectors, a higher score
   means `more anomalous`.

#. :py:meth:`~dtaianomaly.anomaly_detection.BaseDetector.predict_proba`. Compute
   the probability of an anomaly being an anomaly. This is similar to the
   :py:meth:`~dtaianomaly.anomaly_detection.BaseDetector.decision_function`
   method, but the computed scores are normalized to the interval :math:`[0, 1]`,
   which enables the interpretation as a probability.

   .. note::
      The output of a ``predict_proba`` is often a matrix of size ``(n_samples, n_classes)``.
      For anomaly detection, this would lead to a matrix with two columns, one columns
      for the normal probabilities and one column for the anomalous probabilities.
      However, in ``dtaianomaly``, the :py:meth:`~dtaianomaly.anomaly_detection.BaseDetector.predict_proba`
      only returns the probability of a sample being anomalous, because this is
      the probability of interest in many anomaly detection applications.
"""

from ._AutoEncoder import AutoEncoder
from ._BaseDetector import BaseDetector, Supervision, load_detector
from ._BaseNeuralDetector import BaseNeuralDetector
from ._BaseNeuralForecastingDetector import BaseNeuralForecastingDetector
from ._BaseNeuralReconstructionDetector import BaseNeuralReconstructionDetector
from ._BasePyODAnomalyDetector import BasePyODAnomalyDetector
from ._Chronos import Chronos
from ._ClusterBasedLocalOutlierFactor import ClusterBasedLocalOutlierFactor
from ._ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from ._CopulaBasedOutlierDetector import CopulaBasedOutlierDetector
from ._DWT_MLEAD import DWT_MLEAD
from ._HistogramBasedOutlierScore import HistogramBasedOutlierScore
from ._HybridKNearestNeighbors import HybridKNearestNeighbors
from ._IsolationForest import IsolationForest
from ._KernelPrincipalComponentAnalysis import KernelPrincipalComponentAnalysis
from ._KMeansAnomalyDetector import KMeansAnomalyDetector
from ._KNearestNeighbors import KNearestNeighbors
from ._KShapeAnomalyDetector import KShapeAnomalyDetector
from ._LocalOutlierFactor import LocalOutlierFactor
from ._LocalPolynomialApproximation import LocalPolynomialApproximation
from ._LongShortTermMemoryNetwork import LongShortTermMemoryNetwork
from ._MatrixProfileDetector import MatrixProfileDetector
from ._MedianMethod import MedianMethod
from ._MOMENT import MOMENT
from ._MultilayerPerceptron import MultilayerPerceptron
from ._MultivariateDetector import MultivariateDetector
from ._OneClassSupportVectorMachine import OneClassSupportVectorMachine
from ._PrincipalComponentAnalysis import PrincipalComponentAnalysis
from ._RobustPrincipalComponentAnalysis import RobustPrincipalComponentAnalysis
from ._RobustRandomCutForestAnomalyDetector import RobustRandomCutForestAnomalyDetector
from ._SpectralResidual import SpectralResidual
from ._TimeMoE import TimeMoE
from ._TorchTimeSeriesDataSet import (
    ForecastDataset,
    ReconstructionDataset,
    TimeSeriesDataset,
)
from ._Transformer import Transformer
from .baselines import AlwaysAnomalous, AlwaysNormal, RandomDetector

__all__ = [
    # Base
    "BaseDetector",
    "Supervision",
    "load_detector",
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
    "BasePyODAnomalyDetector",
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
    "Chronos",
    "SpectralResidual",
    "MOMENT",
    "TimeMoE",
    "RobustRandomCutForestAnomalyDetector",
    "HybridKNearestNeighbors",
]
