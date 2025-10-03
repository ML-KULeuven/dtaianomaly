
Anomaly detection module
========================

.. automodule:: dtaianomaly.anomaly_detection

Base Objects
------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDetector
    BaseNeuralDetector
    BaseNeuralForecastingDetector
    BaseNeuralReconstructionDetector
    BasePyODAnomalyDetector
    Supervision
    MultivariateDetector

Utility functions
-----------------

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    load_detector


Statistical methods
-------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClusterBasedLocalOutlierFactor
    CopulaBasedOutlierDetector
    HistogramBasedOutlierScore
    IsolationForest
    KernelPrincipalComponentAnalysis
    KMeansAnomalyDetector
    KNearestNeighbors
    LocalOutlierFactor
    OneClassSupportVectorMachine
    PrincipalComponentAnalysis
    RobustPrincipalComponentAnalysis

Time series statistical methods
-------------------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DWT_MLEAD
    KShapeAnomalyDetector
    LocalPolynomialApproximation
    MatrixProfileDetector
    MedianMethod
    RobustRandomCutForestAnomalyDetector
    ROCKAD
    SpectralResidual

Neural methods
--------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoEncoder
    ConvolutionalNeuralNetwork
    HybridKNearestNeighbors
    LongShortTermMemoryNetwork
    MultilayerPerceptron
    Transformer

Time series foundation models
-----------------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Chronos
    MOMENT
    TimeMoE


Baselines
---------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AlwaysNormal
    AlwaysAnomalous
    MovingWindowVariance
    RandomDetector
    SquaredDifference
