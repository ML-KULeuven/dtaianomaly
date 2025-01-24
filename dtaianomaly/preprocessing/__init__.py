"""
This module contains preprocessing functionality.

>>> from dtaianomaly import preprocessing

Custom preprocessors can be implemented by extending the base :py:class:`~dtaianomaly.preprocessing.Preprocessor` class.
"""

from .ChainedPreprocessor import ChainedPreprocessor
from .Differencing import Differencing
from .ExponentialMovingAverage import ExponentialMovingAverage
from .MinMaxScaler import MinMaxScaler
from .MovingAverage import MovingAverage
from .PiecewiseAggregateApproximation import PiecewiseAggregateApproximation
from .Preprocessor import Identity, Preprocessor, check_preprocessing_inputs
from .RobustScaler import RobustScaler
from .StandardScaler import StandardScaler
from .UnderSampler import NbSamplesUnderSampler, SamplingRateUnderSampler

__all__ = [
    "Preprocessor",
    "check_preprocessing_inputs",
    "Identity",
    "ChainedPreprocessor",
    "MinMaxScaler",
    "StandardScaler",
    "MovingAverage",
    "ExponentialMovingAverage",
    "SamplingRateUnderSampler",
    "NbSamplesUnderSampler",
    "Differencing",
    "PiecewiseAggregateApproximation",
    "RobustScaler",
]
