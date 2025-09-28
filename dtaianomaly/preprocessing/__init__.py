"""
This module contains preprocessing functionality, and can be imported as follows:

>>> from dtaianomaly import preprocessing
"""

from ._ChainedPreprocessor import ChainedPreprocessor
from ._Differencing import Differencing
from ._ExponentialMovingAverage import ExponentialMovingAverage
from ._Identity import Identity
from ._MinMaxScaler import MinMaxScaler
from ._MovingAverage import MovingAverage
from ._NbSamplesUnderSampler import NbSamplesUnderSampler
from ._PiecewiseAggregateApproximation import PiecewiseAggregateApproximation
from ._Preprocessor import Preprocessor
from ._RobustScaler import RobustScaler
from ._SamplingRateUnderSampler import SamplingRateUnderSampler
from ._StandardScaler import StandardScaler

__all__ = [
    "Preprocessor",
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
