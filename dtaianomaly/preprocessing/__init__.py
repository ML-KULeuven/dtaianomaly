"""
This module contains preprocessing functionality.

>>> from dtaianomaly import preprocessing

Custom preprocessors can be implemented by extending the base :py:class:`~dtaianomaly.preprocessing.Preprocessor` class.
"""
from .Preprocessor import Preprocessor, check_preprocessing_inputs, Identity
from .ChainedPreprocessor import ChainedPreprocessor
from .MinMaxScaler import MinMaxScaler
from .StandardScaler import StandardScaler
from .MovingAverage import MovingAverage
from .ExponentialMovingAverage import ExponentialMovingAverage
from .UnderSampler import SamplingRateUnderSampler, NbSamplesUnderSampler
from .Differencing import Differencing
from .PiecewiseAggregateApproximation import PiecewiseAggregateApproximation
from .RobustScaler import RobustScaler

__all__ = [
    'Preprocessor',
    'check_preprocessing_inputs',
    'Identity',
    'ChainedPreprocessor',
    'MinMaxScaler',
    'StandardScaler',
    'MovingAverage',
    'ExponentialMovingAverage',
    'SamplingRateUnderSampler',
    'NbSamplesUnderSampler',
    'Differencing',
    'PiecewiseAggregateApproximation',
    'RobustScaler'
]
