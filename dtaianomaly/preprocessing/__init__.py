"""
This module contains preprocessing functionality.

>>> from dtaianomaly import preprocessing

Custom preprocessors can be implemented by extending the base :py:class:`~dtaianomaly.preprocessing.Preprocessor` class.
"""
from .Preprocessor import Preprocessor, check_preprocessing_inputs, Identity
from .ChainedPreprocessor import ChainedPreprocessor
from .MinMaxScaler import MinMaxScaler
from .ZNormalizer import ZNormalizer
from .MovingAverage import MovingAverage
from .ExponentialMovingAverage import ExponentialMovingAverage
from .UnderSampler import SamplingRateUnderSampler, NbSamplesUnderSampler
from .Differencing import Differencing

__all__ = [
    'Preprocessor',
    'check_preprocessing_inputs',
    'Identity',
    'ChainedPreprocessor',
    'MinMaxScaler',
    'ZNormalizer',
    'MovingAverage',
    'ExponentialMovingAverage',
    'SamplingRateUnderSampler',
    'NbSamplesUnderSampler',
    'Differencing'
]
