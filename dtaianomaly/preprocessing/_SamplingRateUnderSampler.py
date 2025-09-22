import numpy as np

from dtaianomaly.preprocessing._Preprocessor import Preprocessor
from dtaianomaly.type_validation import IntegerAttribute

__all__ = ["SamplingRateUnderSampler"]


class SamplingRateUnderSampler(Preprocessor):
    """
    Undersample time series with a given sampling rate.

    Sample every :py:attr:`~dtaianomaly.preprocessing.SamplingRateUnderSampler.sampling_rate`
    element from the time series. After undersampling, only
    `1/:py:attr:`~dtaianomaly.preprocessing.SamplingRateUnderSampler.sampling_rate``
    percent of the original samples remain.

    Parameters
    ----------
    sampling_rate: int
        The rate at which should be sampled.

    Examples
    --------
    >>> from dtaianomaly.preprocessing import SamplingRateUnderSampler
    >>> from dtaianomaly.data import demonstration_time_series
    >>> X, y = demonstration_time_series()
    >>> preprocessor = SamplingRateUnderSampler(sampling_rate=16)
    >>> X_, y_ = preprocessor.fit_transform(X, y)
    """

    sampling_rate: int

    attribute_validation = {
        "sampling_rate": IntegerAttribute(minimum=1),
    }

    def __init__(self, sampling_rate: int) -> None:
        self.sampling_rate = sampling_rate

    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "SamplingRateUnderSampler":
        return self

    def _transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        if self.sampling_rate >= X.shape[0]:
            raise ValueError(
                f"The sampling rate ('{self.sampling_rate}') is too large for a time series of shape {X.shape}!"
            )
        return X[:: self.sampling_rate], (
            None if y is None else y[:: self.sampling_rate]
        )
