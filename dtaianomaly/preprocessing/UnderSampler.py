import numpy as np
from typing import Optional, Tuple

from dtaianomaly.preprocessing.Preprocessor import Preprocessor


class SamplingRateUnderSampler(Preprocessor):
    """
    Undersample time series with sampling rate `sampling_rate`. This means
    that every `sampling_rate` element is taken from the time series. After
    undersampling, only `1/sampling_rate` percent of the original samples
    will remain.

    Parameters
    ----------
    sampling_rate: int
        The rate at which should be sampled.
    """

    sampling_rate: int

    def __init__(self, sampling_rate: int) -> None:
        if sampling_rate <= 0:
            raise ValueError("Sampling rate should be strictly positive.")
        self.sampling_rate = sampling_rate

    def _fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "SamplingRateUnderSampler":
        return self

    def _transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.sampling_rate >= X.shape[0]:
            raise ValueError(
                f"The sampling rate ('{self.sampling_rate}') is too large for a time series of shape {X.shape}!"
            )
        return X[:: self.sampling_rate], (
            None if y is None else y[:: self.sampling_rate]
        )


class NbSamplesUnderSampler(Preprocessor):
    """
    Undersample time series such that exactly `nb_samples` samples
    remain in the original time series. This enables to manually
    set the size of the transformed time series, independent of
    the original size of the time series.

    Parameters
    ----------
    nb_samples: int, default=None
        The number of samples remaining.
    """

    nb_samples: int

    def __init__(self, nb_samples: int) -> None:
        if nb_samples <= 1:
            raise ValueError("Number of samples should be at least 2.")
        self.nb_samples = nb_samples

    def _fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "NbSamplesUnderSampler":
        return self

    def _transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.nb_samples >= X.shape[0]:
            return X, y
        indices = np.linspace(
            0, X.shape[0] - 1, self.nb_samples, dtype=int, endpoint=True
        )
        return X[indices], (None if y is None else y[indices])
