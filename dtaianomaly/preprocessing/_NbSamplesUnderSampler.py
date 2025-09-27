import numpy as np

from dtaianomaly.preprocessing._Preprocessor import Preprocessor
from dtaianomaly.type_validation import IntegerAttribute

__all__ = ["NbSamplesUnderSampler"]


class NbSamplesUnderSampler(Preprocessor):
    """
    Undersample time series to a given number of samples.

    Sample exactly :py:attr:`~dtaianomaly.preprocessing.NbSamplesUnderSampler.nb_samples`
    element from the time series, such that each sample in the processed
    time series was equidistant in the original time series. This enables
    to manually set the size of the transformed time series independent
    of the original size of the time series.

    Parameters
    ----------
    nb_samples : int, default=None
        The number of samples remaining.

    Examples
    --------
    >>> from dtaianomaly.preprocessing import NbSamplesUnderSampler
    >>> from dtaianomaly.data import demonstration_time_series
    >>> X, y = demonstration_time_series()
    >>> preprocessor = NbSamplesUnderSampler(nb_samples=512)
    >>> X_, y_ = preprocessor.fit_transform(X, y)
    """

    nb_samples: int
    attribute_validation = {
        "nb_samples": IntegerAttribute(minimum=2),
    }

    def __init__(self, nb_samples: int) -> None:
        self.nb_samples = nb_samples

    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "NbSamplesUnderSampler":
        return self

    def _transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        if self.nb_samples >= X.shape[0]:
            return X, y
        indices = np.linspace(
            0, X.shape[0] - 1, self.nb_samples, dtype=int, endpoint=True
        )
        return X[indices], (None if y is None else y[indices])
