
import numpy as np
from typing import Optional

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision


class MedianMethod(BaseDetector):
    """
    Anomaly detection based on the Two-sided Median Method.

    The Median Method [basu2007automatic]_ computes the deviation of a sample
    compared to its neighborhood. This neighborhood is computed as a window
    around the sample. The deviation is consequently measured as the number
    of standard deviations     the observations deviates from the mean of its
    neighborhood.

    In contrast to the original paper, this implementation allows to define a
    different neighborhood size before and after the sample, to fine tune how
    much lookahead is allowed. In the ultimate case, if ``neighborhood_size_after = 0``,
    then the Median Method is a purely online anomaly detector. Note, however,
    that this case differs from the One-Sided Median Method discussed in the
    original paper, which also uses the first order difference to detect anomalies.

    Parameters
    ----------
    neighborhood_size_before: int
        The number of observations before the sample to include in the neighborhood.
    neighborhood_size_after: int, default=None
        The number of observations after the sample to include in the neighborhood.
        If None, the same value as ``window_size_before`` will be used.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import MedianMethod
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> median_method = MedianMethod(10).fit(x)
    >>> median_method.decision_function(x)
    array([1.1851476 , 0.68191703, 1.05125284, ..., 0.81373386, 1.86097851,
           0.05852008])

    References
    ----------
    .. [basu2007automatic] Basu, Sabyasachi, and Martin Meckesheimer. "Automatic outlier
       detection for time series: an application to sensor data." Knowledge and Information
       Systems 11 (2007), 137-154, doi: `10.1007/s10115-006-0026-6 <https://doi.org/10.1007/s10115-006-0026-6>`_.
    """
    neighborhood_size_before: int
    neighborhood_size_after: Optional[int]

    def __init__(self, neighborhood_size_before: int, neighborhood_size_after: Optional[int] = None):
        super().__init__(Supervision.UNSUPERVISED)

        if not isinstance(neighborhood_size_before, int) or isinstance(neighborhood_size_before, bool):
            raise TypeError("`neighborhood_size_before` should be an integer")
        if neighborhood_size_before < 1:
            raise ValueError("`neighborhood_size_before` should be strictly positive")

        if neighborhood_size_after is not None:
            if not isinstance(neighborhood_size_after, int) or isinstance(neighborhood_size_after, bool):
                raise TypeError("`neighborhood_size_after` should be an integer")
            if neighborhood_size_after < 0:
                raise ValueError("`neighborhood_size_after` can not be negative!")

        self.neighborhood_size_before = neighborhood_size_before
        self.neighborhood_size_after = neighborhood_size_after

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseDetector':
        return self  # No fitting is required.

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")

        # Convert to a numpy array
        X = np.asarray(X, dtype=float).squeeze()

        # Set the neighborhood size after the observation
        if self.neighborhood_size_after is None:
            neighborhood_size_after = self.neighborhood_size_before
        else:
            neighborhood_size_after = self.neighborhood_size_after

        X_padded = np.pad(X, (self.neighborhood_size_before, neighborhood_size_after), constant_values=(np.nan,))
        neighborhoods = np.lib.stride_tricks.sliding_window_view(X_padded, window_shape=(self.neighborhood_size_before + neighborhood_size_after + 1))
        return np.nan_to_num(np.abs(X - np.nanmean(neighborhoods, axis=1)) / np.nanstd(neighborhoods, axis=1), nan=0.0)
