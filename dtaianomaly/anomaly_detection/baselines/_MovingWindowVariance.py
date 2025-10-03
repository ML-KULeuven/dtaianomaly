import numpy as np

from dtaianomaly import utils
from dtaianomaly.anomaly_detection._BaseDetector import BaseDetector, Supervision
from dtaianomaly.type_validation import WindowSizeAttribute
from dtaianomaly.windowing import (
    WINDOW_SIZE_TYPE,
    compute_window_size,
    reverse_sliding_window,
    sliding_window,
)

__all__ = ["MovingWindowVariance"]


class MovingWindowVariance(BaseDetector):
    """
    Detect anomalies based on the variance within a window.

    Baseline anomaly detector, assigns an anomaly score purely based on the
    variance within a sliding window. This detector does not look at any
    recurring patterns within the data. Formally, the anomaly score :math:`s_i`
    for a window :math:`T_{i,i+w-1}` equals  :math:`s_i = var(T_{i,i+w-1})`.

    Parameters
    ----------
    window_size : int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.

    Attributes
    ----------
    window_size_ : int
        The effectively used window size for this anomaly detector.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import MovingWindowVariance
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> baseline = MovingWindowVariance(16).fit(x)
    >>> baseline.decision_function(x)
    array([0.06820711, 0.07130246, 0.07286874, ..., 0.01125165, 0.00984333,
           0.00986772])
    """

    window_size: WINDOW_SIZE_TYPE
    window_size_: int
    attribute_validation = {"window_size": WindowSizeAttribute()}

    def __init__(self, window_size: WINDOW_SIZE_TYPE):
        super().__init__(Supervision.UNSUPERVISED)
        self.window_size = window_size

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)

    def _decision_function(self, X: np.ndarray) -> np.array:
        # Check if the given dataset is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")

        variances = np.var(sliding_window(X, self.window_size_, 1), axis=1)
        return reverse_sliding_window(variances, self.window_size_, 1, X.shape[0])
