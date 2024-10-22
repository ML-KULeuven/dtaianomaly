
import numpy as np
from typing import Optional
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.exceptions import NotFittedError

from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector
from dtaianomaly.anomaly_detection.windowing_utils import sliding_window, reverse_sliding_window
from dtaianomaly import utils


class IsolationForest(BaseDetector):
    """
    Anomaly detector based on the Isolation Forest algorithm.

    The isolation forest [Liu2008isolation]_ generates random binary trees to
    split the data. If an instance requires fewer splits to isolate it from
    the other data, it is nearer to the root of the tree, and consequently
    receives a higher anomaly score.

    Parameters
    ----------
    window_size: int
        The window size to use for extracting sliding windows from the time series.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    **kwargs
        Arguments to be passed to scikit-learns isolation forest.

    Attributes
    ----------
    detector_ : SklearnIsolationForest
        An Isolation Forest detector of Sklearn. Only available upon fitting

    Notes
    -----
    This is a wrapper for scikit-learn's Isolation Forest
    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>`
    The constructor allows additional keyword arguments that will be passed
    to the underlying scikit-learn Isolation Forest.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import IsolationForest
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> isolation_forest = IsolationForest(10).fit(x)
    >>> isolation_forest.decision_function(x)
    array([0.47552756, 0.48587594, 0.49067661, ..., 0.45292726, 0.45644108,
           0.45439481])

    References
    ----------
    .. [Liu2008isolation] F. T. Liu, K. M. Ting and Z. -H. Zhou, "Isolation Forest,"
       2008 Eighth IEEE International Conference on Data Mining, Pisa, Italy, 2008,
       pp. 413-422, doi: `10.1109/ICDM.2008.17 <https://doi.org/10.1109/ICDM.2008.17>`_.
    """
    window_size: int
    stride: int
    kwargs: dict
    detector_: SklearnIsolationForest

    def __init__(self, window_size: int, stride: int = 1, **kwargs):
        super().__init__()

        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise TypeError("`window_size` should be an integer")
        if window_size < 1:
            raise ValueError("`window_size` should be strictly positive")

        if not isinstance(stride, int) or isinstance(stride, bool):
            raise TypeError("`stride` should be an integer")
        if stride < 1:
            raise ValueError("`stride` should be strictly positive")

        self.window_size = window_size
        self.stride = stride
        self.kwargs = kwargs
        SklearnIsolationForest(**kwargs)  # Try initialization to check the parameters

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'IsolationForest':
        """
        Fit this detector to the given data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.
        y: ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self: IsolationForest
            Returns the instance itself

        Raises
        ------
        ValueError
            If `X` is not a valid array.
        """
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")

        self.detector_ = SklearnIsolationForest(**self.kwargs)

        X = np.asarray(X)
        windows = sliding_window(X, self.window_size, self.stride)
        self.detector_.fit(windows)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores. If the detector has not been fitted prior to calling this function,
        it will be fitted on the input `X`.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.

        Returns
        -------
        anomaly_scores: array-like of shape (n_samples)
            Isolation Forest scores. Higher is more anomalous.

        Raises
        ------
        ValueError
            If `X` is not a valid array.
        NotFittedError
            If this method is called before fitting the anomaly detector.
        """
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")
        if not hasattr(self, 'detector_'):
            raise NotFittedError('Call the fit function before making predictions!')

        X = np.asarray(X)
        windows = sliding_window(X, self.window_size, self.stride)
        per_window_anomaly_scores = -self.detector_.score_samples(windows)
        anomaly_scores = reverse_sliding_window(per_window_anomaly_scores, self.window_size, self.stride, X.shape[0])

        return anomaly_scores
