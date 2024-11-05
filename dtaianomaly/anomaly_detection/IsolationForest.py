
import numpy as np
from typing import Optional, Union
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.exceptions import NotFittedError

from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector
from dtaianomaly.anomaly_detection.windowing_utils import sliding_window, reverse_sliding_window, check_is_valid_window_size, compute_window_size
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
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    **kwargs
        Arguments to be passed to scikit-learns isolation forest.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
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
    window_size: Union[int, str]
    stride: int
    kwargs: dict
    window_size_: int
    detector_: SklearnIsolationForest

    def __init__(self, window_size: Union[int, str], stride: int = 1, **kwargs):
        super().__init__()

        check_is_valid_window_size(window_size)

        if not isinstance(stride, int) or isinstance(stride, bool):
            raise TypeError("`stride` should be an integer")
        if stride < 1:
            raise ValueError("`stride` should be strictly positive")

        self.window_size = window_size
        self.stride = stride
        self.kwargs = kwargs
        SklearnIsolationForest(**kwargs)  # Try initialization to check the parameters

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'IsolationForest':
        """
        Fit this detector to the given data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.
        y: ignored
            Not used, present for API consistency by convention.
        kwargs:
            Additional parameters to be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.

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

        X = np.asarray(X)
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)
        self.detector_ = SklearnIsolationForest(**self.kwargs)
        self.detector_.fit(sliding_window(X, self.window_size_, self.stride))

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
        if not hasattr(self, 'detector_') or not hasattr(self, 'window_size_'):
            raise NotFittedError('Call the fit function before making predictions!')

        X = np.asarray(X)
        per_window_anomaly_scores = -self.detector_.score_samples(sliding_window(X, self.window_size_, self.stride))
        anomaly_scores = reverse_sliding_window(per_window_anomaly_scores, self.window_size_, self.stride, X.shape[0])

        return anomaly_scores
