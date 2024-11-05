
import numpy as np
from typing import Optional, Union
from sklearn.neighbors import LocalOutlierFactor as SklearnLocalOutlierFactor
from sklearn.exceptions import NotFittedError

from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector
from dtaianomaly.anomaly_detection.windowing_utils import sliding_window, reverse_sliding_window, check_is_valid_window_size, compute_window_size
from dtaianomaly import utils


class LocalOutlierFactor(BaseDetector):
    """
    Anomaly detector based on the Local Outlier Factor.

    The local outlier factor [Breunig2000LOF]_ compares the density of each
    sample to the density of the neighboring samples. If the neighbors of a
    sample have a much higher density that the sample itself, the sample is
    considered anomalous. By looking at the local density (i.e., only comparing
    with the neighbors of a sample), the local outlier factor takes into
    account varying densities across the sample space.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    **kwargs:
        Arguments to be passed to scikit-learns local outlier factor

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    detector_ : SklearnLocalOutlierFactor
        A LOF-detector of Sklearn. Only available upon fitting

    Notes
    -----
    This is a wrapper for scikit-learn's `LocalOutlierFactor
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html>`.
    The constructor allows additional keyword arguments that will be passed
    to the underlying scikit-learn Local Outlier Factor model.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import LocalOutlierFactor
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> local_outlier_factor = LocalOutlierFactor(10).fit(x)
    >>> local_outlier_factor.decision_function(x)
    array([0.98505735, 0.9894939 , 0.99214303, ..., 1.02445672, 1.02723816,
           1.01699908])

    References
    ----------
    .. [Breunig2000LOF] Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander.
       2000. LOF: identifying density-based local outliers. In Proceedings of the 2000 ACM
       SIGMOD international conference on Management of data (SIGMOD '00). Association for
       Computing Machinery, New York, NY, USA, 93–104. doi: `10.1145/342009.335388 <https://doi.org/10.1145/342009.335388>`_
    """
    window_size: Union[int, str]
    stride: int
    kwargs: dict
    window_size_: int
    detector_: SklearnLocalOutlierFactor

    def __init__(self, window_size: Union[str, int], stride: int = 1, **kwargs) -> None:
        super().__init__()

        check_is_valid_window_size(window_size)

        if not isinstance(stride, int) or isinstance(stride, bool):
            raise TypeError("`stride` should be an integer")
        if stride < 1:
            raise ValueError("`stride` should be strictly positive")

        self.window_size = window_size
        self.stride = stride
        self.kwargs = kwargs
        SklearnLocalOutlierFactor(**kwargs)  # Try initialization to check the parameters

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'LocalOutlierFactor':
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
        self: LocalOutlierFactor
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
        self.detector_ = SklearnLocalOutlierFactor(**self.kwargs)
        if 'novelty' in self.kwargs and self.kwargs['novelty']:
            # Fitting is only useful for novelty detection
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
            Local density scores. Higher is more anomalous.

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
        windows = sliding_window(X, self.window_size_, self.stride)
        if 'novelty' in self.kwargs and self.kwargs['novelty']:
            # If novelty detection, then no fitting on the given data is necessary
            per_window_anomaly_scores = -self.detector_.decision_function(windows)
        else:
            # Otherwise, look within the given data to compute the local densities
            per_window_anomaly_scores = -self.detector_.fit(windows).negative_outlier_factor_
        anomaly_scores = reverse_sliding_window(per_window_anomaly_scores, self.window_size_, self.stride, X.shape[0])

        return anomaly_scores
