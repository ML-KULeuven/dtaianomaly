
import abc
import numpy as np
from typing import Optional, Union
from sklearn.exceptions import NotFittedError
from pyod.models.base import BaseDetector as PyODBaseDetector
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import sliding_window, reverse_sliding_window, check_is_valid_window_size, compute_window_size
from dtaianomaly import utils


class PyODAnomalyDetector(BaseDetector, abc.ABC):
    """
    Abstract class for anomaly detection based on the PyOD library.

    PyOD [zhao2019pyod]_ is a Python library for detecting anomalies in multivariate
    data. The anomaly detectors in PyOD typically deal with tabular data, which assumes
    i.i.d (independent and identically distributed) data. This is generally not the
    case for time series data, which has a temporal dependency. Nevertheless, the detectors
    of PyOD can be used for detecting anomalies in time series data.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    **kwargs:
        Arguments to be passed to pyod anomaly detector

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    pyod_detector_ : SklearnLocalOutlierFactor
        The PyOD anomaly detector

    References
    ----------
    .. [zhao2019pyod] Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox
       for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96),
       pp.1-7.
    """
    window_size: Union[int, str]
    stride: int
    kwargs: dict
    window_size_: int
    pyod_detector_: PyODBaseDetector

    def __init__(self, window_size: Union[str, int], stride: int = 1, **kwargs):
        super().__init__(self._supervision())

        check_is_valid_window_size(window_size)

        if not isinstance(stride, int) or isinstance(stride, bool):
            raise TypeError("`stride` should be an integer")
        if stride < 1:
            raise ValueError("`stride` should be strictly positive")

        self.window_size = window_size
        self.stride = stride
        self.kwargs = kwargs

        # Check if the PyOD detector can be correctly initialized
        self._initialize_detector(**self.kwargs)

    @abc.abstractmethod
    def _initialize_detector(self, **kwargs) -> PyODBaseDetector:
        """
        Initialize the PyOD anomaly detector.

        Parameters
        ----------
        kwargs:
            The hyperparameters to be passed to the PyOD anomaly detector.

        Returns
        -------
        A PyOD anomaly detector with the given hyperparameters.
        """

    @abc.abstractmethod
    def _supervision(self) -> Supervision:
        """
        Return the supervision of this anomaly detector.

        Returns
        -------
        supervision: Supervision
            The supervision of this PyOD anomaly detector.
        """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'BaseDetector':
        """
        Fit this PyOD anomaly detector on the given data.

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
        self: PyODAnomalyDetector
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
        self.pyod_detector_ = self._initialize_detector(**self.kwargs)
        self.pyod_detector_.fit(sliding_window(X, self.window_size_, self.stride))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision scores.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.

        Returns
        -------
        decision_scores: array-like of shape (n_samples)
            The decision scores of the anomaly detector. Higher indicates more anomalous.

        Raises
        ------
        ValueError
            If `X` is not a valid array.
        NotFittedError
            If this method is called before fitting the anomaly detector.
        """
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")
        if not hasattr(self, 'pyod_detector_') or not hasattr(self, 'window_size_'):
            raise NotFittedError('Call the fit function before making predictions!')

        X = np.asarray(X)
        per_window_decision_scores = self.pyod_detector_.decision_function(sliding_window(X, self.window_size_, self.stride))
        decision_scores = reverse_sliding_window(per_window_decision_scores, self.window_size_, self.stride, X.shape[0])

        return decision_scores
