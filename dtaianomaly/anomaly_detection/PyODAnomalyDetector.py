import abc

import numpy as np
from pyod.models.base import BaseDetector as PyODBaseDetector

from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import (
    check_is_valid_window_size,
    compute_window_size,
    reverse_sliding_window,
    sliding_window,
)


class PyODAnomalyDetector(BaseDetector, abc.ABC):
    """
    Abstract class for anomaly detection based on the PyOD library :cite:`zhao2019pyod`.

    PyOD is a Python library for detecting anomalies in multivariate
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
    """

    window_size: int | str
    stride: int
    kwargs: dict
    window_size_: int
    pyod_detector_: PyODBaseDetector

    def __init__(self, window_size: int | str, stride: int = 1, **kwargs):
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
        detector: PyODBaseDetector
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

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)
        self.pyod_detector_ = self._initialize_detector(**self.kwargs)
        self.pyod_detector_.fit(sliding_window(X, self.window_size_, self.stride))

    def _decision_function(self, X: np.ndarray) -> np.array:
        per_window_decision_scores = self.pyod_detector_.decision_function(
            sliding_window(X, self.window_size_, self.stride)
        )
        decision_scores = reverse_sliding_window(
            per_window_decision_scores, self.window_size_, self.stride, X.shape[0]
        )

        return decision_scores
