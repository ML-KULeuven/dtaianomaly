import numpy as np

from dtaianomaly.anomaly_detection._BaseDetector import BaseDetector, Supervision

__all__ = ["AlwaysAnomalous"]


class AlwaysAnomalous(BaseDetector):
    """
    Detector that predicts all instances to be anomalous.

    Baseline anomaly detector, which predicts that all observations are anomalous.
    This detector should only be used for sanity-check, and not to effectively
    detect anomalies in time series data.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import AlwaysAnomalous
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> baseline = AlwaysAnomalous().fit(x)
    >>> baseline.decision_function(x)
    array([1., 1., 1., ..., 1., 1., 1.])
    """

    def __init__(self):
        super().__init__(Supervision.UNSUPERVISED)

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        """Should not do anything."""

    def _decision_function(self, X: np.ndarray) -> np.array:
        return np.ones(shape=X.shape[0])
