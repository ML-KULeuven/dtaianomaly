from typing import Optional

import numpy as np

from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision


class AlwaysNormal(BaseDetector):
    """
    Baseline anomaly detector, which predicts that all observations are normal.
    This detector should only be used for sanity-check, and not to effectively
    detect anomalies in time series data.
    """

    def __init__(self):
        super().__init__(Supervision.UNSUPERVISED)

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """Should not do anything."""

    def _decision_function(self, X: np.ndarray) -> np.array:
        return np.zeros(shape=X.shape[0])


class AlwaysAnomalous(BaseDetector):
    """
    Baseline anomaly detector, which predicts that all observations are anomalous.
    This detector should only be used for sanity-check, and not to effectively
    detect anomalies in time series data.
    """

    def __init__(self):
        super().__init__(Supervision.UNSUPERVISED)

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """Should not do anything."""

    def _decision_function(self, X: np.ndarray) -> np.array:
        return np.ones(shape=X.shape[0])


class RandomDetector(BaseDetector):
    """
    Baseline anomaly detector, which assigns random anomaly scores. This detector
    should only be used for sanity-check, and not to effectively detect anomalies
    in time series data.

    Parameters
    ----------
    seed: int, default=None
        The seed to use for generating anomaly scores. If None, no seed will be used.
    """

    seed: Optional[int]

    def __init__(self, seed: Optional[int] = None):
        super().__init__(Supervision.UNSUPERVISED)
        self.seed = seed

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """Should not do anything."""

    def _decision_function(self, X: np.ndarray) -> np.array:
        return np.random.default_rng(seed=self.seed).random(size=X.shape[0])
