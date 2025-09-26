import numpy as np

from dtaianomaly.anomaly_detection._BaseDetector import BaseDetector, Supervision

__all__ = ["RandomDetector"]


class RandomDetector(BaseDetector):
    """
    Assign random anomaly scores to the time series observations.

    Baseline anomaly detector, which assigns random anomaly scores. This detector
    should only be used for sanity-check, and not to effectively detect anomalies
    in time series data.

    Parameters
    ----------
    seed: int, default=None
        The seed to use for generating anomaly scores. If None, no seed will be used.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import RandomDetector
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> baseline = RandomDetector(seed=0).fit(x)
    >>> baseline.decision_function(x)
    array([0.63696169, 0.26978671, 0.04097352, ..., 0.70724404, 0.90315986,
           0.8944909 ])
    """

    seed: int | None

    def __init__(self, seed: int = None):
        super().__init__(Supervision.UNSUPERVISED)
        self.seed = seed

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        """Should not do anything."""

    def _decision_function(self, X: np.ndarray) -> np.array:
        return np.random.default_rng(seed=self.seed).random(size=X.shape[0])
