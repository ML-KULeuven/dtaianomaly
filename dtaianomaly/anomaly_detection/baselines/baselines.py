
import numpy as np
from typing import Optional
from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision


class AlwaysNormal(BaseDetector):
    """
    Baseline anomaly detector, which predicts that all observations are normal.
    This detector should only be used for sanity-check, and not to effectively
    detect anomalies in time series data.
    """

    def __init__(self):
        super().__init__(Supervision.UNSUPERVISED)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AlwaysNormal':
        """
        Simply return this detector, because no fitting is required.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.
        y: ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self: AlwaysNormal
            Returns the instance itself
        """
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts 'always normal' anomaly scores, i.e., always returns a '0.0'.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.

        Returns
        -------
        normal_scores: array-like of shape (n_samples)
            All normal anomaly scores.

        Raises
        ------
        ValueError
            If `X` is not a valid array.
        """
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")
        return np.zeros(shape=np.asarray(X).shape[0])


class AlwaysAnomalous(BaseDetector):
    """
    Baseline anomaly detector, which predicts that all observations are anomalous.
    This detector should only be used for sanity-check, and not to effectively
    detect anomalies in time series data.
    """

    def __init__(self):
        super().__init__(Supervision.UNSUPERVISED)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AlwaysAnomalous':
        """
        Simply return this detector, because no fitting is required.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.
        y: ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self: AlwaysAnomalous
            Returns the instance itself
        """
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts 'always anomalous' anomaly scores, i.e., always returns a '1.0'.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.

        Returns
        -------
        anomalous_scores: array-like of shape (n_samples)
            All anomalous anomaly scores.

        Raises
        ------
        ValueError
            If `X` is not a valid array.
        """
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")
        return np.ones(shape=np.asarray(X).shape[0])


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

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'RandomDetector':
        """
        Simply return this detector, because no fitting is required.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.
        y: ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self: RandomDetector
            Returns the instance itself
        """
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts random anomaly scores. Uses numpy random-number generator, without
        adjusting the internal seed of numpy.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.

        Returns
        -------
        random_scores: array-like of shape (n_samples)
            Random anomaly scores.

        Raises
        ------
        ValueError
            If `X` is not a valid array.
        """
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")
        rng = np.random.default_rng(seed=self.seed)
        return rng.random(size=np.asarray(X).shape[0])
