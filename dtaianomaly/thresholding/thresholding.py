import abc

import numpy as np

from dtaianomaly.utils import PrintConstructionCallMixin, is_valid_array_like


class Thresholding(PrintConstructionCallMixin):

    @abc.abstractmethod
    def threshold(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply the thresholding operation to the given anomaly scores

        Parameters
        ----------
        scores: array-like of shape (n_samples)
            The continuous anomaly scores to convert to binary anomaly labels.

        Returns
        -------
        anomaly_labels: array-like of shape (n_samples)
            The discrete anomaly labels, in which a 0 indicates normal and a
            1 indicates anomalous.
        """


class FixedCutoff(Thresholding):
    """
    Thresholding based on a fixed cut-off.

    Values higher than the cut-off are considered anomalous (1),
    values below the cut-off are considered normal (0).

    Parameters
    ----------
    cutoff: float
        The cutoff above which the given anomaly scores indicate an anomaly.
    """

    cutoff: float

    def __init__(self, cutoff: float):
        if not isinstance(cutoff, float):
            raise TypeError("Input must be a float")
        super().__init__()
        self.cutoff = cutoff

    def threshold(self, scores: np.ndarray):
        """
        Apply the cut-off thresholding.

        Parameters
        ----------
        scores: array-like (n_samples)
            Raw anomaly scores

        Returns
        -------
        anomaly_labels: array-like of shape (n_samples)
            Integer array of 1s and 0s, representing anomalous samples
            and normal samples respectively

        Raises
        ------
        ValueError
            If `scores` is not a valid array
        """
        if not is_valid_array_like(scores):
            raise ValueError("Input must be numerical array-like")

        scores = np.asarray(scores)
        return np.asarray(self.cutoff <= scores, dtype=np.int8)


class ContaminationRate(Thresholding):
    """
    Thresholding based on a contamination rate.

    The top `contamination_rate` proportion of anomaly scores are considered anomalous (1),
    Other (lower) scores are considered normal (0).

    Parameters
    ----------
    contamination_rate: float
        The contamination_rate, i.e., the percentage of instances
        that are anomalous.
    """

    contamination_rate: float

    def __init__(self, contamination_rate: float):
        if not isinstance(contamination_rate, float):
            raise TypeError("Rate should be a float")
        if contamination_rate < 0.0 or 1.0 < contamination_rate:
            raise ValueError(
                f"Rate should be between 0 and 1. Received {contamination_rate}"
            )
        self.contamination_rate = contamination_rate

    def threshold(self, scores: np.ndarray):
        """
        Apply the contamination-rate thresholding.

        Parameters
        ----------
        scores: array-like (n_samples)
            Raw anomaly scores

        Returns
        -------
        anomaly_labels: array-like of shape (n_samples)
            Integer array of 1s and 0s, representing anomalous samples
            and normal samples respectively

        Raises
        ------
        ValueError
            If `scores` is not a valid array
        """
        if not is_valid_array_like(scores):
            raise ValueError("Input must be numerical array-like")

        scores = np.asarray(scores)
        cutoff = np.quantile(scores, 1.0 - self.contamination_rate)
        return np.asarray(cutoff <= scores, dtype=np.int8)


class TopN(Thresholding):
    """
    Thresholding based on a top N strategy.

    The top `n` anomaly scores are considered anomalous (1),
    Other (lower) scores are considered normal (0).

    Parameters
    ----------
    n: int
        The number of instances that should be flagged as an anomaly
    """

    n: int

    def __init__(self, n: int):
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("Input should be an integer")
        if n < 0:
            raise ValueError(f"Expecting non-negative input. Received {n}")
        super().__init__()
        self.n = n

    def threshold(self, scores: np.ndarray):
        """
        Apply the top-N thresholding.

        Parameters
        ----------
        scores: array-like (n_samples)
            Raw anomaly scores

        Returns
        -------
        anomaly_labels: array-like of shape (n_samples)
            Integer array of 1s and 0s, representing anomalous samples
            and normal samples respectively

        Raises
        ------
        ValueError
            If `scores` is not a valid array
        """
        if not is_valid_array_like(scores):
            raise ValueError("Input must be numerical array-like")
        if self.n > scores.shape[0]:
            raise ValueError(
                f"There are only {scores.shape[0]} anomaly scores given, but {self.n} should be anomalous!"
            )

        scores = np.asarray(scores)
        cutoff = np.partition(scores, -self.n)[-self.n]
        return np.asarray(cutoff <= scores, dtype=np.int8)
