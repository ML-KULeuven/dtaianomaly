import numpy as np

from dtaianomaly.thresholding._Thresholding import Thresholding
from dtaianomaly.type_validation import IntegerAttribute

__all__ = ["TopNThreshold"]


class TopNThreshold(Thresholding):
    """
    Thresholding based on a top N strategy.

    The top `n` anomaly scores are considered anomalous (1),
    Other (lower) scores are considered normal (0).

    Parameters
    ----------
    n: int
        The number of instances that should be flagged as an anomaly

    Examples
    --------
    >>> from dtaianomaly.thresholding import TopNThreshold
    >>> thresholder = TopNThreshold(3)
    >>> thresholder.threshold([0.1, 0.2, 0.3, 0.6, 0.8, 0.5, 0.3, 0.3])
    array([0, 0, 0, 1, 1, 1, 0, 0])
    """

    n: int

    attribute_validation = {"n": IntegerAttribute(minimum=0)}

    def __init__(self, n: int):
        self.n = n

    def _threshold(self, scores: np.ndarray):
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
            If the number of given anomaly scores is smaller than :py:attr:`~dtaianomaly.thresholding.TopNThreshold.n`.
        """
        if self.n > scores.shape[0]:
            raise ValueError(
                f"There are only {scores.shape[0]} anomaly scores given, but {self.n} observations should be anomalous!"
            )

        cutoff = np.partition(scores, -self.n)[-self.n]
        return np.asarray(cutoff <= scores, dtype=np.int8)
