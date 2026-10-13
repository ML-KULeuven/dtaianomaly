import numpy as np

from dtaianomaly.thresholding._Thresholding import Thresholding
from dtaianomaly.type_validation import FloatAttribute

__all__ = ["ContaminationRateThreshold"]


class ContaminationRateThreshold(Thresholding):
    """
    Thresholding based on a contamination rate.

    The top `contamination_rate` proportion of anomaly scores are considered anomalous (1),
    Other (lower) scores are considered normal (0).

    Parameters
    ----------
    contamination_rate : float
        The contamination_rate, i.e., the percentage of instances
        that are anomalous. A rate of 0 flags no instance as anomalous.

    Examples
    --------
    >>> from dtaianomaly.thresholding import ContaminationRateThreshold
    >>> thresholder = ContaminationRateThreshold(0.25)
    >>> thresholder.threshold([0.1, 0.2, 0.3, 0.6, 0.8, 0.5, 0.3, 0.3])
    array([0, 0, 0, 1, 1, 0, 0, 0])
    """

    contamination_rate: float
    attribute_validation = {
        "contamination_rate": FloatAttribute(minimum=0.0, maximum=1.0)
    }

    def __init__(self, contamination_rate: float):
        self.contamination_rate = contamination_rate

    def _threshold(self, scores: np.ndarray):
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
        """
        if self.contamination_rate == 0.0:
            return np.zeros(scores.shape[0], dtype=np.int8)
        cutoff = np.quantile(scores, 1.0 - self.contamination_rate)
        return np.asarray(cutoff <= scores, dtype=np.int8)
