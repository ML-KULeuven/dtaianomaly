import numpy as np

from dtaianomaly.thresholding._Thresholding import Thresholding
from dtaianomaly.type_validation import FloatAttribute

__all__ = ["FixedCutoffThreshold"]


class FixedCutoffThreshold(Thresholding):
    """
    Thresholding based on a fixed cut-off.

    Values higher than the cut-off are considered anomalous (1),
    values below the cut-off are considered normal (0).

    Parameters
    ----------
    cutoff: float
        The cutoff above which the given anomaly scores indicate an anomaly.

    Examples
    --------
    >>> from dtaianomaly.thresholding import FixedCutoffThreshold
    >>> thresholder = FixedCutoffThreshold(0.7)
    >>> thresholder.threshold([0.1, 0.2, 0.3, 0.6, 0.8, 0.5, 0.3, 0.3])
    array([0, 0, 0, 0, 1, 0, 0, 0])
    """

    cutoff: float

    attribute_validation = {"cutoff": FloatAttribute()}

    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def _threshold(self, scores: np.ndarray):
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
        """
        return np.asarray(self.cutoff <= scores)
