import abc

import numpy as np

from dtaianomaly.type_validation import AttributeValidationMixin
from dtaianomaly.utils import PrintConstructionCallMixin, is_valid_array_like

__all__ = ["Thresholding"]


class Thresholding(PrintConstructionCallMixin, AttributeValidationMixin):
    """
    Base thresholding class.

    A base class for converting continuous anomaly scores to discrete anomaly
    labels. This is useful for actually detecting anomalies, rather than just
    knowing how anomalous an instance is realtive to the other instances.
    """

    def threshold(self, scores: np.ndarray) -> np.ndarray:
        """
        Threshold the given anomaly scores.

        Apply the thresholding operation to the given anomaly scores. This function
        will perform the necessary checks and formatting on the anomaly scores,
        before effectively applying the thresholding.

        Parameters
        ----------
        scores : array-like of shape (n_samples)
            The continuous anomaly scores to convert to binary anomaly labels.

        Returns
        -------
        array-like of shape (n_samples)
            The discrete anomaly labels, in which a 0 indicates normal and a
            1 indicates anomalous.

        Raises
        ------
        ValueError
            If `scores` is not a valid array
        ValueError
            If `scores` is not one-dimensional. If all dimensions but one have a
            size of 1, then no error will be thrown.
        """
        if not is_valid_array_like(scores):
            raise ValueError("Input must be numerical array-like")
        scores_ = np.asarray(scores).squeeze()
        if len(scores_.shape) > 1:
            raise ValueError("The anomaly scores must be one dimensional!")
        return self._threshold(scores_).astype(int)

    @abc.abstractmethod
    def _threshold(self, scores: np.ndarray) -> np.ndarray:
        """Effectively threshold the given scores, without checking the inputs."""
