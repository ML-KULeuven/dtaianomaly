import abc

import numpy as np

from dtaianomaly.type_validation import AttributeValidationMixin
from dtaianomaly.utils import PrintConstructionCallMixin, is_valid_array_like

__all__ = ["Metric"]


class Metric(PrintConstructionCallMixin, AttributeValidationMixin):
    """
    Base class for metrics.

    Class to implement evaluation metrics for anomaly detectors. These
    compute how close the predicted anomalies are to the ground truth
    anomalies.
    """

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Compute the performance score.

        Evaluate how closely the given anomaly scores align to the ground
        truth anomaly scores.

        Parameters
        ----------
        y_true : array-like of shape (n_samples)
            Ground-truth labels.
        y_pred : array-like of shape (n_samples)
            Predicted anomaly scores.
        **kwargs
            Additional arguments used for computing the evaluation metric.

        Returns
        -------
        float
            The alignment score of the given ground truth and
            prediction, according to this score.

        Raises
        ------
        ValueError
            When inputs are not numeric "array-like"s
        ValueError
            If shapes of `y_true` and `y_pred` are not of identical shape
        ValueError
            If `y_true` is non-binary.
        """
        if not is_valid_array_like(y_true):
            raise ValueError("Input 'y_true' should be numeric array-like")
        if not is_valid_array_like(y_pred):
            raise ValueError("Input 'y_pred' should be numeric array-like")
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if not y_true.shape == y_pred.shape:
            raise ValueError("Inputs should have identical shape")
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("The predicted anomaly scores must be binary!")
        return self._compute(y_true, y_pred, **kwargs)

    @abc.abstractmethod
    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Effectively compute the metric."""
