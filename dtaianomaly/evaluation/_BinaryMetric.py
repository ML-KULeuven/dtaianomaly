import abc

import numpy as np

from dtaianomaly.evaluation._Metric import Metric

__all__ = ["BinaryMetric"]


class BinaryMetric(Metric, abc.ABC):
    """
    A metric that takes as input binary anomaly labels.

    A class to indicate that a metric is binary, i.e., it takes
    as input binary decision labels. This class also checks
    whether the given anomaly scores are actually binary.
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
        ValueError
            If `y_pred` is non-binary.
        """
        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("The predicted anomaly scores must be binary!")
        return super().compute(y_true, y_pred)
