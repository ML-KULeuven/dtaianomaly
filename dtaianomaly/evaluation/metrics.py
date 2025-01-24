import abc

import numpy as np

from dtaianomaly import utils
from dtaianomaly.PrettyPrintable import PrettyPrintable
from dtaianomaly.thresholding import Thresholding


class Metric(PrettyPrintable):

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the performance score.

        Parameters
        ----------
        y_true: array-like of shape (n_samples)
            Ground-truth labels.
        y_pred: array-like of shape (n_samples)
            Predicted anomaly scores.

        Returns
        -------
        score: float
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
        if not utils.is_valid_array_like(y_true):
            raise ValueError("Input 'y_true' should be numeric array-like")
        if not utils.is_valid_array_like(y_pred):
            raise ValueError("Input 'y_pred' should be numeric array-like")
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if not y_true.shape == y_pred.shape:
            raise ValueError("Inputs should have identical shape")
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("The predicted anomaly scores must be binary!")
        return self._compute(y_true, y_pred)

    @abc.abstractmethod
    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Effectively compute the metric."""


class ProbaMetric(Metric, abc.ABC):
    """A metric that takes as input continuous anomaly scores."""


class BinaryMetric(Metric, abc.ABC):
    """A metric that takes as input binary anomaly labels."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the performance score.

        Parameters
        ----------
        y_true: array-like of shape (n_samples)
            Ground-truth labels.
        y_pred: array-like of shape (n_samples)
            Predicted anomaly scores.

        Returns
        -------
        score: float
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


class ThresholdMetric(ProbaMetric):
    """
    Wrapper to combine a `BinaryMetric` object with some
    thresholding, to make sure that it can take continuous
    anomaly scores as an input. This is done by first applying
    some thresholding to the predicted anomaly scores, after
    which a binary metric can be computed.

    Parameters
    ----------
    thresholder: Thresholding
        Instance of the desired `Thresholding` class
    metric: Metric
        Instance of the desired `Metric` class
    """

    thresholder: Thresholding
    metric: BinaryMetric

    def __init__(self, thresholder: Thresholding, metric: BinaryMetric) -> None:
        if not isinstance(thresholder, Thresholding):
            raise TypeError(
                f"thresholder expects 'Thresholding', got {type(thresholder)}"
            )
        if not isinstance(metric, BinaryMetric):
            raise TypeError(f"metric expects 'BinaryMetric', got {type(metric)}")
        super().__init__()
        self.thresholder = thresholder
        self.metric = metric

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred_binary = self.thresholder.threshold(y_pred)
        # Can compute the inner method, because checks have already been done at this point.
        return self.metric._compute(y_true=y_true, y_pred=y_pred_binary)

    def __str__(self) -> str:
        return f"{self.thresholder}->{self.metric}"
