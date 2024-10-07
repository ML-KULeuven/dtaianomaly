
import abc
import numpy as np

from dtaianomaly.thresholding import Thresholding
from dtaianomaly import utils
from dtaianomaly.PrettyPrintable import PrettyPrintable


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
        """
        if not utils.is_valid_array_like(y_true):
            raise ValueError("Input 'y_true' should be numeric array-like")
        if not utils.is_valid_array_like(y_pred):
            raise ValueError("Input 'y_pred' should be numeric array-like")
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if not y_true.shape == y_pred.shape:
            raise ValueError("Inputs should have identical shape")
        return self._compute(y_true, y_pred)

    @abc.abstractmethod
    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Effectively compute the metric. """


class ProbaMetric(Metric, abc.ABC):
    """ A metric that takes as input continuous anomaly scores. """


class BinaryMetric(Metric, abc.ABC):
    """ A metric that takes as input binary anomaly labels. """


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
    binary_metric: BinaryMetric

    def __init__(self, thresholder: Thresholding, metric: BinaryMetric) -> None:
        if not isinstance(thresholder, Thresholding):
            raise TypeError(f"thresholder expects 'Thresholding', got {type(thresholder)}")
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
        return f'{self.thresholder}->{self.metric}'