import numpy as np

from dtaianomaly.evaluation._BinaryMetric import BinaryMetric
from dtaianomaly.evaluation._ProbaMetric import ProbaMetric
from dtaianomaly.thresholding import Thresholding
from dtaianomaly.type_validation import ObjectAttribute

__all__ = ["ThresholdMetric"]


class ThresholdMetric(ProbaMetric):
    """
    Apply a :py:class:`~dtaianomaly.threshold.Thresholding` and :py:class:`~dtaianomaly.evaluation.BinaryMetric` in sequence.

    Wrapper to combine a :py:class:`~dtaianomaly.evaluation.BinaryMetric` object with some
    :py:class:`~dtaianomaly.threshold.Thresholding`, to make sure that it can take continuous
    anomaly scores as an input. This is done by first applying
    some :py:class:`~dtaianomaly.threshold.Thresholding` to the predicted anomaly scores, after
    which a :py:class:`~dtaianomaly.evaluation.BinaryMetric` can be computed.

    Parameters
    ----------
    thresholder : Thresholding
        Instance of the desired `Thresholding` class.
    metric : BinaryMetric
        Instance of the desired `Metric` class.

    Examples
    --------
    >>> from dtaianomaly.evaluation import ThresholdMetric, Precision
    >>> from dtaianomaly.thresholding import FixedCutoffThreshold
    >>> metric = ThresholdMetric(FixedCutoffThreshold(0.9), Precision())
    >>> y_true = [   0,   0,   0,   1,   1,   0,   0,   0]
    >>> y_pred = [0.95, 0.5, 0.4, 0.8, 1.0, 0.7, 0.2, 0.1]
    >>> metric.compute(y_true, y_pred)
    0.5
    """

    thresholder: Thresholding
    metric: BinaryMetric

    attribute_validation = {
        "thresholder": ObjectAttribute(Thresholding),
        "metric": ObjectAttribute(BinaryMetric),
    }

    def __init__(self, thresholder: Thresholding, metric: BinaryMetric) -> None:
        self.thresholder = thresholder
        self.metric = metric

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        y_pred_binary = self.thresholder.threshold(y_pred)
        return self.metric._compute(y_true=y_true, y_pred=y_pred_binary)

    def piped_str(self) -> str:
        return f"{self.thresholder}->{self.metric}"
