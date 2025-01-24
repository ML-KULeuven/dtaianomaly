import abc

import numpy as np

from dtaianomaly.evaluation.metrics import BinaryMetric
from dtaianomaly.evaluation.simple_binary_metrics import FBeta, Precision, Recall


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Apply point-adjustment to the given arrays. For each anomalous
    event in the ground truth (a sequence of consecutive anomalous
    observations), if any observation is predicted as an anomaly,
    all observations in the sequence are said to be detected.

    Parameters
    ----------
    y_true: array-like of shape (n_samples)
        Ground-truth labels.
    y_pred: array-like of shape (n_samples)
        Predicted anomaly scores.

    Returns
    -------
    point_adjusted_y_pred: array-like of shape (n_samples)
        The point adjusted predicted anomalies
    """
    # Find the anomalous events
    diff = np.diff(y_true, prepend=0, append=0)
    start_events = np.where(diff == 1)[0]
    end_events = np.where(diff == -1)[0]

    # Check if an anomaly is detected in any anomalous event
    point_adjusted_y_pred = y_pred.copy()
    for start, end in zip(start_events, end_events):
        if y_pred[start:end].any():
            point_adjusted_y_pred[start:end] = 1

    # Return the point adjusted scores
    return point_adjusted_y_pred


class PointAdjusted(BinaryMetric, abc.ABC):

    metric: BinaryMetric

    def __init__(self, metric: BinaryMetric):
        if not isinstance(metric, BinaryMetric):
            raise TypeError("The given `metric` should be a binary metric")
        self.metric = metric

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        point_adjusted_y_pred = point_adjust(y_true, y_pred)
        return self.metric._compute(y_true, point_adjusted_y_pred)


class PointAdjustedPrecision(PointAdjusted):
    """
    Compute the point-adjusted precision: first point-adjust the predicted
    anomaly scores, after which the precision is computed.

    For given binary anomaly predictions and ground truth anomaly labels,
    point-adjusting will treat any sequence of consecutive ground truth
    anomalies as anomalous events. If any of the observations in such an
    event has been detected, then we say that the anomaly has been detected.
    In this case, all predictions in the anomalous event are set to 1,
    thereby indicating that the method predicted an anomaly.

    Attributes
    ----------
    metric: Precision
        The Precision-object used for computing the precision, after the
        prediction has been point adjusted. Note that the object should
        not be passed to the constructor.

    See Also
    --------
    Precision: Compute the standard, not point-adjusted precision.
    """

    def __init__(self):
        super().__init__(Precision())


class PointAdjustedRecall(PointAdjusted):
    """
    Compute the point-adjusted recall: first point-adjust the predicted
    anomaly scores, after which the recall is computed.

    For given binary anomaly predictions and ground truth anomaly labels,
    point-adjusting will treat any sequence of consecutive ground truth
    anomalies as anomalous events. If any of the observations in such an
    event has been detected, then we say that the anomaly has been detected.
    In this case, all predictions in the anomalous event are set to 1,
    thereby indicating that the method predicted an anomaly.

    Attributes
    ----------
    metric: Recall
        The Recall-object used for computing the precision, after the
        recall has been point adjusted. Note that the object should
        not be passed to the constructor.

    See Also
    --------
    Recall: Compute the standard, not point-adjusted recall.
    """

    def __init__(self):
        super().__init__(Recall())


class PointAdjustedFBeta(PointAdjusted):
    """
    Compute the point-adjusted :math:`F_\\beta`: first point-adjust the predicted
    anomaly scores, after which the :math:`F_\\beta` is computed.

    For given binary anomaly predictions and ground truth anomaly labels,
    point-adjusting will treat any sequence of consecutive ground truth
    anomalies as anomalous events. If any of the observations in such an
    event has been detected, then we say that the anomaly has been detected.
    In this case, all predictions in the anomalous event are set to 1,
    thereby indicating that the method predicted an anomaly.

    Parameters
    ----------
    beta: int, float, default=1
        Desired beta parameter.

    Attributes
    ----------
    metric: FBeta
        The FBeta-object used for computing the precision, after the
        :math:`F_\\beta`: has been point adjusted. Note that the object should
        not be passed to the constructor.

    See Also
    --------
    FBeta: Compute the standard, not point-adjusted :math:`F_\\beta`.
    """

    def __init__(self, beta: (float, int) = 1):
        super().__init__(FBeta(beta))

    @property
    def beta(self):
        return self.metric.beta
