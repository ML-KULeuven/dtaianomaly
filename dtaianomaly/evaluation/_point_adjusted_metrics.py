import numpy as np
from sklearn import metrics

from dtaianomaly.evaluation._BinaryMetric import BinaryMetric
from dtaianomaly.evaluation._FBetaMixin import FBetaMixin

__all__ = ["PointAdjustedFBeta", "PointAdjustedPrecision", "PointAdjustedRecall"]


def _point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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


class PointAdjustedPrecision(BinaryMetric):
    """
    Compute the point-adjusted precision.

    For given binary anomaly predictions and ground truth anomaly labels,
    point-adjusting will treat any sequence of consecutive ground truth
    anomalies as anomalous events. If any of the observations in such an
    event has been detected, then we say that the anomaly has been detected.
    In this case, all predictions in the anomalous event are set to 1,
    thereby indicating that the method predicted an anomaly.

    See Also
    --------
    Precision: Compute the standard, not point-adjusted precision.

    Warnings
    --------
    It is known that the point-adjusted metrics heavily overestimate
    the performance of anomaly detectors. It is therefore not recommended
    to solely rely on those metrics to evaluate a model. These metrics
    were implemented for reproducibility of existing works.

    Examples
    --------
    >>> from dtaianomaly.evaluation import PointAdjustedPrecision
    >>> metric = PointAdjustedPrecision()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)
    0.5
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        return metrics.precision_score(
            y_true=y_true, y_pred=_point_adjust(y_true, y_pred)
        )


class PointAdjustedRecall(BinaryMetric):
    """
    Compute the point-adjusted recall.

    For given binary anomaly predictions and ground truth anomaly labels,
    point-adjusting will treat any sequence of consecutive ground truth
    anomalies as anomalous events. If any of the observations in such an
    event has been detected, then we say that the anomaly has been detected.
    In this case, all predictions in the anomalous event are set to 1,
    thereby indicating that the method predicted an anomaly.

    See Also
    --------
    Recall: Compute the standard, not point-adjusted recall.

    Warnings
    --------
    It is known that the point-adjusted metrics heavily overestimate
    the performance of anomaly detectors. It is therefore not recommended
    to solely rely on those metrics to evaluate a model. These metrics
    were implemented for reproducibility of existing works.

    Examples
    --------
    >>> from dtaianomaly.evaluation import PointAdjustedRecall
    >>> metric = PointAdjustedRecall()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)
    1.0
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        return metrics.recall_score(y_true=y_true, y_pred=_point_adjust(y_true, y_pred))


class PointAdjustedFBeta(BinaryMetric, FBetaMixin):
    """
    Compute the point-adjusted :math:`F_\\beta`.

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

    See Also
    --------
    FBeta: Compute the standard, not point-adjusted :math:`F_\\beta`.

    Warnings
    --------
    It is known that the point-adjusted metrics heavily overestimate
    the performance of anomaly detectors. It is therefore not recommended
    to solely rely on those metrics to evaluate a model. These metrics
    were implemented for reproducibility of existing works.

    Examples
    --------
    >>> from dtaianomaly.evaluation import PointAdjustedFBeta
    >>> metric = PointAdjustedFBeta()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)  # doctest: +ELLIPSIS
    0.666...
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        y_pred_adjusted = _point_adjust(y_true, y_pred)
        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred_adjusted)
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred_adjusted)
        return self._f_score(precision, recall)
