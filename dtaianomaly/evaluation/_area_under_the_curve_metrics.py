import numpy as np
from sklearn import metrics

from dtaianomaly.evaluation._ProbaMetric import ProbaMetric

__all__ = ["AreaUnderROC", "AreaUnderPR"]


class AreaUnderROC(ProbaMetric):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) score.

    The AUC-ROC is a widely used metric to evaluate the performance of
    a binary classifier, especially in anomaly detection. The ROC-curve
    plots the true positive rate (recall) against the false positive
    rate across different classification thresholds. The AUC-ROC represents
    the likelihood that the model ranks a randomly chosen anomaly higher
    than a randomly chosen normal instance.
    AUC-ROC provides a single number summarizing the model's ability to
    distinguish between normal and anomalous instances. A value of 1.0
    indicates perfect discrimination, while 0.5 implies the model performs
    no better than random guessing. It is especially useful when anomalies
    are rare, as it considers the trade-off between detecting true anomalies
    (high recall) and minimizing false positives.

    See Also
    --------
    AreaUnderPR: Compute the Area Under the PR-Curve.

    Examples
    --------
    >>> from dtaianomaly.evaluation import AreaUnderROC
    >>> metric = AreaUnderROC()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)  # doctest: +ELLIPSIS
    0.833...
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        return float(metrics.roc_auc_score(y_true=y_true, y_score=y_pred))


class AreaUnderPR(ProbaMetric):
    """
    Computes the Area Under the Precision-Recall Curve (AUC-PR) score.

    The AUC-PR is a performance metric that is especially useful for
    evaluating models in imbalanced datasets, such as anomaly detection,
    where the number of normal instances vastly outnumbers the anomalies.
    The Precision-Recall curve plots precision against recall at various
    thresholds, providing a detailed view of the trade-off between
    detecting true anomalies (recall) and minimizing false alarms (precision).
    AUC-PR summarizes the curve into a single value, representing the overall
    ability of the model to identify anomalies while keeping false positives
    in check. A higher AUC-PR value indicates better performance, meaning the
    model is effective at detecting true anomalies with fewer false positives.

    See Also
    --------
    AreaUnderROC: Compute the Area Under the ROC-Curve.

    Examples
    --------
    >>> from dtaianomaly.evaluation import AreaUnderPR
    >>> metric = AreaUnderPR()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)
    0.75
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
        return float(metrics.auc(recall, precision))
