import numpy as np
from sklearn import metrics

from dtaianomaly.evaluation.metrics import BinaryMetric


class Precision(BinaryMetric):
    """
    Computes the Precision score.

    Precision measures how accurately the model identifies anomalies.
    It reflects the proportion of detected anomalies that are truly
    abnormal. This is particularly important when the cost of false
    positives (normal events incorrectly flagged as anomalies) is high.

    Mathematically, precision is the ratio of true positives (correctly
    identified anomalies) to all predicted positives, which includes
    both true anomalies and false positives (normal events mistakenly
    flagged as anomalies). It can be expressed as:

    .. math::

       \\text{Precision} = \\frac{\\text{True Anomalies}}{\\text{True Anomalies} + \\text{False Positives}}


    A high precision in anomaly detection indicates that the model
    generates few false alarms, ensuring that most flagged anomalies
    are truly abnormal events. However, it does not measure how many
    anomalies were actually identified.
    """

    def __init__(self) -> None:
        super().__init__()

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.precision_score(y_true=y_true, y_pred=y_pred)


class Recall(BinaryMetric):
    """
    Computes the Recall score.

    Recall measures the model's ability to correctly identify all actual
    anomalies. It tells us the proportion of true anomalies that were
    successfully detected by the model. In an anomaly detection system,
    recall answers the question: "Of all the anomalies that occurred,
    how many did the model detect?" A high recall is especially important
    when missing actual anomalies (false negatives) could have severe
    consequences.

    Mathematically, recall is the ratio of true positives (correctly
    identified anomalies) to all actual positives, which includes both
    true anomalies and false negatives (missed anomalies). It can be expressed as:

    .. math::

       \\text{Recall} = \\frac{\\text{True Anomalies}}{\\text{True Anomalies} + \\text{False Negatives}}

    A high recall ensures that most anomalies are detected, but it doesn’t
    account for how many false positives (normal events incorrectly flagged
    as anomalies) were generated, which is handled by precision.
    """

    def __init__(self) -> None:
        super().__init__()

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.recall_score(y_true=y_true, y_pred=y_pred)


class FBeta(BinaryMetric):
    """
    Computes the :math:`F_\\beta` score.

    The :math:`F_\\beta` combines both precision and recall into a single
    value. It provides a balanced evaluation of a model’s performance,
    especially in anomaly detection, where there is often a trade-off
    between catching all anomalies (high recall) and minimizing false
    alarms (high precision). The parameter :math:`\\beta` controls the balance
    between precision and recall. A :math:`\\beta > 1` gives more weight to
    recall, useful when missing anomalies is costly, while :math:`\\beta < 1`
    emphasizes precision, reducing false positives.

    The :math:`F_\\beta` score is the harmonic mean of precision and recall.
    It can be expressed as:

    .. math::

       F_\\beta = \\frac{(1 + \\beta^2) \\text{tp}}
                        {(1 + \\beta^2) \\text{tp} + \\text{fp} + \\beta^2 \\text{fn}}

    A high :math:`F_\\beta` score indicates a good balance between detecting
    actual anomalies and minimizing false positives.

    Parameters
    ----------
    beta: int, float, default=1
        Desired beta parameter.

    See also
    --------
    Precision: Compute the Precision score.
    Recall: Compute the Recall score.
    """

    beta: float

    def __init__(self, beta: (float, int) = 1) -> None:
        if not isinstance(beta, (int, float)) or isinstance(beta, bool):
            raise TypeError("`beta` should be numeric")
        if beta <= 0.0:
            raise ValueError("`beta` should be strictly positive")

        super().__init__()
        self.beta = beta

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.fbeta_score(y_true=y_true, y_pred=y_pred, beta=self.beta)
