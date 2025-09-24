import numpy as np

from dtaianomaly.evaluation._BinaryMetric import BinaryMetric
from dtaianomaly.evaluation._ProbaMetric import ProbaMetric
from dtaianomaly.type_validation import IntegerAttribute, NoneAttribute, ObjectAttribute

__all__ = ["BestThresholdMetric"]


class BestThresholdMetric(ProbaMetric):
    """
    Compute the maximum score across all thresholds.

    Compute the maximum score of a :py:class:`~dtaianomaly.evaluation.BinaryMetric` over all thresholds.
    This method will iterate over the possible thresholds for given
    predicted anomaly scores, compute the :py:class:`~dtaianomaly.evaluation.BinaryMetric` for each
    threshold, and then return the score for the highest threshold.

    Parameters
    ----------
    metric: BinaryMetric
        Instance of the desired :py:class:`~dtaianomaly.evaluation.BinaryMetric` class
    max_nb_thresholds: int, default=None
        The maximum number of thresholds to use for computing the best threshold.
        If ``max_nb_thresholds = None``, all thresholds will be used. Otherwise, the
        value indicates the subsample of all possible thresholds that should be used.
        This subset is created by first sorting the possible unique thresholds, and
        then selecting the threshold at regular intervals (i.e., the 3rd, 6th, 9th, ...).
        We recommend using the default value (use all thresholds), but can be used
        for reducing the resource requirements.

    Attributes
    ----------
    threshold_: float
        The threshold resulting in the best performance.
    thresholds_: array-like of floats
        The thresholds used for evaluating the performance.
    scores_: array-like of floats
        The evaluation scores corresponding to each threshold in ``thresholds_``.

    Examples
    --------
    >>> from dtaianomaly.evaluation import BestThresholdMetric, Precision
    >>> metric = BestThresholdMetric(Precision())
    >>> y_true = [   0,   0,   0,   1,   1,   0,   0,   0]
    >>> y_pred = [0.95, 0.5, 0.4, 0.8, 1.0, 0.7, 0.2, 0.1]
    >>> metric.compute(y_true, y_pred)
    1.0
    """

    metric: BinaryMetric
    max_nb_thresholds: int | None
    threshold_: float
    thresholds_: np.array
    scores_: np.array

    attribute_validation = {
        "metric": ObjectAttribute(BinaryMetric),
        "max_nb_thresholds": IntegerAttribute(minimum=1) | NoneAttribute(),
    }

    def __init__(self, metric: BinaryMetric, max_nb_thresholds: int = None) -> None:
        self.metric = metric
        self.max_nb_thresholds = max_nb_thresholds

    def _compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        thresholds: np.array = None,
    ) -> float:
        """
        Effectively compute the score corresponding to the best threshold.

        Parameters
        ----------
        y_true: array-like of shape (n_samples)
            Ground-truth labels.
        y_pred: array-like of shape (n_samples)
            Predicted anomaly scores.
        thresholds: array-like of float, default=None
            The thresholds that should be used for computing the metric. If
            ``thresholds=None`` (default), then all possible thresholds will
            be used.

        Returns
        -------
        score: float
            The best evaluation score across all thresholds.
        """
        # Sort all the predicted scores
        sorted_predicted_scores = np.sort(np.unique(y_pred))

        # Compute the thresholds if none are given
        if thresholds is None:
            # Get all possible thresholds
            thresholds = (
                sorted_predicted_scores[:-1] + sorted_predicted_scores[1:]
            ) / 2.0

            # Add the minimum and maximum threshold
            thresholds = np.append(np.insert(thresholds, 0, 0), 1)

        # Select a subset of the thresholds, if requested and useful
        if (
            self.max_nb_thresholds is not None
            and 0 < self.max_nb_thresholds < thresholds.shape[0]
        ):
            selected_thresholds = np.linspace(
                0, thresholds.shape[0], self.max_nb_thresholds + 2, dtype=int
            )[1:-1]
            thresholds = thresholds[selected_thresholds]

        # Compute the score for each threshold
        self.thresholds_ = thresholds
        self.scores_ = np.array(
            [
                self.metric._compute(y_true, y_pred >= threshold)
                for threshold in self.thresholds_
            ]
        )

        # Get the best score and the corresponding threshold
        i = np.argmax(self.scores_)
        best_score = self.scores_[i]
        self.threshold_ = self.thresholds_[i]

        # Return the best score
        return float(best_score)
