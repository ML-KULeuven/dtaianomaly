from typing import Optional

import numpy as np

from dtaianomaly.evaluation.metrics import BinaryMetric, ProbaMetric


class BestThresholdMetric(ProbaMetric):
    """
    Compute the maximum score of a binary metric over all thresholds.
    This method will iterate over the possible threshold for given
    predicted anomaly scores, compute the binary metric for each
    threshold, and then return the score for the highest threshold.

    Parameters
    ----------
    metric: BinaryMetric
        Instance of the desired `Metric` class
    max_nb_thresholds: int, default=-1
        The maximum number of thresholds to use for computing the best threshold.
        If ``max_nb_thresholds = -1``, all thresholds will be used. Otherwise, the
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
    """

    metric: BinaryMetric
    max_nb_thresholds: int
    threshold_: float
    thresholds_: np.array
    scores_: np.array

    def __init__(self, metric: BinaryMetric, max_nb_thresholds: int = -1) -> None:
        if not isinstance(metric, BinaryMetric):
            raise TypeError(f"metric expects 'BinaryMetric', got {type(metric)}")
        if not isinstance(max_nb_thresholds, int) or isinstance(
            max_nb_thresholds, bool
        ):
            raise TypeError("`max_nb_thresholds` should be an integer")
        if max_nb_thresholds <= 0 and max_nb_thresholds != -1:
            raise ValueError(
                "`max_nb_thresholds` must be strictly positive or equal to -1!"
            )
        self.metric = metric
        self.max_nb_thresholds = max_nb_thresholds

    def _compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        thresholds: Optional[np.array] = None,
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
        if 0 < self.max_nb_thresholds < thresholds.shape[0]:
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
        return best_score
