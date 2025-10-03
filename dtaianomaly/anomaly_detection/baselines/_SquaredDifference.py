import numpy as np

from dtaianomaly import utils
from dtaianomaly.anomaly_detection._BaseDetector import BaseDetector, Supervision
from dtaianomaly.type_validation import BoolAttribute

__all__ = ["SquaredDifference"]


class SquaredDifference(BaseDetector):
    """
    Compute anomaly scores based on the squared difference.

    Baseline anomaly detector, assings an anomaly score as the squared
    difference of an observation and the previous observation. Formally,
    the anomaly score :math:`s_i` for observation :math:`x_i` equals
    :math:`s_i = (x_i - x_{i-1})^2`.

    Parameters
    ----------
    square_errors : bool, default=True
        If the differences should be squared. If False, this detector equals the
        absolute difference.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import SquaredDifference
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> baseline = SquaredDifference().fit(x)
    >>> baseline.decision_function(x)
    array([0.00779346, 0.00779346, 0.00260361, ..., 0.00286662, 0.05578398,
           0.02683475])
    """

    square_errors: bool
    attribute_validation = {"square_errors": BoolAttribute()}

    def __init__(self, square_errors: bool = True):
        super().__init__(Supervision.UNSUPERVISED)
        self.square_errors = square_errors

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        """Requires no fitting."""

    def _decision_function(self, X: np.ndarray) -> np.array:
        # Check if the given dataset is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")

        decision_scores = np.empty(shape=X.shape[0])
        decision_scores[1:] = np.abs(np.diff(X.squeeze()))
        decision_scores[0] = decision_scores[1]  # Padding
        if self.square_errors:
            decision_scores = np.square(decision_scores)
        return decision_scores


def main():
    from dtaianomaly.data import demonstration_time_series
    from dtaianomaly.visualization import plot_anomaly_scores

    x, y = demonstration_time_series()
    x = x.reshape(-1, 1)

    baseline = SquaredDifference().fit(x)
    y_pred = baseline.decision_function(x)
    plot_anomaly_scores(x, y, y_pred, figsize=(20, 5)).show()


if __name__ == "__main__":
    main()
