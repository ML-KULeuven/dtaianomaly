import copy
import warnings
from typing import Literal

import numpy as np

from dtaianomaly import utils
from dtaianomaly.anomaly_detection._BaseDetector import BaseDetector
from dtaianomaly.type_validation import BoolAttribute, LiteralAttribute, ObjectAttribute

__all__ = ["MultivariateDetector"]


class MultivariateDetector(BaseDetector):
    """
    Multivariate wrapper for anomaly detection.

    Wraps around existing anomaly detectors to detect anomalies in multivariate
    time series. This is done by detecting anomalies in each attribute independently.
    This approach lifts univariate models to the multivariate setting. In addition,
    methods which detect anomalies using a multivariate sliding window (e.g., all
    ``PyODAnomalyDetector``) receive a lot of input features. The ``MultivariateDetector``
    limits the amount of input features, which may improve the performance.

    Note that each feature is handled independently, which makes it impossible
    to detect anomalies based on the relation of multiple attributes.

    Parameters
    ----------
    detector: BaseDetector
        The anomaly detector used to detect anomalies in each attribute.
    aggregation: {'min', 'max', 'mean'}, default='max'
        Manner to aggregate the anomaly scores across each dimension.
    raise_warning_for_univariate: bool, default=True
        Whether to raise a warning when the model is fitted on a univariate
        time series. Teh value does not change the output of the model, but
        only serves to surpress the warning message.

    Attributes
    ----------
    fitted_detectors_: list of BaseDetector
        The fitted anomaly detectors, one for each attribute.

    Examples
    --------
    >>> import numpy as np
    >>> from dtaianomaly.anomaly_detection import MultivariateDetector, IsolationForest
    >>> x = np.array([[4, 8], [1, 2], [0, 1], [6, 5], [1, 4], [4, 3], [0, 9], [8, 2], [4, 5], [8, 3]])
    >>> detector = MultivariateDetector(IsolationForest(window_size=3, random_state=0), aggregation='mean').fit(x)
    >>> detector.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([-0.03045931, -0.04993609, -0.05237944, -0.07038518, -0.05778077,
           -0.0489984 , -0.02691477, -0.02928812, -0.02847268, -0.0387197 ])
    """

    detector: BaseDetector
    aggregation: Literal["min", "max", "mean"]
    raise_warning_for_univariate: bool
    fitted_detectors_: list[BaseDetector]

    attribute_validation = {
        "detector": ObjectAttribute(BaseDetector),
        "aggregation": LiteralAttribute("min", "max", "mean"),
        "raise_warning_for_univariate": BoolAttribute(),
    }

    def __init__(
        self,
        detector: BaseDetector,
        aggregation: Literal["min", "max", "mean"] = "max",
        raise_warning_for_univariate: bool = True,
    ):
        self.detector = detector
        super().__init__(detector.supervision)
        self.aggregation = aggregation
        self.raise_warning_for_univariate = raise_warning_for_univariate

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:

        # Raise a warning if fitted on univariate data
        if utils.is_univariate(X):
            X = X.reshape(-1, 1)
            if self.raise_warning_for_univariate:
                warnings.warn(
                    f"Applying {self.__class__.__name__} on univariate data. "
                    f"This will simply apply the anomaly detector."
                )

        # Fit detectors on each attribute
        self.fitted_detectors_ = [
            copy.deepcopy(self.detector).fit(X[:, dimension], y)
            for dimension in range(utils.get_dimension(X))
        ]

    def _decision_function(self, X: np.ndarray) -> np.array:

        # Check if valid dimension
        if utils.get_dimension(X) != len(self.fitted_detectors_):
            raise ValueError(
                f"This {self.__class__.__name__} was fitted on an {len(self.fitted_detectors_)}-dimensional "
                f"time series, but received {utils.get_dimension(X)}-dimensional data!"
            )

        # Make sure the second dimension exists for univariate time series
        if utils.is_univariate(X):
            X = X.reshape(-1, 1)

        # Compute the univariate decision scores
        decision_scores = np.empty(shape=(X.shape[0], len(self.fitted_detectors_)))
        for dimension, detector in enumerate(self.fitted_detectors_):
            decision_scores[:, dimension] = detector.decision_function(X[:, dimension])

        # Aggregate the decision scores
        if self.aggregation == "min":
            return np.min(decision_scores, axis=1)
        elif self.aggregation == "max":
            return np.max(decision_scores, axis=1)
        elif self.aggregation == "mean":
            return np.mean(decision_scores, axis=1)
