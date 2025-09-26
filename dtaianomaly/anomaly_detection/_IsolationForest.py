from typing import Literal

from pyod.models.iforest import IForest

from dtaianomaly.anomaly_detection._BaseDetector import Supervision
from dtaianomaly.anomaly_detection._BasePyODAnomalyDetector import (
    BasePyODAnomalyDetector,
)
from dtaianomaly.type_validation import (
    FloatAttribute,
    IntegerAttribute,
    LiteralAttribute,
)
from dtaianomaly.windowing import WINDOW_SIZE_TYPE

__all__ = ["IsolationForest"]


class IsolationForest(BasePyODAnomalyDetector):
    """
    Anomaly detector based on the Isolation Forest algorithm :cite:`liu2008isolation`.

    The isolation forest generates random binary trees to
    split the data. If an instance requires fewer splits to isolate it from
    the other data, it is nearer to the root of the tree, and consequently
    receives a higher anomaly score.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    n_estimators: int, default=100
        The number of base trees in the ensemble.
    max_samples: int or float, default='auto'
        The number of samples to draw for training each base estimator:

        - if ``int``: Draw at most ``max_samples`` samples.
        - if ``float``: Draw at most ``max_samples`` percentage of the samples.
        - if ``'auto'``: Set ``max_samples=min(256, n_windows)``.

    max_features: int or float, default=1.0
        The number of features to use for training each base estimator:

        - if ``int``: Use at most ``max_features`` features.
        - if ``float``: Use at most ``max_features`` percentage of the features.

    **kwargs
        Arguments to be passed to the PyOD isolation forest.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    pyod_detector_ : IForest
        An Isolation Forest detector of PyOD

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import IsolationForest
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> isolation_forest = IsolationForest(10).fit(x)
    >>> isolation_forest.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([-0.02301142, -0.01266304, -0.00786237, ..., -0.04561172, -0.0420979 , -0.04414417]...)
    """

    n_estimators: int
    max_samples: float | int | Literal["auto"]
    max_features: int | float

    attribute_validation = {
        "n_estimators": IntegerAttribute(minimum=1),
        "max_samples": IntegerAttribute(minimum=1)
        | FloatAttribute(0.0, 1.0, inclusive_minimum=False)
        | LiteralAttribute("auto"),
        "max_features": IntegerAttribute(minimum=1)
        | FloatAttribute(0.0, 1.0, inclusive_minimum=False),
    }

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        stride: int = 1,
        n_estimators: int = 100,
        max_samples: float | int = "auto",
        max_features: int | float = 1.0,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        super().__init__(window_size, stride, **kwargs)

    def _initialize_detector(self, **kwargs) -> IForest:
        return IForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            **kwargs,
        )

    def _supervision(self):
        return Supervision.UNSUPERVISED
