from typing import Literal

from pyod.models.iforest import IForest

from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.PyODAnomalyDetector import PyODAnomalyDetector


class IsolationForest(PyODAnomalyDetector):
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

    Notes
    -----
    The isolation forest inherets from :py:class:`~dtaianomaly.anomaly_detection.PyodAnomalyDetector`.
    """

    n_estimators: int
    max_samples: float | int | Literal["auto"]
    max_features: int | float

    def __init__(
        self,
        window_size: int | str,
        stride: int = 1,
        n_estimators: int = 100,
        max_samples: float | int = "auto",
        max_features: int | float = 1.0,
        **kwargs,
    ):

        if not isinstance(n_estimators, int) or isinstance(n_estimators, bool):
            raise TypeError("`n_estimators` should be integer or 'auto'")
        if n_estimators < 1:
            raise ValueError("`n_estimators` should be strictly positive")

        if isinstance(max_samples, str):
            if max_samples != "auto":
                raise ValueError(
                    "If `max_samples` is a string, it should equal 'auto'."
                )
        else:
            if not isinstance(max_samples, (float, int)) or isinstance(
                max_samples, bool
            ):
                raise TypeError("`max_samples` should be integer or 'auto'")
            if isinstance(max_samples, float) and (max_samples <= 0 or max_samples > 1):
                raise ValueError("`max_samples` between 0 and 1")
            if isinstance(max_samples, int) and max_samples <= 0:
                raise ValueError("`max_samples` must be at least 1")

        if not isinstance(max_features, (float, int)) or isinstance(max_features, bool):
            raise TypeError("`max_features` should be numeric")
        if isinstance(max_features, float) and (max_features <= 0 or max_features > 1):
            raise ValueError("`max_features` between 0 and 1")
        if isinstance(max_features, int) and max_features <= 0:
            raise ValueError("`max_features` must be at least 1")

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
