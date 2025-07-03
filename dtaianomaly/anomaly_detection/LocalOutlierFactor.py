import numpy as np
import scipy
from pyod.models.lof import LOF

from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.PyODAnomalyDetector import PyODAnomalyDetector


class LocalOutlierFactor(PyODAnomalyDetector):
    """
    Anomaly detector based on the Local Outlier Factor :cite:`breunig2000lof`.

    The local outlier factor  compares the density of each
    sample to the density of the neighboring samples. If the neighbors of a
    sample have a much higher density that the sample itself, the sample is
    considered anomalous. By looking at the local density (i.e., only comparing
    with the neighbors of a sample), the local outlier factor takes into
    account varying densities across the sample space.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    n_neighbors: int, default=5
        The number of neighbors to use for the nearest neighbor queries.
    metric: str, default='minkowski'
        Distance metric for distance computations. any metric of scikit-learn and
        scipy.spatial.distance can be used.
    **kwargs:
        Arguments to be passed to the PyOD local outlier factor

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    pyod_detector_ : LOF
        A LOF-detector of PyOD

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import LocalOutlierFactor
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> local_outlier_factor = LocalOutlierFactor(10).fit(x)
    >>> local_outlier_factor.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.98370943, 0.98533454, 0.98738196, ..., 1.02394282, 1.02648068, 1.01827158]...)

    Notes
    -----
    The Local Outlier Factor inherets from :py:class:`~dtaianomaly.anomaly_detection.PyodAnomalyDetector`.
    """

    n_neighbors: int
    metric: str

    def __init__(
        self,
        window_size: int | str,
        stride: int = 1,
        n_neighbors: int = 5,
        metric: str = "minkowski",
        **kwargs,
    ):

        if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool):
            raise TypeError("`n_neighbors` should be integer")
        if n_neighbors < 1:
            raise ValueError("`n_neighbors` should be at least 1")

        if not isinstance(metric, str):
            raise TypeError("`metric` must be a string")

        scipy.spatial.distance.pdist(np.array([[0, 0], [1, 1]]), metric=metric)

        self.n_neighbors = n_neighbors
        self.metric = metric

        super().__init__(window_size, stride, **kwargs)

    def _initialize_detector(self, **kwargs) -> LOF:
        return LOF(n_neighbors=self.n_neighbors, metric=self.metric, **kwargs)

    def _supervision(self):
        return Supervision.UNSUPERVISED
