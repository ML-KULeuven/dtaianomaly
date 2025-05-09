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

    def _initialize_detector(self, **kwargs) -> IForest:
        return IForest(**kwargs)

    def _supervision(self):
        return Supervision.UNSUPERVISED
