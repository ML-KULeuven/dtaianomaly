from pyod.models.iforest import IForest
from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.PyODAnomalyDetector import PyODAnomalyDetector


class IsolationForest(PyODAnomalyDetector):
    """
    Anomaly detector based on the Isolation Forest algorithm.

    The isolation forest [Liu2008isolation]_ generates random binary trees to
    split the data. If an instance requires fewer splits to isolate it from
    the other data, it is nearer to the root of the tree, and consequently
    receives a higher anomaly score.

    Notes
    -----
    The isolation forest inherets from :py:class:`~dtaianomaly.anomaly_detection.PyodAnomalyDetector`.

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
    >>> isolation_forest.decision_function(x)
    array([-0.02301142, -0.01266304, -0.00786237, ..., -0.04561172,
           -0.0420979 , -0.04414417])

    References
    ----------
    .. [Liu2008isolation] F. T. Liu, K. M. Ting and Z. -H. Zhou, "Isolation Forest,"
       2008 Eighth IEEE International Conference on Data Mining, Pisa, Italy, 2008,
       pp. 413-422, doi: `10.1109/ICDM.2008.17 <https://doi.org/10.1109/ICDM.2008.17>`_.
    """

    def _initialize_detector(self, **kwargs) -> IForest:
        return IForest(**kwargs)

    def _supervision(self):
        return Supervision.UNSUPERVISED
