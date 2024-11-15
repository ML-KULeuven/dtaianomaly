
from pyod.models.hbos import HBOS
from dtaianomaly.anomaly_detection.PyODAnomalyDetector import PyODAnomalyDetector


class HistogramBasedOutlierScore(PyODAnomalyDetector):
    """
    Anomaly detector based on the Histogram Based Outlier Score (HBOS) algorithm.

    Histogram Based Outlier Score (HBOS) [goldstein2012histogram]_ constructs for each feature
    a univariate histogram. Bins with a small height (for static bin widths) or wider bins (for
    dynamic bin widths) correspond to sparse regions of the feature space. Thus, values falling
    in these bins lay in sparse regions of the feature space and are considered more anomalous.

    In this implementation, it is possible to set a window size to take the past observations into
    account. However, HBOS assumes feature independence. Therefore, for a time series with :math:`D`
    attributes and a window size :math:`w`, HBOS constructs :math:`D \\times w` independent histograms,
    from which the anomaly score is computed.

    Notes
    -----
    The HBOS detector inherets from :py:class:`~dtaianomaly.anomaly_detection.PyODAnomalyDetector`.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    **kwargs
        Arguments to be passed to the PyOD histogram based outlier score.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    pyod_detector_ : HBOS
        An HBOS detector of PyOD

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import HistogramBasedOutlierScore
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> hbos = HistogramBasedOutlierScore(1).fit(x)
    >>> hbos.decision_function(x)
    array([0.51808795, 0.51808795, 0.51808795, ..., 0.48347552, 0.48347552,
           0.48347552])

    References
    ----------
    .. [goldstein2012histogram] Goldstein, Markus, and Andreas Dengel. "Histogram-based outlier score (hbos):
       A fast unsupervised anomaly detection algorithm." KI-2012: poster and demo track 1 (2012): 59-63.
    """

    def _initialize_detector(self, **kwargs) -> HBOS:
        return HBOS(**kwargs)
