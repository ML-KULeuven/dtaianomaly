
from pyod.models.lof import LOF
from dtaianomaly.anomaly_detection._PyOD import PyODAnomalyDetector


class LocalOutlierFactor(PyODAnomalyDetector):
    """
    Anomaly detector based on the Local Outlier Factor.

    The local outlier factor [Breunig2000LOF]_ compares the density of each
    sample to the density of the neighboring samples. If the neighbors of a
    sample have a much higher density that the sample itself, the sample is
    considered anomalous. By looking at the local density (i.e., only comparing
    with the neighbors of a sample), the local outlier factor takes into
    account varying densities across the sample space.

    Notes
    -----
    The Local Outlier Factor inherets from :py:class:`~dtaianomaly.anomaly_detection.PyodAnomalyDetector`.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
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
    >>> local_outlier_factor.decision_function(x)
    array([0.98370943, 0.98533454, 0.98738196, ..., 1.02394282, 1.02648068,
           1.01827158])

    References
    ----------
    .. [Breunig2000LOF] Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander.
       2000. LOF: identifying density-based local outliers. In Proceedings of the 2000 ACM
       SIGMOD international conference on Management of data (SIGMOD '00). Association for
       Computing Machinery, New York, NY, USA, 93–104. doi: `10.1145/342009.335388 <https://doi.org/10.1145/342009.335388>`_
    """

    def _initialize_detector(self, **kwargs) -> LOF:
        return LOF(**kwargs)
