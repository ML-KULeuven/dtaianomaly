from pyod.models.pca import PCA

from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.PyODAnomalyDetector import PyODAnomalyDetector


class PrincipalComponentAnalysis(PyODAnomalyDetector):
    """
    Anomaly detector based on the Principal Component Analysis (PCA).

    PCA [Charu2015outlier]_ maps the data to a lower dimensional space
    through linear projections. The goal of these projections is to
    capture the most important information of the samples. This important
    information is related to the type of behaviors that occur frequently
    in the data. Thus, anomalies are detected by measuring the deviation
    of the samples in the lower dimensional space.

    Notes
    -----
    PCA inherets from :py:class:`~dtaianomaly.anomaly_detection.PyodAnomalyDetector`.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    **kwargs:
        Arguments to be passed to the PyOD PCA.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    pyod_detector_ : PCA
        A PCA-detector of PyOD

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import PrincipalComponentAnalysis
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> pca = PrincipalComponentAnalysis(10).fit(x)
    >>> pca.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([16286.63724327, 15951.05917741, 15613.5739773 , ..., 18596.5273311 , 18496.96613747, 18483.47985554]...)

    References
    ----------
    .. [Charu2015outlier] Charu C Aggarwal. Outlier analysis. In Data mining, 75–79. Springer, 2015.
    """

    def _initialize_detector(self, **kwargs) -> PCA:
        return PCA(**kwargs)

    def _supervision(self):
        return Supervision.SEMI_SUPERVISED
