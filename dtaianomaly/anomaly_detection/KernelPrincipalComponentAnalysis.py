from pyod.models.kpca import KPCA

from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.PyODAnomalyDetector import PyODAnomalyDetector


class KernelPrincipalComponentAnalysis(PyODAnomalyDetector):
    """
    Anomaly detector based on the Kernel Principal Component Analysis (KPCA) :cite:`hoffmann2007kernel`.

    Standard PCA maps the data to a lower dimensional space through linear
    projections. Deviations in this lower dimensional space are then
    considered to be anomalies. KPCA is a non-linear
    extension of PCA, which maps the data into a new kernel space, from
    which the principal components are learned.

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
    pyod_detector_ : KPCA
        A KPCA-detector of PyOD

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import KernelPrincipalComponentAnalysis
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> kpca = KernelPrincipalComponentAnalysis(10, n_components=2).fit(x)
    >>> kpca.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.03151377, 0.03697829, 0.04415575, ..., 0.03345565, 0.0330048 , 0.03089501]...)

    Notes
    -----
    KPCA inherets from :py:class:`~dtaianomaly.anomaly_detection.PyodAnomalyDetector`.
    """

    def _initialize_detector(self, **kwargs) -> KPCA:
        return KPCA(**kwargs)

    def _supervision(self):
        return Supervision.SEMI_SUPERVISED
