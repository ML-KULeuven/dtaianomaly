from pyod.models.cblof import CBLOF

from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.PyODAnomalyDetector import PyODAnomalyDetector


class ClusterBasedLocalOutlierFactor(PyODAnomalyDetector):
    """
    Anomaly detector based on the Cluster-based Local Outlier Factor (CBLOF).

    CBLOF [he2003discovering]_ is a cluster-based LOF which uses the distance to
    clusters in the data to compute an outlier score. Specifically, CBLOF first
    clusters the data using some clustering algorithm (by default K-means). Next,
    the clusters are separated in the so-called 'large clusters' $LC$ and 'small
    clusters' $SC$, depending on the parameters :math:`\\alpha` and :math:`\\beta`.
    Then, the Cluster-based Local outlier Factor of an observation :math:`o` belonging
    to cluster :math:`C_i` is computed as follows:

    .. math::

       \\begin{equation}
           CBLOF(o) = \\lvert C_i \\rvert \\cdot
           \\begin{cases}
               dist(o, C_i), & \\text{if $C_i \\in LC$}. \\\\
               min_{C_j \\in LC} (dist(o, C_j)), & \\text{if $C_i \\in SC$}.
           \\end{cases}
       \\end{equation}

    Specifically, if :math:`o` is part of a large cluster :math:`C_i`, we multiply the size
    of :math:`C_i` with the distance of :math:`o` to  :math:`C_i`. If :math:`o` is in a small
    cluster, then the size of  :math:`C_i` is multiplied by the distance to the nearest
    *large* cluster  :math:`C_j`.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    **kwargs
        Arguments to be passed to the PyOD CBLOF.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    pyod_detector_ : CBLOF
        A CBLOF detector of PyOD

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import ClusterBasedLocalOutlierFactor
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> cblof = ClusterBasedLocalOutlierFactor(10).fit(x)
    >>> cblof.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.50321076, 0.5753145 , 0.61938076, ..., 0.29794485, 0.30720306,  0.29857479]...)


    Notes
    -----
    CBLOF inherets from :py:class:`~dtaianomaly.anomaly_detection.PyodAnomalyDetector`.

    References
    ----------
    .. [he2003discovering] He, Zengyou, Xiaofei Xu, and Shengchun Deng. "Discovering
       cluster-based local outliers." Pattern recognition letters 24.9-10 (2003): 1641-1650,
       doi: `10.1016/S0167-8655(03)00003-5 <https://doi.org/10.1016/S0167-8655(03)00003-5>`_.
    """

    def _initialize_detector(self, **kwargs) -> CBLOF:
        return CBLOF(**kwargs)

    def _supervision(self):
        return Supervision.UNSUPERVISED
