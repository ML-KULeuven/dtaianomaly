from pyod.models.cblof import CBLOF

from dtaianomaly.anomaly_detection._BaseDetector import Supervision
from dtaianomaly.anomaly_detection._BasePyODAnomalyDetector import (
    BasePyODAnomalyDetector,
)
from dtaianomaly.type_validation import FloatAttribute, IntegerAttribute
from dtaianomaly.windowing import WINDOW_SIZE_TYPE

__all__ = ["ClusterBasedLocalOutlierFactor"]


class ClusterBasedLocalOutlierFactor(BasePyODAnomalyDetector):
    """
    Anomaly detector based on the Cluster-based Local Outlier Factor (CBLOF) :cite:`he2003discovering`.

    CBLOF is a cluster-based LOF which uses the distance to
    clusters in the data to compute an outlier score. Specifically, CBLOF first
    clusters the data using some clustering algorithm (K-means in this implemention). Next,
    the clusters are separated in the so-called 'large clusters' :math:`LC` and 'small
    clusters' :math:`SC`, depending on the parameters :math:`\\alpha` and :math:`\\beta`.
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
    n_clusters: int, default=8
        The number of clusters to form and the number of centroids to generate.
    alpha: float in [0.5, 1.0], default=0.9
        The ratio for deciding small and large clusters. :math:`\\alpha` equals the ratio of number
        of samlples in large clusters to the number of samples in small clusters.
    beta: float, default=5.0
        The ratio for deciding small and large clusters. :math:`\\beta` equals a cutoff for
        the small and large clusters, such that for clusters ordered by size, we have that
        :math:`\\lvert C_k \\rvert / \\lvert C_{k+1} \\rvert = \\beta`.
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
    """

    n_clusters: int
    alpha: float
    beta: float

    attribute_validation = {
        "n_clusters": IntegerAttribute(minimum=2),
        "alpha": FloatAttribute(minimum=0.5, maximum=1.0),
        "beta": FloatAttribute(minimum=1.0),
    }

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        stride: int = 1,
        n_clusters: int = 8,
        alpha: float = 0.9,
        beta: float = 5.0,
        **kwargs,
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        super().__init__(window_size, stride, **kwargs)

    def _initialize_detector(self, **kwargs) -> CBLOF:
        return CBLOF(
            n_clusters=self.n_clusters, alpha=self.alpha, beta=self.beta, **kwargs
        )

    def _supervision(self):
        return Supervision.UNSUPERVISED
