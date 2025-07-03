import numpy as np
from sklearn.cluster import KMeans

from dtaianomaly.anomaly_detection import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import (
    check_is_valid_window_size,
    compute_window_size,
    reverse_sliding_window,
    sliding_window,
)


class KMeansAnomalyDetector(BaseDetector):
    """
    Use KMeans clustering to detect anomalies :cite:`yairi2001fault`.

    KMeans anomaly detector  first clusters the data using
    the KMeasn clustering algorithm. Next, for new data, the corresponding
    cluster is predicted, and the distance to the cluster centroid is computed.
    This distance corresponds to the decision scores of this anomaly detector:
    if an instance is far from the centroid, it is more anomalous. The input
    of KMeans clustering is a sliding window.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    n_clusters: int, default=8
        The number of clusters to use for K-means clustering.
    **kwargs:
        Arguments to be passed to KMeans clustering of scikit-learn anomaly detector

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    k_means_ : KMeans
        The KMeans clustering algorithm from scikit-learn

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import KMeansAnomalyDetector
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> kmeans_ad = KMeansAnomalyDetector(10).fit(x)
    >>> kmeans_ad.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.50321076, 0.5753145 , 0.61938076, ..., 0.29794485, 0.30720306, 0.29857479]...)
    """

    window_size: int | str
    stride: int
    n_clusters: int
    kwargs: dict
    window_size_: int
    k_means_: KMeans

    def __init__(
        self, window_size: int | str, stride: int = 1, n_clusters: int = 8, **kwargs
    ):
        super().__init__(Supervision.UNSUPERVISED)

        check_is_valid_window_size(window_size)
        if not isinstance(stride, int) or isinstance(stride, bool):
            raise TypeError("`stride` should be an integer")
        if stride < 1:
            raise ValueError("`stride` should be strictly positive")

        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
            raise TypeError("`n_clusters` should be integer")
        if n_clusters <= 1:
            raise ValueError("`n_clusters` should be at least 2")

        self.window_size = window_size
        self.stride = stride
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        KMeans(
            n_clusters=self.n_clusters, **self.kwargs
        )  # Check if KMeans can be initialized

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)
        self.k_means_ = KMeans(n_clusters=self.n_clusters, **self.kwargs)
        self.k_means_.fit(sliding_window(X, self.window_size_, self.stride))

    def _decision_function(self, X: np.ndarray) -> np.array:
        sliding_windows = sliding_window(X, self.window_size_, self.stride)
        clusters = self.k_means_.predict(sliding_windows)
        distance_to_cluster_centers = np.linalg.norm(
            sliding_windows - self.k_means_.cluster_centers_[clusters], axis=1
        )
        decision_scores = reverse_sliding_window(
            distance_to_cluster_centers, self.window_size_, self.stride, X.shape[0]
        )

        return decision_scores
