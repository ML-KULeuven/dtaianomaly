
import numpy as np
from typing import Optional, Union
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

from dtaianomaly import utils
from dtaianomaly.anomaly_detection import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import sliding_window, reverse_sliding_window, check_is_valid_window_size, compute_window_size


class KMeansAnomalyDetector(BaseDetector):
    """
    Use KMeans clustering to detect anomalies.

    KMeans anomaly detector [yairi2001fault]_ first clusters the data using
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
    >>> kmeans_ad.decision_function(x)
    array([0.50321076, 0.5753145 , 0.61938076, ..., 0.29794485, 0.30720306,
           0.29857479])

    References
    ----------
    .. [yairi2001fault] T. Yairi, Y. Kato, and K. Hori. Fault detection by mining association
       rules from house-keeping data. In proceedings of the 6th International Symposium on
       Artificial Intelligence, Robotics and Automation in Space, volume 18, page 21. Citeseer,
       2001.
    """
    window_size: Union[int, str]
    stride: int
    kwargs: dict
    window_size_: int
    k_means_: KMeans

    def __init__(self, window_size: Union[int, str], stride: int = 1, **kwargs):
        super().__init__(Supervision.UNSUPERVISED)

        check_is_valid_window_size(window_size)
        if not isinstance(stride, int) or isinstance(stride, bool):
            raise TypeError("`stride` should be an integer")
        if stride < 1:
            raise ValueError("`stride` should be strictly positive")

        self.window_size = window_size
        self.stride = stride
        self.kwargs = kwargs
        KMeans(**self.kwargs)  # Check if KMeans can be initialized

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'KMeansAnomalyDetector':
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")

        X = np.asarray(X)
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)
        self.k_means_ = KMeans(**self.kwargs)
        self.k_means_.fit(sliding_window(X, self.window_size_, self.stride))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")
        if not hasattr(self, 'k_means_') or not hasattr(self, 'window_size_'):
            raise NotFittedError('Call the fit function before making predictions!')

        X = np.asarray(X)
        sliding_windows = sliding_window(X, self.window_size_, self.stride)
        clusters = self.k_means_.predict(sliding_windows)
        distance_to_cluster_centers = np.linalg.norm(sliding_windows - self.k_means_.cluster_centers_[clusters], axis=1)
        decision_scores = reverse_sliding_window(distance_to_cluster_centers, self.window_size_, self.stride, X.shape[0])

        return decision_scores
