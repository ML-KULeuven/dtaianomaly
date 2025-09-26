import numpy as np
import stumpy
from scipy.spatial.distance import pdist, squareform
from tslearn.clustering import KShape

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.type_validation import WindowSizeAttribute
from dtaianomaly.windowing import (
    compute_window_size,
    reverse_sliding_window,
    sliding_window,
)


class KShapeAnomalyDetector(BaseDetector):
    """
    Anomaly detector based on KShape-clustering :cite:`paparrizos2017fast`.

    Use the KShapeAD algorithm to detect anomalies in time series.
    The subsequences are first clustered using KShape-clustering,
    in which the clusters represent the different normal behaviors
    in the data. For each cluster there is also a weight computed
    based on the size of the cluster and the centrality of that
    cluster in comparison to the other clusters. Anomalies are then
    detected by computing a weighted average of the distance of
    a subsequence to each other cluster. KShapeAD equals the
    offline version of SAND :cite:`boniol2021sand`.

    Parameters
    ----------
    window_size: int or str
        The window size, the length of the subsequences that will be detected as anomalies. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    n_clusters: int, default=3
        The number of clusters to use for KShape clustering.
    sequence_length_multiplier: float, default=4
        The amount by which the window size should be multiplied to create
        sliding windows for clustering the data using KShape. Should be
        at least 1, to make sure that the cluster-centroids are larger
        than the sequences to detect anomalies in.
    overlap_rate: float, default=0.5
        The overlap of the sliding windows for clustering the data. Will
        be used to compute a relative stride to avoid trivial matches
        when clustering subsequences.
    **kwargs:
        Arguments to be passed to KShape-clustering of tsslearn.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for computing the matrix profile
    centroids_: list of array-like of shape (window_size_*sequence_length_multiplier,)
        The centroids computed by KShape clustering.
    weights_: list of float
        The normalized weights corresponding to each cluster.
    kshape_: KShape
        The fitted KShape-object of tslearn, used to cluster the data.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import KShapeAnomalyDetector
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> kshape = KShapeAnomalyDetector(window_size=50).fit(x)
    >>> kshape.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([1.01942655, 1.03008335, 1.03906465, ..., 1.29643677, 1.3256903 , 1.34704128]...)

    Notes
    -----
    KshapeAD only handles univariate time series.
    """

    window_size: str | int
    n_clusters: int
    sequence_length_multiplier: float
    overlap_rate: float
    kwargs: dict

    window_size_: int
    centroids_: list[np.array]
    weights_: np.array
    kshape_: KShape

    attribute_validation = {"window_size": WindowSizeAttribute()}

    def __init__(
        self,
        window_size: str | int,
        n_clusters: int = 3,
        sequence_length_multiplier: float = 4,
        overlap_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(Supervision.UNSUPERVISED)

        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
            raise TypeError("`n_clusters` should be integer")
        if n_clusters <= 1:
            raise ValueError("`n_clusters` should be at least 2")

        if not isinstance(sequence_length_multiplier, (float, int)) or isinstance(
            sequence_length_multiplier, bool
        ):
            raise TypeError("`sequence_length_multiplier` should be numeric")
        if sequence_length_multiplier < 1.0:
            raise ValueError(
                "`sequence_length_multiplier` should be at least 1 or larger"
            )

        if not isinstance(overlap_rate, float):
            raise TypeError("`overlap_rate` should be a float")
        if not 0 < overlap_rate <= 1.0:
            raise ValueError("`overlap_rate` should be at larger than 0 and at most 1")

        # Check if KShape can be initialized
        KShape(n_clusters=n_clusters, **kwargs)

        self.window_size = window_size
        self.n_clusters = n_clusters
        self.sequence_length_multiplier = sequence_length_multiplier
        self.overlap_rate = overlap_rate
        self.kwargs = kwargs

    def theta_(self) -> list[(np.array, float)]:
        """
        Computes :math:`\\Theta = \\{(C_0, w_0), \\dots, (C_k, w_k)\\}`, the normal
        behavior consisting of  :math:`k` clusters.

        Returns
        -------
        theta: list of tuples of array-likes of shape (window_size_*sequence_length_multiplier,) and floats
            A list of tuples in which the first element consists of the centroid
            corresponding to each cluster and the second element corresponds to
            the normalized weight of that cluster.
        """
        self.check_is_fitted()
        return list(zip(self.centroids_, self.weights_))

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        # Make sure the data is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")
        X = X.squeeze()

        # Compute the window size
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)

        # Compute sliding windows
        sequence_length = int(self.window_size_ * self.sequence_length_multiplier)
        stride = int(sequence_length * self.overlap_rate)
        windows = sliding_window(X, sequence_length, stride)

        # Apply K-Shape clustering
        self.kshape_ = KShape(n_clusters=self.n_clusters, **self.kwargs)
        cluster_labels = self.kshape_.fit_predict(windows)

        # Extract the centroids
        self.centroids_ = list(map(np.squeeze, self.kshape_.cluster_centers_))
        _, cluster_sizes = np.unique(cluster_labels, return_counts=True)
        summed_cluster_distances = squareform(
            pdist(self.centroids_, metric=_shape_based_distance)
        ).sum(axis=0)

        # Normalize cluster size and summed cluster distances
        cluster_sizes = _min_max_normalization(cluster_sizes)
        summed_cluster_distances = _min_max_normalization(summed_cluster_distances)

        # Compute the weights
        self.weights_ = cluster_sizes**2 / summed_cluster_distances
        self.weights_ /= self.weights_.sum()

    def _decision_function(self, X: np.ndarray) -> np.array:
        # Make sure the data is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")
        X = X.squeeze()

        # Compute the minimum distance of each subsequence to each cluster using matrix profile
        min_distance = np.array(
            [
                stumpy.stump(X, self.window_size_, centroid, ignore_trivial=False)[:, 0]
                for centroid in self.centroids_
            ]
        )

        # Anomaly scores are weighted average of the minimum distances
        anomaly_scores = np.matmul(self.weights_, min_distance)

        # Return anomaly score per window
        return reverse_sliding_window(anomaly_scores, self.window_size_, 1, X.shape[0])


def _min_max_normalization(x: np.array) -> np.array:
    return (x - x.min()) / (x.max() - x.min() + 0.0000001) + 1


def _shape_based_distance(x: np.array, y: np.array) -> float:
    ncc = _ncc_c(x, y)
    return 1 - ncc.max()


def _ncc_c(x: np.array, y: np.array) -> np.array:
    den = np.array(np.linalg.norm(x) * np.linalg.norm(y))
    den[den == 0] = np.inf

    fft_size = 1 << (2 * x.shape[0] - 1).bit_length()
    cc = np.fft.ifft(np.fft.fft(x, fft_size) * np.conj(np.fft.fft(y, fft_size)))
    cc = np.concatenate((cc[-(x.shape[0] - 1) :], cc[: x.shape[0]]))
    return np.real(cc) / den
