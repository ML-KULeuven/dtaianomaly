from typing import List, Optional, Tuple, Union

import numpy as np
import stumpy
from scipy.spatial.distance import pdist, squareform
from sklearn.exceptions import NotFittedError
from tslearn.clustering import KShape

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import (
    check_is_valid_window_size,
    compute_window_size,
    reverse_sliding_window,
    sliding_window,
)


class KShapeAnomalyDetector(BaseDetector):
    """
    Anomaly detector based on KShape-clustering.

    Use the KShapeAD algorithm to detect anomalies in time series [paparrizos2017fast]_.
    The subsequences are first clustered using KShape-clustering,
    in which the clusters represent the different normal behaviors
    in the data. For each cluster there is also a weight computed
    based on the size of the cluster and the centrality of that
    cluster in comparison to the other clusters. Anomalies are then
    detected by computing a weighted average of the distance of
    a subsequence to each other cluster. KShapeAD equals the
    offline version of SAND [boniol2021sand]_

    Parameters
    ----------
    window_size: int or str
        The window size, the length of the subsequences that will be detected as anomalies. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
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

    Notes
    -----
    KshapeAD only handles univariate time series.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import KShapeAnomalyDetector
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> matrix_profile = KShapeAnomalyDetector(window_size=50).fit(x)
    >>> matrix_profile.decision_function(x)
    array([1.01942655, 1.03008335, 1.03906465, ..., 1.29643677, 1.3256903 ,
           1.34704128])

    References
    ----------
    .. [paparrizos2017fast] Paparrizos, J. and Gravano, L., 2017. Fast and accurate
       time-series clustering. ACM Transactions on Database Systems (TODS), 42(2),
       pp.1-49, doi: `10.1145/3044711 <https://doi.org/10.1145/3044711>`_
    .. [boniol2021sand] Boniol, P., Paparrizos, J., Palpanas, T. and Franklin, M.J.,
       2021. SAND: streaming subsequence anomaly detection. Proceedings of the VLDB
       Endowment, 14(10), pp.1717-1729, doi: `10.14778/3467861.3467863 <https://doi.org/10.14778/3467861.3467863>`_
    """

    window_size: Union[str, int]
    sequence_length_multiplier: float
    overlap_rate: float
    kwargs: dict

    window_size_: int
    centroids_: List[np.array]
    weights_: np.array
    kshape_: KShape

    def __init__(
        self,
        window_size: Union[str, int],
        sequence_length_multiplier: float = 4,
        overlap_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(Supervision.UNSUPERVISED)

        check_is_valid_window_size(window_size)

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
        KShape(**kwargs)

        self.window_size = window_size
        self.sequence_length_multiplier = sequence_length_multiplier
        self.overlap_rate = overlap_rate
        self.kwargs = kwargs

    def theta_(self) -> List[Tuple[np.array, float]]:
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
        if (
            not hasattr(self, "kshape_")
            or not hasattr(self, "window_size_")
            or not hasattr(self, "centroids_")
            or not hasattr(self, "weights_")
        ):
            raise NotFittedError("Call the fit function before making predictions!")
        return list(zip(self.centroids_, self.weights_))

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> "BaseDetector":
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")

        # Compute the window size
        X = np.asarray(X)
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)

        # Compute sliding windows
        sequence_length = int(self.window_size_ * self.sequence_length_multiplier)
        stride = int(sequence_length * self.overlap_rate)
        windows = sliding_window(X, sequence_length, stride)

        # Apply K-Shape clustering
        self.kshape_ = KShape(**self.kwargs)
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

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not utils.is_valid_array_like(X):
            raise ValueError(f"Input must be numerical array-like")
        if (
            not hasattr(self, "kshape_")
            or not hasattr(self, "window_size_")
            or not hasattr(self, "centroids_")
            or not hasattr(self, "weights_")
        ):
            raise NotFittedError("Call the fit function before making predictions!")

        # Make sure X is a numpy array
        X = np.asarray(X)

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
