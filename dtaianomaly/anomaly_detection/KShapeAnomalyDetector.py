from typing import List, Optional, Tuple, Union

import numpy as np
import stumpy
from scipy.spatial.distance import pdist, squareform
from sklearn.exceptions import NotFittedError
from tslearn.clustering import KShape

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import (
    compute_window_size,
    reverse_sliding_window,
    sliding_window,
)


class KShapeAnomalyDetector(BaseDetector):

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
        overlap_rate: float = 0.25,
        **kwargs,
    ):
        super().__init__(Supervision.SUPERVISED)
        # TODO checks on input

        self.window_size = window_size
        self.sequence_length_multiplier = sequence_length_multiplier
        self.overlap_rate = overlap_rate  # Should be 0 < overlap_rate <= 1 (1 for non-overlapping sequences), not 0 because then stride of 0
        self.kwargs = kwargs

    @property
    def theta_(self) -> List[Tuple[np.array, float]]:
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
        self.centroids_ = list(
            map(_clean_cluster_tslearn, self.kshape_.cluster_centers_)
        )
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
            not hasattr(self, "window_size_")
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


def _clean_cluster_tslearn(cluster) -> np.array:  # TODO double check this method
    return np.array([val[0] for val in cluster])


def _shape_based_distance(x: np.array, y: np.array) -> float:
    ncc = _ncc_c(x, y)
    return 1 - ncc.max()


def _ncc_c(x: np.array, y: np.array):  # TODO double check this method
    den = np.array(np.linalg.norm(x) * np.linalg.norm(y))
    den[den == 0] = np.inf

    x_len = len(x)
    fft_size = 1 << (2 * x_len - 1).bit_length()
    cc = np.fft.ifft(np.fft.fft(x, fft_size) * np.conj(np.fft.fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len - 1) :], cc[:x_len]))
    return np.real(cc) / den


def main():
    from dtaianomaly.data import demonstration_time_series
    from dtaianomaly.visualization import plot_anomaly_scores

    X, y = demonstration_time_series()
    kshape = KShapeAnomalyDetector("fft", sequence_length_multiplier=2.5, n_clusters=6)
    kshape.fit(X)
    y_pred = kshape.predict_proba(X)
    plot_anomaly_scores(X, y, y_pred).show()


if __name__ == "__main__":
    main()
