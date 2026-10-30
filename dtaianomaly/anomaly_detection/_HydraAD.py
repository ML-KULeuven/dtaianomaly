from typing import Literal

import numba as nb
import numpy as np
import torch
from hydraad._Hydra import Hydra
from hydraad._SparseScaler import SparseScaler
from sklearn.preprocessing import StandardScaler

from dtaianomaly.anomaly_detection import BaseDetector, Supervision
from dtaianomaly.utils import get_dimension, is_univariate
from dtaianomaly.windowing import WINDOW_SIZE_TYPE, compute_window_size

__all__ = ["HydraAD"]


class HydraAD(BaseDetector):
    window_size: WINDOW_SIZE_TYPE
    periodicity: float
    stride: int | float
    quantile: float
    scaling: Literal["sparse", "standard", "none"]
    # Hydra-related parameters
    n_kernels: int
    n_groups: int
    dilations: list[int] | Literal["auto"]
    first_order_difference: bool
    kernel_size: int
    relative_n_channels: float
    max_n_channels: int | None
    device: torch.device | None
    seed: int | None

    hydra_: Hydra

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        periodicity: float = 1.0,
        stride: int | float = 0.25,
        quantile: float = 0.9,
        scaling: Literal["sparse", "standard", "none"] = "sparse",
        n_kernels: int = 4,
        n_groups: int = 32,
        dilations: list[int] | Literal["auto"] = "auto",
        first_order_difference: bool = True,
        kernel_size: int = 9,
        relative_n_channels: float = 0.3,
        max_n_channels: int | None = None,
        device: torch.device | None = None,
        seed: int | None = None,
    ):
        super().__init__(Supervision.UNSUPERVISED)
        self.window_size = window_size
        self.periodicity = periodicity
        if isinstance(stride, float):
            assert 0 <= stride <= 1, (
                "When `stride` is a float it is the fraction of the window size "
                "and must be between 0 and 1."
            )
        self.stride = stride
        self.quantile = quantile
        self.scaling = scaling

        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.dilations = dilations
        self.first_order_difference = first_order_difference
        self.kernel_size = kernel_size
        self.relative_n_channels = relative_n_channels
        self.max_n_channels = max_n_channels
        self.device = device
        self.seed = seed

    @property
    def stride_(self) -> int:
        if isinstance(self.stride, int):
            return self.stride
        return max(1, int(self.window_size_ * self.stride))

    def embed(self, X: np.ndarray) -> torch.Tensor:
        X_tensor = torch.as_tensor(X, dtype=torch.float32).to(self.device)

        # Shape required by Hydra.forward: (n_channels, T)
        if X.ndim == 1:
            X_tensor = X_tensor.unsqueeze(0)  # (1, T)
        else:
            X_tensor = X_tensor.T  # (n_channels, T)

        with torch.no_grad():
            activations = self.hydra_.forward(X_tensor)
            soft_counts = self.hydra_.soft_counts(activations)
            embedding = self._aggregate_windows(
                soft_counts, self.window_size_, self.stride_
            )  # (n_windows, n_features)

        # Scaling
        if self.scaling == "sparse":
            embedding = SparseScaler().fit_transform(embedding)
        if self.scaling == "standard":
            embedding = torch.Tensor(StandardScaler().fit_transform(embedding))

        return embedding

    def _aggregate_windows(
        self, counts: torch.Tensor, window_size: int, stride: int
    ) -> torch.Tensor:
        """Aggregate per-timestep counts into per-window counts.

        Sums the per-timestep soft/hard counts over each sliding window.
        Summation reproduces both count types: soft counts sum the argmax
        magnitudes and hard counts sum the argmin occurrences, exactly as the
        (windowing) Hydra transform did via ``scatter_add_``.

        Parameters
        ----------
        counts : torch.Tensor
            Per-timestep counts of shape ``(T, F)`` from
            :meth:`Hydra.soft_counts` / :meth:`Hydra.hard_counts`.
        window_size : int
            Length of each window along the time axis.
        stride : int
            Step between consecutive windows.

        Returns
        -------
        torch.Tensor
            Per-window counts of shape ``(n_windows, F)`` where
            ``n_windows = (T - window_size) // stride + 1``. Trailing timesteps
            that do not fill a full window are dropped.
        """
        return counts.unfold(0, window_size, stride).sum(-1)

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        window_size_ = int(self._compute_window_size(X, **kwargs) * self.periodicity)
        self.window_size_ = max(self.kernel_size, window_size_)

        if self.dilations == "auto":
            max_exponent = np.log2((self.window_size_ - 1) / (self.kernel_size - 1))
            dilations = (2 ** torch.arange(int(max_exponent) + 1)).tolist()
        else:
            dilations = self.dilations

        # Hydra
        self.hydra_ = (
            Hydra(
                n_kernels=self.n_kernels,
                n_groups=self.n_groups,
                dilations=dilations,
                first_order_difference=self.first_order_difference,
                kernel_size=self.kernel_size,
                relative_n_channels=self.relative_n_channels,
                max_n_channels=self.max_n_channels,
                seed=self.seed,
            )
            .initialize_weights(n_channels=get_dimension(X))
            .to(self.device)
        )

    def _decision_function(self, X: np.ndarray) -> np.array:
        # Create the embedding
        embedding = self.embed(X)

        # Decision scores
        decision_scores = (
            torch.quantile(embedding.abs(), self.quantile, dim=1).cpu().numpy()
        )
        if decision_scores.min() == decision_scores.max():
            decision_scores -= decision_scores.min()

        return _reverse_sliding_window(
            decision_scores, self.window_size_, self.stride_, X.shape[0]
        )

    def _compute_window_size(self, X: np.ndarray, **kwargs) -> int:
        # Compute the window size
        if is_univariate(X):
            return int(compute_window_size(X, self.window_size, **kwargs))
        else:
            default_widow_size = (
                kwargs["default_window_size"] if "default_window_size" in kwargs else -1
            )
            kwargs["default_window_size"] = -1
            window_sizes = list(
                filter(
                    lambda i: i > 0,
                    [
                        compute_window_size(X[:, i], self.window_size, **kwargs)
                        for i in range(X.shape[1])
                    ],
                )
            )
            if len(window_sizes) == 0:
                assert (
                    default_widow_size != -1
                ), "Could not compute window size from the data."
                return kwargs["default_window_size"]

            else:
                return int(np.mean(window_sizes))


@nb.njit(fastmath=True)
def _reverse_sliding_window(
    per_window_anomaly_scores: np.ndarray,
    window_size: int,
    stride: int,
    length_time_series: int,
) -> np.ndarray:
    # Convert to array
    scores_time = np.empty(length_time_series)

    start_window_index = 0
    min_start_window = 0
    end_window_index = 0
    min_end_window = 0
    for t in range(length_time_series - window_size):
        while min_start_window + window_size <= t:
            start_window_index += 1
            min_start_window += stride
        while t >= min_end_window:
            end_window_index += 1
            min_end_window += stride
        scores_time[t] = np.mean(
            per_window_anomaly_scores[start_window_index:end_window_index]
        )

    for t in range(length_time_series - window_size, length_time_series):
        while min_start_window + window_size <= t:
            start_window_index += 1
            min_start_window += stride
        if start_window_index < per_window_anomaly_scores.shape[0]:
            scores_time[t] = np.mean(per_window_anomaly_scores[start_window_index:])
        else:
            scores_time[t] = 0.0

    return scores_time
