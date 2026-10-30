from typing import Literal

import numpy as np
import torch

from dtaianomaly.anomaly_detection import BaseDetector, Supervision
from dtaianomaly.utils import get_dimension, is_univariate
from dtaianomaly.windowing import (
    WINDOW_SIZE_TYPE,
    compute_window_size,
    reverse_sliding_window,
)

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

    hydra_: _Hydra

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
            embedding = _sparse_scaling(embedding)
        if self.scaling == "standard":
            embedding = _standard_scaling(embedding)

        return embedding

    def _aggregate_windows(
        self, counts: torch.Tensor, window_size: int, stride: int
    ) -> torch.Tensor:
        """Aggregate per-timestep counts into per-window counts.

        Sums the per-timestep soft counts over each sliding window, which
        sums the argmax magnitudes exactly as the (windowing) Hydra transform
        did via ``scatter_add_``.

        Parameters
        ----------
        counts : torch.Tensor
            Per-timestep counts of shape ``(T, F)`` from
            :meth:`_Hydra.soft_counts`.
        window_size : int
            Length of each window along the time axis.
        stride : int
            Step between consecutive windows.

        Returns
        -------
        torch.Tensor
            Per-window counts of shape ``(n_windows, F)``. As in
            :py:func:`~dtaianomaly.windowing.sliding_window`, a final window
            ending at the last timestep is added if the stride does not
            align with the end of the series.
        """
        windows = counts.unfold(0, window_size, stride).sum(-1)
        if (counts.shape[0] - window_size) % stride != 0:
            windows = torch.cat([windows, counts[-window_size:].sum(0, keepdim=True)])
        return windows

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
            _Hydra(
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

        return reverse_sliding_window(
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


class _Hydra(torch.nn.Module):
    """
    Adaptation of the Hydra transform :cite:`dempster2023hydra`, which computes
    per-timestep activations instead of per-series counts.

    Adapted from https://github.com/angus924/hydra/blob/main/code/hydra.py.
    """

    dilations: int | list[int]
    n_kernels: int
    n_groups: int
    first_order_difference: bool
    kernel_size: int
    relative_n_channels: float
    max_n_channels: int | None
    seed: int | None

    n_dilations_: int
    num_channels_per_: int
    n_features_: int
    W_: torch.Tensor  # (D, n_groups, n_kernels, kernel_size)
    I_: torch.Tensor  # (D, n_groups, num_channels_per) sampled channels per group
    dilations_: torch.Tensor  # (D,) dilation per pass
    paddings_: torch.Tensor  # (D,) conv padding per pass
    group_use_diff_: (
        torch.Tensor
    )  # (n_groups,) bool: whether each group uses the diff series

    def __init__(
        self,
        n_kernels: int = 8,
        n_groups: int = 64,
        dilations: int | list[int] = 1,
        first_order_difference: bool = True,
        kernel_size: int = 9,
        relative_n_channels: float = 0.3,
        max_n_channels: int | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.dilations = dilations
        self.first_order_difference = first_order_difference
        self.kernel_size = kernel_size
        self.relative_n_channels = relative_n_channels
        self.max_n_channels = max_n_channels
        self.seed = seed

    @property
    def n_features_(self) -> int:
        return self.n_dilations_ * self.n_groups * self.n_kernels

    def initialize_weights(self, n_channels: int) -> "_Hydra":
        for attr in ["W_", "I_", "dilations_", "paddings_", "group_use_diff_"]:
            if not hasattr(self, attr):
                self.register_buffer(attr, None)

        if self.seed is not None:
            torch.manual_seed(self.seed)

        if isinstance(self.dilations, int):
            self.dilations_ = torch.IntTensor([self.dilations])
        else:
            self.dilations_ = torch.IntTensor(self.dilations)
        self.n_dilations_ = len(self.dilations_)

        self.paddings_ = torch.div(
            (self.kernel_size - 1) * self.dilations_, 2, rounding_mode="floor"
        ).int()

        # Kernels with explicit (dilation, group, kernel) axes; the
        # in_channels/groups dimension required by conv1d is added on the fly in
        # forward() rather than stored here.
        self.W_ = torch.randn(
            self.n_dilations_, self.n_groups, self.n_kernels, self.kernel_size
        )
        self.W_ = self.W_ - self.W_.mean(-1, keepdim=True)
        self.W_ = self.W_ / self.W_.abs().sum(-1, keepdim=True)

        # Subspaces: channels sampled per (dilation, group).
        self.num_channels_per_ = int(
            np.clip(int(n_channels * self.relative_n_channels), 1, self.max_n_channels)
        )
        self.I_ = torch.randint(
            0, n_channels, (self.n_dilations_, self.n_groups, self.num_channels_per_)
        )

        # The last n_groups // 2 groups convolve the diff series; the rest the
        # raw series. Disabled when first_order_difference is off or there is
        # only a single group.
        self.group_use_diff_ = torch.zeros(self.n_groups, dtype=torch.bool)
        if self.first_order_difference and self.n_groups > 1:
            self.group_use_diff_[self.n_groups - self.n_groups // 2 :] = True

        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.unsqueeze(0)  # (1, n_channels, T)
        n_channels = X.shape[1]
        T = X.shape[-1]
        diff_X = torch.diff(X, dim=-1) if bool(self.group_use_diff_.any()) else None

        raw_groups = ~self.group_use_diff_
        diff_groups = self.group_use_diff_

        per_dilation = []
        for d in range(self.n_dilations_):
            dilation = self.dilations_[d].item()
            padding = self.paddings_[d].item()

            # Raw groups first, diff groups last -- this matches the group order
            # in group_use_diff_, so a plain concat keeps the columns in
            # ascending group order (no scatter needed).
            blocks = []
            for group_mask, X_in in ((raw_groups, X), (diff_groups, diff_X)):
                if not bool(group_mask.any()):
                    continue
                _Z = self._convolve_block(
                    X_in, group_mask, d, dilation, padding, n_channels
                )
                pad = T - _Z.shape[0]
                if pad > 0:
                    _Z = torch.nn.functional.pad(_Z, (0, 0, pad, 0))
                blocks.append(_Z)

            per_dilation.append(torch.cat(blocks, dim=1))

        return torch.cat(per_dilation, dim=1)  # (T, F)

    def _convolve_block(
        self,
        X_in: torch.Tensor,
        group_mask: torch.Tensor,
        d: int,
        dilation: int,
        padding: int,
        n_channels: int,
    ) -> torch.Tensor:
        W_block = self.W_[d, group_mask]
        n_block = W_block.shape[0]
        weight = W_block.reshape(n_block * self.n_kernels, 1, self.kernel_size)

        if n_channels > 1:
            I_block = self.I_[d, group_mask]
            _Z_full = torch.nn.functional.conv1d(
                X_in[:, I_block].sum(2),
                weight,
                dilation=dilation,
                padding=padding,
                groups=n_block,
            )
        else:
            _Z_full = torch.nn.functional.conv1d(
                X_in,
                weight,
                dilation=dilation,
                padding=padding,
            )

        return _Z_full.squeeze(0).T

    def soft_counts(self, activations: torch.Tensor) -> torch.Tensor:
        blocks = self._as_blocks(activations)
        mv, mi = blocks.max(dim=-1)
        out = torch.zeros_like(blocks)
        out.scatter_(-1, mi.unsqueeze(-1), mv.unsqueeze(-1))
        return out.reshape(activations.shape)

    def _as_blocks(self, activations: torch.Tensor) -> torch.Tensor:
        return activations.reshape(
            activations.shape[0],
            self.n_dilations_,
            self.n_groups,
            self.n_kernels,
        )


def _sparse_scaling(X: torch.Tensor, exponent: int = 4) -> torch.Tensor:
    """
    Adaptation of standard scaling to deal with sparse data :cite:`dempster2023hydra`.

    Applies a square root transformation and standardizes each feature, adding an
    epsilon to the standard deviation that grows with the sparsity of the feature.
    Zero values remain zero after the transformation.

    Adapted from https://github.com/angus924/hydra/blob/main/code/hydra.py.
    """
    X = X.clamp(0).sqrt()
    epsilon = (X == 0).float().mean(0) ** exponent + 1e-8
    mu = X.mean(0)
    sigma = X.std(0) + epsilon
    return ((X - mu) * (X != 0)) / sigma


def _standard_scaling(X: torch.Tensor) -> torch.Tensor:
    """
    Standardize each feature to zero mean and unit variance, equivalent to
    :py:class:`sklearn.preprocessing.StandardScaler`. Features with zero
    variance are only centered.
    """
    sigma = X.std(0, correction=0)
    sigma[sigma == 0] = 1.0
    return (X - X.mean(0)) / sigma
