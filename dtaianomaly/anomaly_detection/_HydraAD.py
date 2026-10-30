from typing import Literal

import numpy as np
import torch

from dtaianomaly.anomaly_detection._BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection._BaseNeuralDetector import check_device
from dtaianomaly.type_validation import (
    BoolAttribute,
    FloatAttribute,
    IntegerAttribute,
    ListAttribute,
    LiteralAttribute,
    NoneAttribute,
    WindowSizeAttribute,
)
from dtaianomaly.utils import get_dimension, is_univariate
from dtaianomaly.windowing import (
    WINDOW_SIZE_TYPE,
    check_stride,
    compute_window_size,
    reverse_sliding_window,
)

__all__ = ["HydraAD"]


class HydraAD(BaseDetector):
    """
    Detect anomalies with competing random kernels using HydraAD :cite:`carpentier2026hydraad`.

    HydraAD adapts the Hydra transform :cite:`dempster2023hydra` for time series
    classification to anomaly detection. The time series is convolved with random
    kernels, which are organised into groups. At each time step, the kernels within
    a group compete with each other: only the kernel with the largest activation
    keeps it, while all other activations are set to zero. These activations are
    summed within sliding windows to obtain an embedding of each window. After
    scaling the embedding, the anomaly score of a window equals a quantile of the
    absolute values in its embedding. Hence, a window is anomalous if it strongly
    activates the kernels in comparison with the other windows.

    Parameters
    ----------
    window_size : int or str
        The window size, the length of the subsequences that will be detected as anomalies. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
        For multivariate time series, the window size is computed for each attribute separately,
        after which the average window size is used.
    periodicity : float, default=1.0
        The factor with which the computed window size is multiplied. The window size is
        at least ``kernel_size``.
    stride : int or float, default=0.25
        The stride, i.e., the step size for extracting sliding windows from the time series.
        If a float is given, the stride equals this fraction of the window size.
    quantile : float, default=0.9
        The quantile of the absolute embedding values used as anomaly score of a window.
    scaling : {'sparse', 'standard', 'none'}, default='sparse'
        How the embedding is scaled before computing the anomaly scores. Valid options are:

        - ``'sparse'``: Sparse scaling as proposed by :cite:t:`dempster2023hydra`, which
          scales the square root of the features while taking their sparsity into account.
        - ``'standard'``: Standardize each feature to zero mean and unit variance.
        - ``'none'``: Do not scale the embedding.

    n_kernels : int, default=4
        The number of kernels in each group.
    n_groups : int, default=32
        The number of groups of competing kernels for each dilation.
    dilations : list of int or 'auto', default='auto'
        The dilations of the kernels. If ``'auto'``, the dilations are the powers of 2
        for which the receptive field of the kernel does not exceed the window size.
    first_order_difference : bool, default=True
        Whether half of the groups convolve the first order difference of the time
        series instead of the raw time series.
    kernel_size : int, default=9
        The length of each kernel.
    relative_n_channels : float, default=0.3
        The fraction of the channels of a multivariate time series that each group
        convolves. The selected channels are summed before the convolution.
    max_n_channels : int, default=None
        The maximum number of channels that each group convolves. If ``None``, the
        number of channels is not bounded.
    device : str, default="cpu"
        The device on which the convolutions are computed.
        For more information, see: https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-device.
    seed : int, default=None
        The random seed used to initialise the kernels and select the channels.

    Attributes
    ----------
    window_size_ : int
        The effectively used window size for detecting anomalies.
    stride_ : int
        The effectively used stride for detecting anomalies.
    hydra_ : torch.nn.Module
        The random convolutional kernels of the Hydra transform.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import HydraAD
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> hydra_ad = HydraAD(64, seed=0).fit(x)
    >>> hydra_ad.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE, +SKIP
    array([1.45365405, 1.45365405, 1.45365405, ..., 1.58905673, 1.58905673,
           1.58905673])
    """

    window_size: WINDOW_SIZE_TYPE
    periodicity: float
    stride: int | float
    quantile: float
    scaling: Literal["sparse", "standard", "none"]
    n_kernels: int
    n_groups: int
    dilations: list[int] | Literal["auto"]
    first_order_difference: bool
    kernel_size: int
    relative_n_channels: float
    max_n_channels: int | None
    device: str
    seed: int | None

    window_size_: int
    hydra_: "_Hydra"

    attribute_validation = {
        "window_size": WindowSizeAttribute(),
        "periodicity": FloatAttribute(minimum=0.0, inclusive_minimum=False),
        "stride": IntegerAttribute(1)
        | FloatAttribute(0.0, 1.0, inclusive_minimum=False),
        "quantile": FloatAttribute(0.0, 1.0),
        "scaling": LiteralAttribute(["sparse", "standard", "none"]),
        "n_kernels": IntegerAttribute(1),
        "n_groups": IntegerAttribute(1),
        "dilations": ListAttribute(IntegerAttribute(1), minimum_length=1)
        | LiteralAttribute("auto"),
        "first_order_difference": BoolAttribute(),
        "kernel_size": IntegerAttribute(2),
        "relative_n_channels": FloatAttribute(0.0, 1.0, inclusive_minimum=False),
        "max_n_channels": IntegerAttribute(1) | NoneAttribute(),
        "seed": IntegerAttribute() | NoneAttribute(),
    }

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
        device: str = "cpu",
        seed: int | None = None,
    ):
        super().__init__(Supervision.UNSUPERVISED)
        check_device(device)
        self.window_size = window_size
        self.periodicity = periodicity
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
        """
        Embed the sliding windows of the given time series.

        The time series is convolved with the random kernels, after which the
        activations of the winning kernels are summed within each sliding window.
        The resulting embedding is scaled as specified by ``scaling``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_attributes)
            The time series to embed.

        Returns
        -------
        torch.Tensor of shape (n_windows, n_features)
            The scaled embedding of each sliding window.
        """
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
        check_stride(self.window_size_, self.stride_)

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
        if is_univariate(X):
            return int(compute_window_size(X, self.window_size, **kwargs))

        # For multivariate time series, use the mean window size of the attributes
        default_window_size = kwargs.pop("default_window_size", None)
        window_sizes = [
            compute_window_size(
                X[:, i], self.window_size, default_window_size=-1, **kwargs
            )
            for i in range(X.shape[1])
        ]
        window_sizes = [window_size for window_size in window_sizes if window_size > 0]
        if len(window_sizes) > 0:
            return int(np.mean(window_sizes))
        if default_window_size is None:
            raise ValueError(
                f"Could not compute the window size using '{self.window_size}' for "
                f"any attribute of the multivariate time series!"
            )
        return default_window_size


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

        # Use a local generator to not change the global random state of torch
        generator = torch.Generator()
        if self.seed is None:
            generator.seed()
        else:
            generator.manual_seed(self.seed)

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
            self.n_dilations_,
            self.n_groups,
            self.n_kernels,
            self.kernel_size,
            generator=generator,
        )
        self.W_ = self.W_ - self.W_.mean(-1, keepdim=True)
        self.W_ = self.W_ / self.W_.abs().sum(-1, keepdim=True)

        # Subspaces: channels sampled per (dilation, group).
        self.num_channels_per_ = int(
            np.clip(int(n_channels * self.relative_n_channels), 1, self.max_n_channels)
        )
        self.I_ = torch.randint(
            0,
            n_channels,
            (self.n_dilations_, self.n_groups, self.num_channels_per_),
            generator=generator,
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
