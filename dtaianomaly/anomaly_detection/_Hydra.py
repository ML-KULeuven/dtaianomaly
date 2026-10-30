import numpy as np
import torch

__all__ = ["Hydra"]


class Hydra(torch.nn.Module):
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

    def fit(self, X: torch.Tensor, y=None) -> "Hydra":
        return self.initialize_weights(X.shape[0])

    def initialize_weights(self, n_channels: int) -> "Hydra":
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

    def hard_counts(self, activations: torch.Tensor) -> torch.Tensor:
        blocks = self._as_blocks(activations)
        mi = blocks.min(dim=-1).indices
        out = torch.zeros_like(blocks)
        out.scatter_(
            -1,
            mi.unsqueeze(-1),
            torch.ones(mi.unsqueeze(-1).shape, dtype=out.dtype, device=out.device),
        )
        out[0, :, self.group_use_diff_, :] = 0.0
        return out.reshape(activations.shape)

    def _as_blocks(self, activations: torch.Tensor) -> torch.Tensor:
        return activations.reshape(
            activations.shape[0],
            self.n_dilations_,
            self.n_groups,
            self.n_kernels,
        )
