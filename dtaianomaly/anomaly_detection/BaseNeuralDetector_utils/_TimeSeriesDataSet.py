import numpy as np
import torch
from sklearn.preprocessing import scale


class TimeSeriesDataset(torch.utils.data.Dataset):

    X: np.ndarray
    window_size: int
    stride: int
    standard_scaling: bool
    device: str

    def __init__(
        self,
        X: np.ndarray,
        window_size: int,
        stride: int,
        standard_scaling: bool,
        device: str,
    ):
        self.X = X
        self.window_size = window_size
        self.stride = stride
        self.standard_scaling = standard_scaling
        self.device = device

    def _scale(self, window: np.ndarray) -> np.ndarray:
        if self.standard_scaling:
            window = scale(window)
        return window


class ForecastDataset(TimeSeriesDataset):

    forecast_length: int

    def __init__(
        self,
        X: np.ndarray,
        window_size: int,
        stride: int,
        standard_scaling: bool,
        device: str,
        forecast_length: int,
    ):
        super().__init__(X, window_size, stride, standard_scaling, device)
        self.forecast_length = forecast_length

    def __len__(self) -> int:
        return (
            int(
                np.ceil(
                    (self.X.shape[0] - self.forecast_length - self.window_size)
                    / self.stride
                )
            )
            + 1
        )

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        # Find the indices
        start = idx * self.stride
        end = start + self.window_size + self.forecast_length

        # Handle the last window
        if end >= self.X.shape[0]:
            end = self.X.shape[0]
            start = end - self.window_size - self.forecast_length

        # Retrieve the window
        window = self._scale(self.X[start:end])

        # Split in history and future
        history = window[: self.window_size].reshape(-1)
        future = window[-self.forecast_length :].reshape(-1)

        # Return the data
        return [
            torch.Tensor(history).to(self.device),
            torch.Tensor(future).to(self.device),
        ]


class ReconstructionDataset(TimeSeriesDataset):

    def __len__(self) -> int:
        return int(np.ceil((self.X.shape[0] - self.window_size) / self.stride)) + 1

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        start = idx * self.stride
        end = start + self.window_size

        # Handle the last window
        if end >= self.X.shape[0]:
            end = self.X.shape[0]
            start = end - self.window_size

        # Retrieve the window
        window = self._scale(self.X[start:end]).reshape(-1)

        # Return the data
        return [torch.Tensor(window).to(self.device)]
