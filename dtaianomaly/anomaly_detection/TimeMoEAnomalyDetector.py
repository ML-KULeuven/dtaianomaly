from typing import Literal

import numpy as np
import torch

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import (
    check_is_valid_window_size,
    compute_window_size,
)


class TimeMoEAnomalyDetector(BaseDetector):
    """
    Detect anomalies using the Time-MoE foundation model :cite:`shi2025timemoe`.

    Time-MoE is a decoder-only time series foundation model based on classical
    transformers, but in which the dense layers are replaced by a mixture of
    experts. This enables the model to automatically select and activate the
    most relevant experts for the given time series characteristics. Time-MoE
    is used to forecast windows in the time series, after which anomalies are
    detected based on the mean squared error with the actual observations.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    model_path: {'TimeMoE-50M', 'TimeMoE-200M' default='TimeMoE-50M'
        The Time-MoE model to use.
    batch_size: int, default=16
        The number of windows to feed simultaneously to Chronos, within a batch.
    prediction_length: int, default=1
        The number of samples to predict for each window.
    normalize_sequences: bool, default=True
        Whether each sequence must be normalized before feeding it Time-MoE.
    min_std: float, default=1e-8
        The lowest possible standard deviation to use for normalization.
    device: str, default='cpu'
        The device to use.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    time_moe_ : transformers.AutoModelForCausalLM
        The Time-MoE model used for forecasting the time series

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import TimeMoEAnomalyDetector
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> time_moe = TimeMoEAnomalyDetector(10).fit(x)
    >>> time_moe.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.00027719, 0.00027719, 0.00027719, ..., 0.00058781, 0.02628242,
           0.00010728]...)

    Notes
    -----
    - TimeMoEAnomalyDetector only handles univariate time series.
    - The max_position_embeddings for Time-MoE is set to during training.
      This means the maximum sequence length for Time-MoE is 4096. To
      achieve optimal forecasting performance, it is recommended that the
      sum of window_size_ and prediction_length does not exceed 4096.
    """

    window_size: int | str
    model_path: Literal["TimeMoE-50M", "TimeMoE-200M"]
    batch_size: int
    prediction_length: int
    normalize_sequences: bool
    min_std: float
    device: str

    window_size_: int
    time_moe_: any

    def __init__(
        self,
        window_size: int | str,
        model_path: Literal["TimeMoE-50M", "TimeMoE-200M"] = "TimeMoE-50M",
        batch_size: int = 16,
        prediction_length: int = 1,
        normalize_sequences: bool = True,
        min_std: float = 1e-8,
        device: str = "cpu",
    ):
        super().__init__(Supervision.UNSUPERVISED)

        try:
            import transformers
        except ImportError:
            raise Exception(
                "Module 'transformers' is not available, make sure you install it before using Time-MoE!"
            )

        check_is_valid_window_size(window_size)

        if not isinstance(model_path, str):
            raise TypeError("The 'model_path' must be a string!")
        if model_path not in ["TimeMoE-50M", "TimeMoE-200M"]:
            raise ValueError(
                f"The given 'model_path' is not valid. Received {model_path}, but must be one of ['TimeMoE-50M', 'TimeMoE-200M']."
            )

        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("`batch_size` should be an integer")
        if batch_size < 1:
            raise ValueError("`batch_size` should be strictly positive")

        if not isinstance(prediction_length, int) or isinstance(
            prediction_length, bool
        ):
            raise TypeError("`prediction_length` should be an integer")
        if prediction_length < 1:
            raise ValueError("`prediction_length` should be strictly positive")

        if not isinstance(normalize_sequences, bool):
            raise TypeError("`normalize_sequences` should be a bool")

        if not isinstance(min_std, (float, int)) or isinstance(min_std, bool):
            raise TypeError("`min_std` should be an integer")
        if min_std <= 0:
            raise ValueError(
                f"Variable 'min_std' must be at least 0, received '{min_std}'!"
            )

        torch.device(device)

        self.window_size = window_size
        self.model_path = model_path
        self.batch_size = batch_size
        self.prediction_length = prediction_length
        self.normalize_sequences = normalize_sequences
        self.min_std = min_std
        self.device = device

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:

        # Check if the given dataset is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")

        # Make sure the time series array has only one dimension
        X = X.squeeze()

        # Compute the window size
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)

        from transformers import AutoModelForCausalLM

        self.time_moe_ = AutoModelForCausalLM.from_pretrained(
            f"Maple728/{self.model_path}",
            device_map=self.device,
            trust_remote_code=True,
        )

    def _decision_function(self, X: np.ndarray) -> np.array:

        decision_scores = np.empty(X.shape[0])
        decision_scores = np.full_like(decision_scores, np.nan)

        batches = self._get_batch_starts(X.shape[0])

        for batch_starts in batches:

            # Create the batch
            batch = torch.tensor(
                np.array([X[i : i + self.window_size_] for i in batch_starts]),
                dtype=torch.float32,
            ).to(self.device)

            # Apply normalization
            if self.normalize_sequences:
                mean, std = batch.mean(dim=-1, keepdim=True), batch.std(
                    dim=-1, keepdim=True
                )
                std_for_division = torch.where(std < self.min_std, 1, std)
                batch = (batch - mean) / std_for_division

            # Use Time-MoE to make the forecasts
            forecasts = self.time_moe_.generate(
                batch, max_new_tokens=self.prediction_length
            )[:, -self.prediction_length :]

            # Reverse the normalization
            if self.normalize_sequences:
                forecasts = forecasts * std_for_division + mean

            # Extract the expected values
            batch_expected = np.array(
                [
                    X[
                        i
                        + self.window_size_ : i
                        + self.window_size_
                        + self.prediction_length
                    ]
                    for i in batch_starts
                ]
            )

            # Convert the forecasts to a numpy array
            forecasts = forecasts.to("cpu").numpy()

            # Compute the mean squared error
            decision_scores[np.array(batch_starts) + self.window_size_] = np.mean(
                (forecasts - batch_expected) ** 2, axis=1
            )

        # Padding
        decision_scores[: self.window_size_] = decision_scores[self.window_size_]
        decision_scores[-self.prediction_length + 1 :] = decision_scores[
            -self.prediction_length
        ]

        return decision_scores

    def _get_batch_starts(self, length_time_series: int):
        start_batches = [[]]
        for t in range(
            length_time_series - self.prediction_length - self.window_size_ + 1
        ):
            if len(start_batches[-1]) >= self.batch_size:
                start_batches.append([])
            start_batches[-1].append(t)
        return start_batches
