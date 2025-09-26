from typing import Literal

import numpy as np
import torch

from dtaianomaly import utils
from dtaianomaly.anomaly_detection._BaseDetector import BaseDetector, Supervision
from dtaianomaly.type_validation import (
    BoolAttribute,
    FloatAttribute,
    IntegerAttribute,
    LiteralAttribute,
    WindowSizeAttribute,
)
from dtaianomaly.windowing import WINDOW_SIZE_TYPE, compute_window_size

__all__ = ["TimeMoE", "MODEL_PATHS"]

MODEL_PATH_TYPE = Literal["TimeMoE-50M", "TimeMoE-200M"]
MODEL_PATHS = ["TimeMoE-50M", "TimeMoE-200M"]


class TimeMoE(BaseDetector):
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
    >>> from dtaianomaly.anomaly_detection import TimeMoE
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> time_moe = TimeMoE(10).fit(x)
    >>> time_moe.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE +SKIP
    array([6.34949149e-05, 6.34949149e-05, 6.34949149e-05, ...,
           6.34949149e-05, 6.34949149e-05, 6.34949149e-05]...)

    Warnings
    --------
    If you want to run Time-MoE, be sure to install the optional dependency ``time_moe``:

    .. code-block:: bash

        pip install dtaianomaly[time_moe]

    Notes
    -----
    - TimeMoE only handles univariate time series.
    - The max_position_embeddings for Time-MoE is set to during training.
      This means the maximum sequence length for Time-MoE is 4096. To
      achieve optimal forecasting performance, it is recommended that the
      sum of ``window_size_`` and ``prediction_length`` does not exceed 4096.
    """

    window_size: WINDOW_SIZE_TYPE
    model_path: MODEL_PATH_TYPE
    batch_size: int
    prediction_length: int
    normalize_sequences: bool
    min_std: float
    device: str

    window_size_: int
    time_moe_: any

    attribute_validation = {
        "window_size": WindowSizeAttribute(),
        "model_path": LiteralAttribute(MODEL_PATHS),
        "batch_size": IntegerAttribute(1),
        "prediction_length": IntegerAttribute(1),
        "normalize_sequences": BoolAttribute(),
        "min_std": FloatAttribute(0.0),
    }

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        model_path: MODEL_PATH_TYPE = "TimeMoE-50M",
        batch_size: int = 16,
        prediction_length: int = 1,
        normalize_sequences: bool = True,
        min_std: float = 1e-8,
        device: str = "cpu",
    ):
        try:
            import transformers
        except ImportError:
            raise Exception(
                "Module 'transformers' is not available, make sure you install it before using Time-MoE!"
            )

        super().__init__(Supervision.UNSUPERVISED)
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

        # Check if the given dataset is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")

        # Make sure the time series array has only one dimension
        X = X.squeeze()

        decision_scores = np.empty(X.shape[0])
        for batch_starts in self._get_batch_starts(X.shape[0]):

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


def main():
    from dtaianomaly.data import demonstration_time_series

    X, y = demonstration_time_series()
    TimeMoEAnomalyDetector(64).fit(X).decision_function(X)


if __name__ == "__main__":

    main()

    import doctest

    doctest.testmod()
