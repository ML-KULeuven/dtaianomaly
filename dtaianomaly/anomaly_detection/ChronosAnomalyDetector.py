import os
import tempfile
from typing import Literal

import numpy as np
import pandas as pd
import torch

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import (
    check_is_valid_window_size,
    compute_window_size,
)

_CHRONOS_MODEL_PATH_TYPE = Literal[
    "tiny",
    "mini",
    "small",
    "base",
    "large",
    "bolt_tiny, bolt_mini",
    "bolt_small",
    "bolt_large",
]
_VALID_CHRONOS_MODEL_PATH = [
    "tiny",
    "mini",
    "small",
    "base",
    "large",
    "bolt_tiny",
    "bolt_mini",
    "bolt_small",
    "bolt_large",
]


class ChronosAnomalyDetector(BaseDetector):
    """
    Detect anomalies in time series using Chronos :cite:`ansari2024chronos`.

    Chronos is pre-trained time series foundation model for time series forecasting.
    By computing the difference between the forecasted values and the actual observations,
    Chronos is able to detecat anomalies in time series. Chronos first bins the
    incoming time series sequences to obtain a finite set of values, after which
    an encoder-decoder model is used to forecast the sequence. The network is trained
    with a cross-entropy loss, using a combination of real, synthetic and semi-synthetic
    data.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    model_path: {'tiny', 'mini', 'small', 'base', 'large', 'bolt_tiny', 'bolt_mini', 'bolt_small', 'bolt_large'}, default='bolt_small'
        The Chronos-model to use for forecasting.
    batch_size: int, default=16
        The number of windows to feed simultaneously to Chronos, within a batch.
    forecast_horizon: int, default=1
        The number of samples to forecast for each window.
    do_fine_tuning: bool, default=False
        Whether to fine tune the model during fitting. If False, then the model will
        perform zero-shot forecasting.
    fine_tune_kwargs: dict, default=None
        Additional arguments for finetuning Chronos. Check out https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-model-zoo.html#autogluon.timeseries.models.ChronosModel
        for more information on the options.
    device: str, default='cpu'
        The device to use for running Chronos.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    chronos_ : autogluon.timeseries.TimeSeriesPredictor
        The Chronos model used for forecasting the time series

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import ChronosAnomalyDetector
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> chronos = ChronosAnomalyDetector(10).fit(x)
    >>> chronos.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.00027719, 0.00027719, 0.00027719, ..., 0.00058781, 0.02628242,
           0.00010728]...)

    Notes
    -----
    ChronosAnomalyDetector only handles univariate time series.
    """

    window_size: int | str
    model_path: _CHRONOS_MODEL_PATH_TYPE
    batch_size: int
    forecast_horizon: int
    do_fine_tuning: bool
    fine_tune_kwargs: dict[str, any] | None
    device: str

    window_size_: int
    chronos_: any

    def __init__(
        self,
        window_size: int | str,
        model_path: _CHRONOS_MODEL_PATH_TYPE = "bolt_small",
        batch_size: int = 16,
        forecast_horizon: int = 1,
        do_fine_tuning: bool = False,
        fine_tune_kwargs: dict[str, any] | None = None,
        device: str = "cpu",
    ):
        super().__init__(Supervision.UNSUPERVISED)

        try:
            import autogluon.timeseries
        except ImportError:
            raise Exception(
                "Module 'autogluon.timeseries' is not available, make sure you install it before using Chronos!"
            )

        check_is_valid_window_size(window_size)

        if not isinstance(model_path, str):
            raise TypeError("The 'model_path' must be a string!")
        if model_path not in _VALID_CHRONOS_MODEL_PATH:
            raise ValueError(
                f"The given 'model_path' is not valid. Received {model_path}, but must be one of {_VALID_CHRONOS_MODEL_PATH}"
            )

        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("`batch_size` should be an integer")
        if batch_size < 1:
            raise ValueError("`batch_size` should be strictly positive")

        if not isinstance(forecast_horizon, int) or isinstance(forecast_horizon, bool):
            raise TypeError("`forecast_horizon` should be an integer")
        if forecast_horizon < 1:
            raise ValueError("`forecast_horizon` should be strictly positive")

        if not isinstance(do_fine_tuning, bool):
            raise TypeError("`do_fine_tuning` should be a bool")

        torch.device(device)

        self.window_size = window_size
        self.model_path = model_path
        self.batch_size = batch_size
        self.forecast_horizon = forecast_horizon
        self.do_fine_tuning = do_fine_tuning
        self.fine_tune_kwargs = fine_tune_kwargs
        self.device = device

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:

        # Check if the given dataset is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")

        # Compute the window size
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)

        # Initialize the hyperparameters
        hyperparameters = {
            "model_path": self.model_path,
            "device": self.device,
            "batch_size": self.batch_size,
        }

        # Enable fine-tuning
        if self.do_fine_tuning:
            hyperparameters["fine_tune"] = True
            hyperparameters.update(self.fine_tune_kwargs or {})

        # Initialize the chronos model
        from autogluon.timeseries import TimeSeriesPredictor

        tmp_dir = tempfile.mkdtemp(prefix="chronos")
        os.rmdir(tmp_dir)  # To prevent warning that the dir already exists
        self.chronos_ = TimeSeriesPredictor(
            prediction_length=self.forecast_horizon, path=tmp_dir
        )
        self.chronos_.fit(
            self._format_data(X),
            hyperparameters={"Chronos": hyperparameters},
            skip_model_selection=True,
            enable_ensemble=False,
            verbosity=0,
        )

    def _decision_function(self, X: np.ndarray) -> np.array:

        # Check if the given dataset is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")

        forecasts = self.make_forecasts(X)
        decision_scores = np.empty(shape=X.shape[0])
        for group, df in forecasts.groupby("item_id"):
            forecast = df.sort_index()["mean"].values
            actual = X[
                group
                + self.window_size_ : group
                + self.window_size_
                + self.forecast_horizon
            ]
            error = np.mean((forecast - actual) ** 2)
            decision_scores[group + self.window_size_] = error
        decision_scores[: self.window_size_] = decision_scores[self.window_size_]

        if self.forecast_horizon > 1:
            decision_scores[-self.forecast_horizon + 1 :] = decision_scores[
                -self.forecast_horizon
            ]
        return decision_scores

    def make_forecasts(self, X: np.ndarray):
        return self.chronos_.predict(self._format_data(X))

    def _format_data(self, X: np.ndarray):
        records = []
        for i in range(X.shape[0] - self.window_size_ - self.forecast_horizon + 1):
            input_window = X[i : i + self.window_size_ + self.forecast_horizon]
            records.extend([(i, j, val) for j, val in enumerate(input_window)])
        df = pd.DataFrame(records, columns=["item_id", "timestamp", "target"])

        from autogluon.timeseries import TimeSeriesDataFrame

        return TimeSeriesDataFrame.from_data_frame(df)
