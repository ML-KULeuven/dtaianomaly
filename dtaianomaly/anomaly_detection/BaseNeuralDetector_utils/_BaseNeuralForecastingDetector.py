import abc
from typing import Callable, Literal

import numpy as np
import torch

from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.BaseNeuralDetector import (
    _COMPILE_MODE_TYPE,
    _MODEL_PARAMETERS_TYPE,
    _OPTIMIZER_TYPE,
    BaseNeuralDetector,
)
from dtaianomaly.anomaly_detection.BaseNeuralDetector_utils._TimeSeriesDataSet import (
    ForecastDataset,
)


class BaseNeuralForecastingDetector(BaseNeuralDetector, abc.ABC):
    """
    Base class for forecasting-based neural anomaly detectors.

    Forecasting-based anomaly detection detect anomalies by measuring
    the difference of a forecasted value with the actually observed
    value. Specifically, the neural network takes as input a sliding
    window of the time series, and aims at predicting the future
    values. The assumption is that anomalies are much harder to
    forecast. Thus, the difference between the forecasted value and
    the observed value will be high for anomalies, but low for normal
    observations.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    supervision: Supervision, default=Supervision.SEMI_SUPERVISED
        The type of supervision this anomaly detector requires.
    error_metric: {"mean-absolute-error", "mean-squared-error"}, default="mean-absolute-error"
        The error measure between the forecasted value and the observed values.
    forecast_length: int default=1
        The number of time steps the neural network must forecast.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    standard_scaling: bool, default=True
        Whether to standard scale each window independently, before feeding it to the network.
    batch_size: int, default=32
        The size of the batches to feed to the network.
    data_loader_kwargs: dictionary, default=None
        Additional kwargs to be passed to the data loader.
        For more information, see: https://docs.pytorch.org/docs/stable/data.html
    optimizer: {"adam", "sgd"} or callable default="adam"
        The optimizer to use for learning the weights. If "adam" is given,
        then the torch.optim.Adam optimizer will be used. If "sgd" is given,
        then the torch.optim.SGD optimizer will be used. Otherwise, a callable
        should be given, which takes as input the network parameters, and then
        creates an optimizer.
    learning_rate: float, default=1e-3
        The learning rate to use for training the network. Has no effect
        if optimize is a callable.
    compile_model: bool, default=False
        Whether the network architecture should be compiled or not before
        training the weights.
        For more information, see: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
    compile_mode: {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}, default="default"
        Method to compile the architecture.
        For more information, see: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
    n_epochs: int, default=10
        The number of epochs for which the neural network should be trained.
    loss_function: torch.nn.Module, default=torch.nn.MSELoss()
        The loss function to use for updating the weights.
    device: str, default="cpu"
        The device on which te neural network should be trained.
        For more information, see: https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-device
    seed: int, default=None
        The seed used for training the model. This seed will update the torch
        and numpy seed at the beginning of the fit method.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector.
    optimizer_: torch.optim.Optimizer
        The optimizer used for learning the weights of the network.
    neural_network_: torch.nn.Module
        The PyTorch network architecture.

    See also
    --------
    MultilayerPerceptron: An implementation of this class using an feed-forward neural network.
    """

    error_metric: Literal["mean-absolute-error", "mean-squared-error"]
    forecast_length: int

    def __init__(
        self,
        window_size: str | int,
        supervision: Supervision = Supervision.SEMI_SUPERVISED,
        error_metric: Literal[
            "mean-absolute-error", "mean-squared-error"
        ] = "mean-absolute-error",
        forecast_length: int = 1,
        stride: int = 1,
        standard_scaling: bool = True,
        batch_size: int = 32,
        data_loader_kwargs: dict[str, any] = None,
        optimizer: (
            _OPTIMIZER_TYPE | Callable[[_MODEL_PARAMETERS_TYPE], torch.optim.Optimizer]
        ) = "adam",
        learning_rate: float = 1e-3,
        compile_model: bool = False,
        compile_mode: _COMPILE_MODE_TYPE = "default",
        n_epochs: int = 10,
        loss_function: torch.nn.Module = torch.nn.MSELoss(),
        device: str = "cpu",
        seed: int = None,
    ):
        super().__init__(
            supervision=supervision,
            window_size=window_size,
            stride=stride,
            standard_scaling=standard_scaling,
            batch_size=batch_size,
            data_loader_kwargs=data_loader_kwargs,
            optimizer=optimizer,
            learning_rate=learning_rate,
            compile_model=compile_model,
            compile_mode=compile_mode,
            n_epochs=n_epochs,
            loss_function=loss_function,
            device=device,
            seed=seed,
        )

        if not isinstance(error_metric, str):
            raise TypeError("`error_metric` should be a string")
        if error_metric not in ["mean-absolute-error", "mean-squared-error"]:
            raise ValueError(
                f"Unknown error_metric '{error_metric}'. Valid options are ['mean-absolute-error', 'mean-squared-error']"
            )

        if not isinstance(forecast_length, int) or isinstance(forecast_length, bool):
            raise TypeError("`forecast_length` should be an integer")
        if forecast_length < 1:
            raise ValueError("`forecast_length` should be strictly positive")

        self.error_metric = error_metric
        self.forecast_length = forecast_length

    def _build_dataset(self, X: np.ndarray) -> torch.utils.data.Dataset:
        return ForecastDataset(
            X=X,
            window_size=self.window_size_,
            stride=self.stride,
            standard_scaling=self.standard_scaling,
            device=self.device,
            forecast_length=self.forecast_length,
        )

    def _train_batch(self, batch: list[torch.Tensor]) -> float:

        # Set the type of the batch
        history, future = batch

        # Initialize the gradients to zero
        self.optimizer_.zero_grad()

        # Feed the data to the neural network
        forecast = self.neural_network_(history).reshape(future.shape)

        # Compute the loss
        loss = self.loss_function(forecast, future)

        # Compute the gradients of the loss
        loss.backward()

        # Update the weights of the neural network
        self.optimizer_.step()

        # Return the loss
        return loss.item()

    def _evaluate_batch(self, batch: list[torch.Tensor]) -> torch.Tensor:

        # Set the type of the batch
        history, future = batch

        # Forecast the data
        forecast = self.neural_network_(history).reshape(future.shape)

        # Compute the difference with the given data
        if self.error_metric == "mean-squared-error":
            return torch.mean(
                (forecast - future) ** 2, dim=tuple(range(1, forecast.ndim))
            )
        if self.error_metric == "mean-absolute-error":
            return torch.mean(
                torch.abs(forecast - future), dim=tuple(range(1, forecast.ndim))
            )

        # Raise an error if invalid metric is given
        raise ValueError(
            f"Unknown error_metric '{self.error_metric}'. Valid options are ['mean-squared-error', 'mean-absolute-error']"
        )

    def _evaluate(self, data_loader: torch.utils.data.DataLoader) -> np.array:
        decision_scores = super()._evaluate(data_loader)
        return np.concatenate(
            ([decision_scores[0] for _ in range(self.forecast_length)], decision_scores)
        )
