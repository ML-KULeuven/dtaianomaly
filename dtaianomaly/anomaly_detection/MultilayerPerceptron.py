from typing import Literal

import numpy as np
import torch

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.BaseNeuralDetector import (
    _ACTIVATION_FUNCTION_TYPE,
    _COMPILE_MODE_TYPE,
    _OPTIMIZER_TYPE,
    BaseNeuralDetector,
    ForecastDataset,
)


class MultilayerPerceptron(BaseNeuralDetector):
    """
    TODO ...

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import MultilayerPerceptron
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> mlp = MultilayerPerceptron(10, seed=0).fit(x)
    >>> mlp.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([1.8944391 , 1.8944391 , 1.83804671, ..., 0.59621549, 0.54421651,
           0.05852008]...)
    """

    error_metric: Literal["mean-absolute-error", "mean-squared-error"]
    forecast_length: int
    hidden_layers: list[int]
    dropout_rate: float
    activation_function: _ACTIVATION_FUNCTION_TYPE
    batch_normalization: bool

    def __init__(
        self,
        window_size: str | int,
        error_metric: Literal[
            "mean-absolute-error", "mean-squared-error"
        ] = "mean-absolute-error",
        forecast_length: int = 1,
        hidden_layers: list[int] = (64, 32),
        dropout_rate: float = 0.2,
        activation_function: _ACTIVATION_FUNCTION_TYPE = "relu",
        batch_normalization: bool = True,
        stride: int = 1,
        standard_scaling: bool = True,
        batch_size: int = 32,
        data_loader_kwargs: dict[str, any] = None,
        optimizer: _OPTIMIZER_TYPE = "adam",
        learning_rate: float = 1e-3,
        optimizer_kwargs: dict[str, any] = None,
        compile_model: bool = False,
        compile_mode: _COMPILE_MODE_TYPE = "default",
        n_epochs: int = 10,
        loss_function: torch.nn.Module = torch.nn.MSELoss(),
        device: str = "cpu",
        seed: int = None,
    ):
        super().__init__(
            supervision=Supervision.SEMI_SUPERVISED,
            window_size=window_size,
            stride=stride,
            standard_scaling=standard_scaling,
            batch_size=batch_size,
            data_loader_kwargs=data_loader_kwargs,
            optimizer=optimizer,
            learning_rate=learning_rate,
            optimizer_kwargs=optimizer_kwargs,
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

        if not utils.is_valid_list(hidden_layers, int):
            raise TypeError("`hidden_layers` should be a list of integer")
        if any(map(lambda x: x <= 0, hidden_layers)):
            raise ValueError(
                "All values in `hidden_layers` should be strictly positive"
            )

        if not isinstance(activation_function, str):
            raise TypeError("`activation_function` should be a string")
        if activation_function not in self._ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unknown `activation_function` '{error_metric}'. Valid options are {list(self._ACTIVATION_FUNCTIONS.keys())}"
            )

        if not isinstance(batch_normalization, bool):
            raise TypeError("`batch_normalization` should be a list of bools or a bool")

        if not isinstance(dropout_rate, (float, int)) or isinstance(dropout_rate, bool):
            raise TypeError("`dropout_rate` should be a list of floats or a float")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"`dropout_rate` should be in interval [0, 1[.")

        self.error_metric = error_metric
        self.forecast_length = forecast_length
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function
        self.batch_normalization = batch_normalization

    def _build_dataset(self, X: np.ndarray) -> torch.utils.data.Dataset:
        return ForecastDataset(
            X=X,
            window_size=self.window_size_,
            stride=self.stride,
            standard_scaling=self.standard_scaling,
            device=self.device,
            forecast_length=self.forecast_length,
        )

    def _build_architecture(self, n_attributes: int) -> torch.nn.Module:
        # Initialize layer inputs and outputs
        inputs = [n_attributes * self.window_size_, *self.hidden_layers]
        outputs = [*self.hidden_layers, n_attributes * self.forecast_length]

        # Initialize the encoder
        mlp = torch.nn.Sequential()

        # Add all the layers
        for i in range(len(inputs)):

            # Add the linear layer
            mlp.add_module(f"linear-{i}", torch.nn.Linear(inputs[i], outputs[i]))

            # Add batch normalization
            if self.batch_normalization and i > 0 and i < len(inputs) - 1:
                mlp.add_module(f"batch-norm-{i}", torch.nn.BatchNorm1d(outputs[i]))

            # Add the activation function
            mlp.add_module(
                f"activation-{i}",
                self._build_activation_function(self.activation_function),
            )

            # Add the dropout layer
            if self.dropout_rate > 0 and i < len(inputs) - 1:
                mlp.add_module(f"dropout-{i}", torch.nn.Dropout(self.dropout_rate))

        # Return the encoder
        return mlp

    def _train_batch(self, batch: list[torch.Tensor]) -> float:

        # Set the type of the batch
        history, future = batch

        # Initialize the gradients to zero
        self.optimizer_.zero_grad()

        # Feed the data to the neural network
        forecast = self.neural_network_(history)

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
        forecast = self.neural_network_(history)

        # Compute the difference with the given data
        if self.error_metric == "mean-squared-error":
            return torch.mean((forecast - future) ** 2, dim=1)
        if self.error_metric == "mean-absolute-error":
            return torch.mean(torch.abs(forecast - future), dim=1)

        # Raise an error if invalid metric is given
        raise ValueError(
            f"Unknown error_metric '{self.error_metric}'. Valid options are ['mean-squared-error', 'mean-absolute-error']"
        )

    def _evaluate(self, data_loader: torch.utils.data.DataLoader) -> np.array:
        decision_scores = super()._evaluate(data_loader)
        return np.concatenate(
            ([decision_scores[0] for _ in range(self.forecast_length)], decision_scores)
        )
