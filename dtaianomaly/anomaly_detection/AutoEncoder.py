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
    ReconstructionDataset,
)


class AutoEncoder(BaseNeuralDetector):
    """
    Use an AutoEncoder to detect anomalies :cite:`sakurada2014anomaly`.

    An auto encoder is a neural network that consists of two parts: an encoder
    and a decoder. The encoder maps the input features to a lower dimensional
    space, the latent space, while the decoder reconstructs the latent embedding
    back into the original feature space. Samples that are common during the
    training phase (i.e., normal behavior) are more easily reconstructed compared
    to rare observations (i.e., anomalies). Thus, anomalies are detected by
    reconstructing the time series data and measuring the deviation of the
    reconstruction from the original data.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    error_metric: {'mean-absolute-error', 'mean-squared-error'}, default='mean-absolute-error'
        Used metric for computing the distance between the actual given data and the
        reconstructed data.
    TODO ...

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    optimizer_: torch.optim.Optimizer
    neural_network_: torch.nn.Module

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import AutoEncoder
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> auto_encoder = AutoEncoder(10, seed=0).fit(x)
    >>> auto_encoder.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.59210092, 0.56707534, 0.56629006, ..., 0.58380051, 0.5808109 , 0.54450774]...)
    """

    error_metric: Literal["mean-absolute-error", "mean-squared-error"]
    encoder_dimensions: list[int]
    latent_space_dimension: int
    decoder_dimensions: list[int]
    dropout_rate: float
    activation_function: _ACTIVATION_FUNCTION_TYPE
    batch_normalization: bool

    def __init__(
        self,
        window_size: str | int,
        error_metric: Literal[
            "mean-absolute-erro", "mean-squared-error"
        ] = "mean-absolute-error",
        encoder_dimensions: list[int] = (64,),
        latent_space_dimension: int = 32,
        decoder_dimensions: list[int] = (64,),
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
        if error_metric not in ["mean-squared-error", "mean-absolute-error"]:
            raise ValueError(
                f"Unknown error_metric '{error_metric}'. Valid options are ['mean-squared-error', 'mean-absolute-error']"
            )

        if not utils.is_valid_list(encoder_dimensions, int):
            raise TypeError("`encoder_dimensions` should be a list of integer")
        if any(map(lambda x: x <= 0, encoder_dimensions)):
            raise ValueError(
                "All values in `encoder_dimensions` should be strictly positive"
            )

        if not isinstance(latent_space_dimension, int) or isinstance(
            latent_space_dimension, bool
        ):
            raise TypeError("`latent_space_dimension` should be an integer")
        if latent_space_dimension <= 0:
            raise ValueError("`latent_space_dimension` should strictly positive")

        if not utils.is_valid_list(decoder_dimensions, int):
            raise TypeError("`decoder_dimensions` should be a list of integer")
        if any(map(lambda x: x <= 0, decoder_dimensions)):
            raise ValueError(
                "All values in `decoder_dimensions` should be strictly positive"
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
        self.encoder_dimensions = encoder_dimensions
        self.latent_space_dimension = latent_space_dimension
        self.decoder_dimensions = decoder_dimensions
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function
        self.batch_normalization = batch_normalization

    def _build_dataset(self, X: np.ndarray) -> torch.utils.data.Dataset:
        return ReconstructionDataset(
            X=X,
            window_size=self.window_size_,
            stride=self.stride,
            standard_scaling=self.standard_scaling,
            device=self.device,
        )

    def _build_architecture(self, n_attributes: int) -> torch.nn.Module:
        return _AutoEncoderArchitecture(
            encoder=self._build_encoder(n_attributes * self.window_size_),
            decoder=self._build_decoder(n_attributes * self.window_size_),
        )

    def _train_batch(self, batch: list[torch.Tensor]) -> float:

        # Set the type of the batch
        data = batch[0].to(self.device).float()

        # Initialize the gradients to zero
        self.optimizer_.zero_grad()

        # Feed the data to the neural network
        reconstructed = self.neural_network_(data)

        # Compute the loss
        loss = self.loss_function(reconstructed, data)

        # Compute the gradients of the loss
        loss.backward()

        # Update the weights of the neural network
        self.optimizer_.step()

        # Return the loss
        return loss.item()

    def _evaluate_batch(self, batch: list[torch.Tensor]) -> torch.Tensor:

        # Set the type of the batch
        data = batch[0].to(self.device).float()

        # Reconstruct the batch
        reconstructed = self.neural_network_(data)

        # Compute the difference with the given data
        if self.error_metric == "mean-squared-error":
            return torch.mean((reconstructed - data) ** 2, dim=1)
        if self.error_metric == "mean-absolute-error":
            return torch.mean(torch.abs(reconstructed - data), dim=1)

        # Raise an error if invalid metric is given
        raise ValueError(
            f"Unknown error_metric '{self.error_metric}'. Valid options are ['mean-squared-error', 'mean-absolute-error']"
        )

    def _build_encoder(self, input_size: int) -> torch.nn.Module:

        # Initialize layer inputs and outputs
        inputs = [input_size, *self.encoder_dimensions]
        outputs = [*self.encoder_dimensions, self.latent_space_dimension]

        # Initialize the encoder
        encoder = torch.nn.Sequential()

        # Add all the layers
        for i in range(len(inputs)):

            # Add the linear layer
            encoder.add_module(f"linear-{i}", torch.nn.Linear(inputs[i], outputs[i]))

            # Add batch normalization
            if self.batch_normalization and i > 0:
                encoder.add_module(f"batch-norm-{i}", torch.nn.BatchNorm1d(outputs[i]))

            # Add the activation function
            encoder.add_module(
                f"activation-{i}",
                self._build_activation_function(self.activation_function),
            )

            # Add the dropout layer
            if self.dropout_rate > 0:
                encoder.add_module(f"dropout-{i}", torch.nn.Dropout(self.dropout_rate))

        # Return the encoder
        return encoder

    def _build_decoder(self, input_size: int) -> torch.nn.Module:

        # Initialize layer inputs and outputs
        inputs = [self.latent_space_dimension, *self.decoder_dimensions]
        outputs = [*self.decoder_dimensions, input_size]

        # Initialize the decoder
        decoder = torch.nn.Sequential()

        # Add all the layers
        for i in range(len(inputs)):

            # Add the linear layer
            decoder.add_module(f"linear-{i}", torch.nn.Linear(inputs[i], outputs[i]))

            # Add batch normalization
            if self.batch_normalization and i < len(inputs) - 1:
                decoder.add_module(f"batch-norm-{i}", torch.nn.BatchNorm1d(outputs[i]))

            # Add the activation function
            decoder.add_module(
                f"activation-{i}",
                self._build_activation_function(self.activation_function),
            )

            # Add the dropout layer
            if self.dropout_rate > 0 and i < len(inputs) - 1:
                decoder.add_module(f"dropout-{i}", torch.nn.Dropout(self.dropout_rate))

        # Return the decoder
        return decoder


class _AutoEncoderArchitecture(torch.nn.Module):

    encoder: torch.nn.Module
    decoder: torch.nn.Module

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))
