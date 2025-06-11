from typing import Literal

import numpy as np
import torch

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.BaseNeuralDetector import BaseNeuralDetector


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
    error_metric: {'mae', 'mse'}, default='mae'
        Used metric for computing the distance between the actual given data and the
        reconstructed data. Valid options are:

        - ``'mae'``: Compute the mean absolute error
        - ``'mse'``: Compute the mean squared error
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
    >>> auto_encoder = AutoEncoder(10, random_state=0).fit(x)
    >>> auto_encoder.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.21060435, 0.21896446, 0.21212289, ..., 0.98364538, 0.97652012,
           0.97193861]...)
    """

    error_metric: Literal["mae", "mse"]
    hidden_layer_dimension: list[int]
    dropout_rate: float
    activation_function: Literal  # TODO type
    batch_normalization: bool

    def __init__(
        self,
        window_size: str | int,
        error_metric: Literal["mae", "mse"] = "mae",
        hidden_layer_dimension: list[int] = (64, 32),
        dropout_rate: float = 0.2,
        activation_function: Literal = "relu",  # TODO type
        batch_normalization: bool = True,
        stride: int = 1,
        standard_scaling: bool = True,
        batch_size: int = 32,
        data_loader_kwargs: dict[str, any] = None,
        optimizer: Literal["adam", "sgd", "adagrad", "rmsprop"] = "adam",
        learning_rate: float = 1e-3,
        optimizer_kwargs: dict[str, any] = None,
        compile_model: bool = False,
        compile_mode: Literal[
            "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        ] = "default",
        n_epochs: int = 10,
        loss_function: torch.nn.Module = torch.nn.MSELoss(),
        device: str = "cpu",
        random_state: int = None,
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
            random_state=random_state,
        )

        if not isinstance(error_metric, str):
            raise TypeError("`error_metric` should be a string")
        if error_metric not in ["mse", "mae"]:
            raise ValueError(
                f"Unknown error_metric '{error_metric}'. Valid options are ['mse', 'mae']"
            )

        if not utils.is_valid_list(hidden_layer_dimension, int):
            raise TypeError("`hidden_layer_dimension` should be a list of integer")
        if any(map(lambda x: x <= 0, hidden_layer_dimension)):
            raise ValueError(
                "All values in `hidden_layer_dimension` should be strictly positive"
            )
        if len(hidden_layer_dimension) < 1:
            raise ValueError(
                "`hidden_layer_dimension` should have at least one element"
            )

        if not isinstance(activation_function, str):
            raise TypeError("`activation_function` should be a string")
        if activation_function not in []:
            raise ValueError(
                f"Unknown error_metric '{error_metric}'. Valid options are []"
            )

        if not isinstance(batch_normalization, bool):
            raise TypeError("`batch_normalization` should be a list of bools or a bool")

        if not isinstance(dropout_rate, float):  # TODO check that 0 (the integer) is ok
            raise TypeError("`dropout_rate` should be a list of floats or a float")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"`dropout_rate` should be in interval [0, 1[.")

        self.error_metric = error_metric
        self.hidden_layer_dimension = hidden_layer_dimension
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function
        self.batch_normalization = batch_normalization

    def _build_dataset(self, X: np.ndarray) -> torch.utils.data.Dataset:
        return torch.utils.data.TensorDataset(torch.from_numpy(X))

    def _build_architecture(self, input_size: int) -> torch.nn.Module:
        raise NotImplementedError  # TODO

    def _train_batch(self, batch: torch.Tensor) -> float:
        # Initialize the gradients to zero
        self.optimizer_.zero_grad()

        # Feed the data to the neural network
        reconstructed = self.neural_network_(batch)

        # Compute the loss
        loss = self.loss_function(reconstructed, batch)

        # Compute the gradients of the loss
        loss.backward()

        # Update the weights of the neural network
        self.optimizer_.step()

        # Return the loss
        return loss.item()

    def _evaluate_batch(self, batch: torch.Tensor) -> torch.Tensor:
        # Reconstruct the batch
        reconstructed = self.neural_network_(batch)

        # Compute the difference with the given data
        if self.error_metric == "mse":
            return torch.mean((reconstructed - batch) ** 2, dim=1)
        elif self.error_metric == "mae":
            return torch.mean(torch.abs(reconstructed - batch), dim=1)

        # Raise an error if invalid metric is given
        raise ValueError(
            f"Unknown error_metric '{self.error_metric}'. Valid options are ['mse', 'mae'"
        )


# class _AutoEncoderArchitecture(torch.nn.Module):
#
#     def __init__(
#         self,
#         input_layer_dimension: int,
#         latent_space_dimension: int,
#         encoder_hidden_layer_dimension: List[int],
#         decoder_hidden_layer_dimension: List[int],
#         encoder_activation_functions: Union[str, List[str]],
#         decoder_activation_functions: Union[str, List[str]],
#         encoder_batch_normalization: Union[bool, List[bool]],
#         decoder_batch_normalization: Union[bool, List[bool]],
#         encoder_dropout_rate: Union[float, List[float]],
#         decoder_dropout_rate: Union[float, List[float]],
#     ):
#         super().__init__()
#
#         # Inputs do not need to be checked here, as they are already checked in the
#         # constructor of the auto encoder
#
#         # Initialize the encoder
#         self.encoder = _initialize_mlp(
#             # initialize encoder and decoder as a sequential
#             input_dimension=input_layer_dimension,
#             output_dimension=latent_space_dimension,
#             hidden_layer_dimension=encoder_hidden_layer_dimension,
#             activation_function=encoder_activation_functions,
#             batch_normalization=encoder_batch_normalization,
#             batch_normalization_first_layer=False,
#             batch_normalization_last_layer=True,
#             dropout_rate=encoder_dropout_rate,
#             dropout_first_layer=True,
#             dropout_last_layer=True,
#         )
#
#         # Initialize the decoder
#         self.decoder = _initialize_mlp(
#             input_dimension=latent_space_dimension,
#             output_dimension=input_layer_dimension,
#             hidden_layer_dimension=decoder_hidden_layer_dimension,
#             activation_function=decoder_activation_functions,
#             batch_normalization=decoder_batch_normalization,
#             batch_normalization_first_layer=True,
#             batch_normalization_last_layer=False,
#             dropout_rate=decoder_dropout_rate,
#             dropout_first_layer=True,
#             dropout_last_layer=False,
#         )
#
#     def forward(self, x):
#         return self.decoder(self.encoder(x))
