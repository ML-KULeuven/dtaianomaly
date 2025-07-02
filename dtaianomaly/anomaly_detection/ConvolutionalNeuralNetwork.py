from typing import Callable, Literal

import numpy as np
import torch

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.BaseNeuralDetector import (
    _ACTIVATION_FUNCTION_TYPE,
    _COMPILE_MODE_TYPE,
    _MODEL_PARAMETERS_TYPE,
    _OPTIMIZER_TYPE,
)
from dtaianomaly.anomaly_detection.BaseNeuralDetector_utils import (
    BaseNeuralForecastingDetector,
)


class ConvolutionalNeuralNetwork(BaseNeuralForecastingDetector):
    """
    Use a convolutional neural network to detect anomalies.

    The Convolutional Neural Network (CNN) is a  neural network consisting
    of convolutional layers, each consisting of multiple kernels. Given some
    input, the convolutional layer computes the convolution of the input with
    each kernel to create an output. The input sequences are fed through
    multiple such convolutional layers, and the task is to forecast the
    time series. Hence, by computing the difference between the forecasted
    value and the actually observed values, the neural network can detect
    anomalies.

    The architecture of the CNN consists of blocks, in which each block applies
    the following operations: convolutional layer :math:`\\rightarrow` batch
    normalization :math:`\\rightarrow` activation function :math:`\\rightarrow`
    average pooling. The first and final layers of the network has no batch
    normalization.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    error_metric: {"mean-absolute-error", "mean-squared-error"}, default="mean-absolute-error"
        The error measure between the reconstructed window and the original window.
    kernel_size: int, default=3
        The size of the kernels in the convolutional layers.
    hidden_layers: list of ints, default=[64, 32]
        The number of kernels in each hidden layer. Must contain at least 1 value.
    activation_function: {"linear", "relu", "sigmoid", "tanh"} default="relu"
        The activation function to use at the end of each convolutional layer.
    batch_normalization: bool = True,
        Whether to add batch normalization after each convolutional layer or not.
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

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import ConvolutionalNeuralNetwork
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> cnn = ConvolutionalNeuralNetwork(10, seed=0).fit(x)
    >>> cnn.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE, +SKIP
    array([0.07708263, 0.07708263, 0.06242053, ..., 0.1827196 , 0.2396274 ,
           0.06390759]...)

    See also
    --------
    BaseNeuralForecastingDetector: Use a neural network to forecast the time
        series, and detect anomalies by measuring the difference with the
        actual observations.
    """

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
        kernel_size: int = 3,
        hidden_layers: list[int] = (64, 32),
        activation_function: _ACTIVATION_FUNCTION_TYPE = "relu",
        batch_normalization: bool = True,
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
            window_size=window_size,
            supervision=Supervision.SEMI_SUPERVISED,
            error_metric=error_metric,
            forecast_length=forecast_length,
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

        if not isinstance(kernel_size, int) or isinstance(kernel_size, bool):
            raise TypeError("`kernel_size` should be integer")
        if kernel_size < 1:
            raise ValueError("`kernel_size` should be strictly positive")

        if not utils.is_valid_list(hidden_layers, int):
            raise TypeError("`hidden_layers` should be a list of integer")
        if any(map(lambda x: x <= 0, hidden_layers)):
            raise ValueError(
                "All values in `hidden_layers` should be strictly positive"
            )
        if len(hidden_layers) == 0:
            raise ValueError("There should at least be one hidden layer!")

        if not isinstance(activation_function, str):
            raise TypeError("`activation_function` should be a string")
        if activation_function not in self._ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unknown `activation_function` '{activation_function}'. Valid options are {list(self._ACTIVATION_FUNCTIONS.keys())}"
            )

        if not isinstance(batch_normalization, bool):
            raise TypeError("`batch_normalization` should be a bool")

        self.kernel_size = kernel_size
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.batch_normalization = batch_normalization

    def _build_architecture(self, n_attributes: int) -> torch.nn.Module:
        # Initialize the CNN
        cnn = torch.nn.Sequential()
        padding = int((self.kernel_size - 1) / 2)

        # Initialize layer inputs and outputs
        inputs = [n_attributes, *self.hidden_layers]
        outputs = [*self.hidden_layers, n_attributes * self.forecast_length]

        for i in range(len(inputs) - 1):

            # Add the convolutional layer
            cnn.add_module(
                f"conv-{i}",
                torch.nn.Conv1d(
                    in_channels=inputs[i],
                    out_channels=outputs[i],
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=padding,
                ),
            )

            # Add batch normalization
            if self.batch_normalization and 0 < i < len(inputs) - 1:
                cnn.add_module(f"batch-norm-{i}", torch.nn.BatchNorm1d(outputs[i]))

            # Add the activation function
            cnn.add_module(
                f"activation-{i}",
                self._build_activation_function(self.activation_function),
            )

            # Add a pooling layer
            cnn.add_module(f"pool-{i}", torch.nn.AvgPool1d(kernel_size=2))

        # Add a linear layer
        cnn.add_module("flatten", torch.nn.Flatten())
        channel_correction_term = int(
            np.floor(self.window_size_ / 2 ** len(self.hidden_layers))
        )
        cnn.add_module(
            "linear",
            torch.nn.Linear(
                in_features=inputs[-1] * channel_correction_term,
                out_features=outputs[-1],
            ),
        )

        # Return the CNN
        return _CNN(n_attributes, cnn)


class _CNN(torch.nn.Module):

    n_attributes: int
    cnn: torch.nn.Sequential

    def __init__(self, n_attributes: int, cnn: torch.nn.Sequential):
        super().__init__()
        self.n_attributes = n_attributes
        self.cnn = cnn

    def forward(self, x):
        x = x.view(x.shape[0], self.n_attributes, -1)
        x = self.cnn(x)
        return x
