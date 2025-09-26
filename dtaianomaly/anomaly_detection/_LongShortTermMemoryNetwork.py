import torch

from dtaianomaly.anomaly_detection._BaseDetector import Supervision
from dtaianomaly.anomaly_detection._BaseNeuralDetector import (
    COMPILE_MODE_TYPE,
    LOSS_TYPE,
    OPTIMIZER_TYPE,
)
from dtaianomaly.anomaly_detection._BaseNeuralForecastingDetector import (
    ERROR_METRIC_TYPE,
    BaseNeuralForecastingDetector,
)
from dtaianomaly.type_validation import BoolAttribute, FloatAttribute, IntegerAttribute
from dtaianomaly.windowing import WINDOW_SIZE_TYPE

__all__ = ["LongShortTermMemoryNetwork"]


class LongShortTermMemoryNetwork(BaseNeuralForecastingDetector):
    """
    Use an LSTM to detect anomalies :cite:`malhotra2015long`.

    The Long-Short Term Memory (LSTM) anomaly detector combines a decoder
    with LSTM layers with a liner layer to forecast the time series, given
    a subsequence. The anomalies are then detected by measuring the deviation
    between the forecasted values and the actual observations. The LSTM-
    decoder reads each subsequence sequentially, and constructs a hidden
    representation at each time point. The hidden representation at each
    time step is based on the observations at that time step, but also
    on the hidden state at the previous step. The LSTM units include
    learnable gates, which guide the information flow to avoid issues
    with gradients as faced in RNN networks. Once the complete sequence
    is processed, the output is fed to a linear layer, which will forecast
    the data.

    The architecture of the LSTM consists of 2 blocks: (1) an LSTM-decoder
    consisting of one or more LST-layers with multiple LSTM-units, and (2)
    a single linear layer.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    error_metric: {"mean-absolute-error", "mean-squared-error"}, default="mean-absolute-error"
        The error measure between the reconstructed window and the original window.
    hidden_units: int, default=8
        The number of hidden unit in each LSTM layer.
    num_lstm_layers: int, default=1
        The number of LSTM layers in the LSTM-block.
    dropout_rate: float in interval [0, 1[, default=0.0
        The dropout rate to put on each layer in the LSTM block.
    bias: bool, default=True
        Whether to use bias weights in each layer of the LSTM block.
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
    loss_function: {"mse", "l1", "huber} or torch.nn.Module, default="mse"
        The loss function to use for updating the weights. Valid options are:

        - ``'mse'``: Use the Mean Squared Error loss.
        - ``'l1'``: Use the L1-loss or the mean absolute error.
        - ``'huber'``: Use the huber loss, which smoothly combines the MSE-loss with the L1-loss.
        - ``torch.nn.Module``: a custom torch module to use for the loss function.

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
    >>> from dtaianomaly.anomaly_detection import LongShortTermMemoryNetwork
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> lstm = LongShortTermMemoryNetwork(10, seed=0).fit(x)
    >>> lstm.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE, +SKIP
    array([0.354334  , 0.354334  , 0.28025536, ..., 0.61675562, 0.90525854,
           0.39284754]...)

    See also
    --------
    BaseNeuralForecastingDetector: Use a neural network to forecast the time
        series, and detect anomalies by measuring the difference with the
        actual observations.
    """

    hidden_units: int
    num_lstm_layers: int
    bias: bool
    dropout_rate: float

    attribute_validation = {
        "hidden_units": IntegerAttribute(minimum=1),
        "num_lstm_layers": IntegerAttribute(minimum=1),
        "bias": BoolAttribute(),
        "dropout_rate": FloatAttribute(0.0, 1.0, inclusive_maximum=False),
    }

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        error_metric: ERROR_METRIC_TYPE = "mean-absolute-error",
        forecast_length: int = 1,
        hidden_units: int = 8,
        num_lstm_layers: int = 1,
        bias: bool = True,
        dropout_rate: float = 0.0,
        stride: int = 1,
        standard_scaling: bool = True,
        batch_size: int = 32,
        data_loader_kwargs: dict[str, any] = None,
        optimizer: OPTIMIZER_TYPE = "adam",
        learning_rate: float = 1e-3,
        compile_model: bool = False,
        compile_mode: COMPILE_MODE_TYPE = "default",
        n_epochs: int = 10,
        loss_function: LOSS_TYPE = "mse",
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
        self.hidden_units = hidden_units
        self.num_lstm_layers = num_lstm_layers
        self.bias = bias
        self.dropout_rate = dropout_rate

    def _build_architecture(self, n_attributes: int) -> torch.nn.Module:
        return _LSTM(
            n_attributes=n_attributes,
            lstm=torch.nn.LSTM(
                input_size=n_attributes,
                hidden_size=self.hidden_units * self.forecast_length,
                num_layers=self.num_lstm_layers,
                bias=self.bias,
                batch_first=True,
                dropout=self.dropout_rate,
            ),
            linear=torch.nn.Linear(
                in_features=self.window_size_
                * self.hidden_units
                * self.forecast_length,
                out_features=n_attributes * self.forecast_length,
            ),
        )


class _LSTM(torch.nn.Module):

    n_attributes: int
    lstm: torch.nn.LSTM
    flatten: torch.nn.Flatten
    linear: torch.nn.Module

    def __init__(self, n_attributes: int, lstm: torch.nn.LSTM, linear: torch.nn.Linear):
        super().__init__()
        self.n_attributes = n_attributes
        self.lstm = lstm
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1, self.n_attributes)
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
