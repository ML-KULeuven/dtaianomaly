import torch

from dtaianomaly.anomaly_detection._BaseDetector import Supervision
from dtaianomaly.anomaly_detection._BaseNeuralDetector import (
    ACTIVATION_FUNCTION_TYPE,
    ACTIVATION_FUNCTIONS,
    COMPILE_MODE_TYPE,
    LOSS_TYPE,
    OPTIMIZER_TYPE,
)
from dtaianomaly.anomaly_detection._BaseNeuralForecastingDetector import (
    ERROR_METRIC_TYPE,
    BaseNeuralForecastingDetector,
)
from dtaianomaly.type_validation import (
    BoolAttribute,
    FloatAttribute,
    IntegerAttribute,
    LiteralAttribute,
)
from dtaianomaly.windowing import WINDOW_SIZE_TYPE

__all__ = ["Transformer"]


class Transformer(BaseNeuralForecastingDetector):
    """
    Use a transformer to detect anomalies :cite:`vaswani2017attention`.

    A transformer anomaly detector first forecasts the time series, and
    then detects anomalies by measuring the deviation from the forecasted
    values to the actual observations. A transformer is a neural network
    consisting of only attention-layers: all you need is attention. The
    forecasting network therefore consists first of a transformer-encoder,
    which is connected to a linear layer to forecast the time series.

    The architecture of the transformer consists of 2 blocks: (1) a transformer
    -decoder consisting of one or more attention-layers, and (2) a single linear
    layer.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    error_metric: {"mean-absolute-error", "mean-squared-error"}, default="mean-absolute-error"
        The error measure between the reconstructed window and the original window.
    num_heads: int, default=12
        The number of heads in each attention layer.
    num_transformer_layers: int, default=1
        The number of attention layers.
    dimension_feedforward: int, default=32,
        The dimension of the linear layer at the end of each attention layer.
    bias: bool, default=True
        Whether to use bias weights in each layer of the LSTM block.
    dropout_rate: float in interval [0, 1[, default=0.0
        The dropout rate to put on each attention layer.
    activation_function: {"linear", "relu", "sigmoid", "tanh"} default="relu"
        The activation function to use at the end of each attention layer.
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
    >>> from dtaianomaly.anomaly_detection import Transformer
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> transformer = Transformer(10, seed=0).fit(x)
    >>> transformer.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE, +SKIP
    array([0.41845179, 0.41845179, 0.3603762 , ..., 0.46213843, 0.63743933,
           0.08675425]...)

    See also
    --------
    BaseNeuralForecastingDetector: Use a neural network to forecast the time
        series, and detect anomalies by measuring the difference with the
        actual observations.
    """

    num_heads: int
    num_transformer_layers: int
    dimension_feedforward: int
    bias: bool
    dropout_rate: float
    activation_function: ACTIVATION_FUNCTION_TYPE

    attribute_validation = {
        "num_heads": IntegerAttribute(minimum=1),
        "num_transformer_layers": IntegerAttribute(minimum=1),
        "dimension_feedforward": IntegerAttribute(minimum=1),
        "dropout_rate": FloatAttribute(0.0, 1.0, inclusive_maximum=False),
        "activation_function": LiteralAttribute(ACTIVATION_FUNCTIONS),
        "bias": BoolAttribute(),
    }

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        error_metric: ERROR_METRIC_TYPE = "mean-absolute-error",
        forecast_length: int = 1,
        num_heads: int = 12,
        num_transformer_layers: int = 1,
        dimension_feedforward: int = 32,
        bias: bool = True,
        dropout_rate: float = 0.0,
        activation_function: ACTIVATION_FUNCTION_TYPE = "relu",
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
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.dimension_feedforward = dimension_feedforward
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function

    def _build_architecture(self, n_attributes: int) -> torch.nn.Module:

        transformer = torch.nn.Sequential()
        transformer.add_module("flatten", torch.nn.Flatten())

        d_model = n_attributes * self.window_size_
        nhead = _adjust_nhead(d_model, self.num_heads)

        transformer.add_module(
            "transformer",
            torch.nn.TransformerEncoder(
                encoder_layer=torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=self.dimension_feedforward,
                    dropout=self.dropout_rate,
                    activation=self._build_activation_function(
                        self.activation_function
                    ),
                    batch_first=True,
                    bias=self.bias,
                ),
                num_layers=self.num_transformer_layers,
                enable_nested_tensor=(nhead % 2) == 0,
            ),
        )

        transformer.add_module(
            "linear",
            torch.nn.Linear(
                in_features=n_attributes * self.window_size_,
                out_features=n_attributes * self.forecast_length,
            ),
        )

        return transformer


def _adjust_nhead(d_model, nhead) -> int:
    """
    Computes a valid nhead for the given parameters, such that the constraint

        (d_model // nhead) * nhead == d_model

    is satisfied. This is done by finding the value closest to nhead which
    satisfies the constraint. This value can thus be larger or smaller. If
    the constraint is not satisfied by the given values, and there are
    two values equally far from nhead that satisfy the constraint, then the
    smaller value is returned (e.g., the constraint is not satisfied if
    d_model=100 and nhead=3, but it is satisfied for both 2 and 4, but 2
    will be returned since it is lower).
    """
    if d_model % nhead == 0:
        return nhead  # Already valid

    # Search for closest valid nhead
    lower = nhead - 1
    upper = nhead + 1

    while lower > 1 and upper <= d_model:
        if d_model % lower == 0:
            return lower
        if d_model % upper == 0:
            return upper
        lower -= 1
        upper += 1

    return 1
