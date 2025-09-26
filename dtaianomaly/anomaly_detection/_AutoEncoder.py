import torch

from dtaianomaly.anomaly_detection._BaseDetector import Supervision
from dtaianomaly.anomaly_detection._BaseNeuralDetector import (
    ACTIVATION_FUNCTION_TYPE,
    ACTIVATION_FUNCTIONS,
    COMPILE_MODE_TYPE,
    LOSS_TYPE,
    OPTIMIZER_TYPE,
)
from dtaianomaly.anomaly_detection._BaseNeuralReconstructionDetector import (
    ERROR_METRIC_TYPE,
    BaseNeuralReconstructionDetector,
)
from dtaianomaly.type_validation import (
    BoolAttribute,
    FloatAttribute,
    IntegerAttribute,
    ListAttribute,
    LiteralAttribute,
)
from dtaianomaly.windowing import WINDOW_SIZE_TYPE

__all__ = ["AutoEncoder"]


class AutoEncoder(BaseNeuralReconstructionDetector):
    """
    Use an auto encoder to detect anomalies :cite:`sakurada2014anomaly`.

    An auto encoder is a neural network that consists of two parts: an encoder
    and a decoder. The encoder maps the input features to a lower dimensional
    space, the latent space, while the decoder reconstructs the latent embedding
    back into the original feature space. Samples that are common during the
    training phase (i.e., normal behavior) are more easily reconstructed compared
    to rare observations (i.e., anomalies). Thus, anomalies are detected by
    reconstructing the time series data and measuring the deviation of the
    reconstruction from the original data.

    The architecture of the autoencoder consists of blocks, in which each block
    applies the following operations: fully-connected layer :math:`\\rightarrow`
    batch normalization :math:`\\rightarrow` activation function :math:`\\rightarrow`
    dropout layer. The first layer of the encoder has no batch normalization,
    and the final layer of the decoder has no batch normalization nor dropout.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    error_metric: {"mean-absolute-error", "mean-squared-error"}, default="mean-absolute-error"
        The error measure between the reconstructed window and the original window.
    encoder_dimensions: list of ints, default=[64]
        The number of neurons in each layer of the encoder. If an empty list is given, then
        the input of the encoder is directly connected to the latent space in a fully-connected
        manner.
    latent_space_dimension: int default=32
        The dimension of the latent space.
    decoder_dimensions: list of ints, default=[64]
        The number of neurons in each layer of the decoder. If an empty list is given, then
        the latent space is directly connected to the output in a fully-connected manner.
    dropout_rate: float in interval [0, 1[, default=0.2
        The dropout rate for the dropout layers. If the dropout rate is 0, no dropout layers
        will be added to the auto encoder.
    activation_function: {"linear", "relu", "sigmoid", "tanh"} default="relu"
        The activation function to use at the end of each layer.
    batch_normalization: bool = True,
        Whether to add batch normalization after each layer or not.
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
    >>> from dtaianomaly.anomaly_detection import AutoEncoder
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> auto_encoder = AutoEncoder(10, seed=0).fit(x)
    >>> auto_encoder.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE, +SKIP
    array([0.59210092, 0.56707534, 0.56629006, ..., 0.58380051, 0.5808109 , 0.54450774]...)

    See also
    --------
    BaseNeuralReconstructionDetector: Use a neural network to reconstruct
        windows in the time series, and detect anomalies as windows that
        are incorrectly reconstructed.
    """

    encoder_dimensions: list[int]
    latent_space_dimension: int
    decoder_dimensions: list[int]
    dropout_rate: float
    activation_function: ACTIVATION_FUNCTION_TYPE
    batch_normalization: bool

    attribute_validation = {
        "encoder_dimensions": ListAttribute(IntegerAttribute(minimum=1)),
        "latent_space_dimension": IntegerAttribute(minimum=1),
        "decoder_dimensions": ListAttribute(IntegerAttribute(minimum=1)),
        "dropout_rate": FloatAttribute(0.0, 1.0, inclusive_maximum=False),
        "activation_function": LiteralAttribute(ACTIVATION_FUNCTIONS),
        "batch_normalization": BoolAttribute(),
    }

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        error_metric: ERROR_METRIC_TYPE = "mean-absolute-error",
        encoder_dimensions: list[int] = (64,),
        latent_space_dimension: int = 32,
        decoder_dimensions: list[int] = (64,),
        dropout_rate: float = 0.2,
        activation_function: ACTIVATION_FUNCTION_TYPE = "relu",
        batch_normalization: bool = True,
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
            supervision=Supervision.SEMI_SUPERVISED,
            error_metric=error_metric,
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
        self.encoder_dimensions = list(encoder_dimensions)
        self.latent_space_dimension = latent_space_dimension
        self.decoder_dimensions = list(decoder_dimensions)
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function
        self.batch_normalization = batch_normalization

    def _build_architecture(self, n_attributes: int) -> torch.nn.Module:
        return _AutoEncoderArchitecture(
            encoder=self._build_encoder(n_attributes),
            decoder=self._build_decoder(n_attributes),
        )

    def _build_encoder(self, n_attributes: int) -> torch.nn.Module:

        # Initialize the encoder
        encoder = torch.nn.Sequential()
        encoder.add_module("flatten", torch.nn.Flatten())

        # Initialize layer inputs and outputs
        inputs = [n_attributes * self.window_size_, *self.encoder_dimensions]
        outputs = [*self.encoder_dimensions, self.latent_space_dimension]

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

    def _build_decoder(self, n_attributes: int) -> torch.nn.Module:

        # Initialize the decoder
        decoder = torch.nn.Sequential()

        # Initialize layer inputs and outputs
        inputs = [self.latent_space_dimension, *self.decoder_dimensions]
        outputs = [*self.decoder_dimensions, n_attributes * self.window_size_]

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

        # Restore the dimensions of the window
        decoder.add_module(
            "unflatten", torch.nn.Unflatten(1, (self.window_size_, n_attributes))
        )

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
        return self.decoder(self.encoder(x)).reshape(x.shape)
