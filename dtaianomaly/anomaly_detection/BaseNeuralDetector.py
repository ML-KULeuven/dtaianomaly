import abc
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import (
    check_is_valid_window_size,
    compute_window_size,
    reverse_sliding_window,
)

_OPTIMIZER_TYPE = Literal["adam", "sgd"]
_COMPILE_MODE_TYPE = Literal[
    "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
]
_ACTIVATION_FUNCTION_TYPE = Literal["linear", "relu", "sigmoid", "tanh"]
_MODEL_PARAMETERS_TYPE = any


####################################################################
# BASE NEURAL DETECTOR
####################################################################


class BaseNeuralDetector(BaseDetector, abc.ABC):
    """
    Base class for neural anomaly detectors, based on PyTorch.

    This class implements the main functionality for training a model and
    detecting anomalies, including building the data loader, building the
    optimizer, and implementing the main train and evaluation loops. Extensions
    of this class should also implement methods to build the data set,
    the neural architecture, and how to train and evaluate on a single batch.

    Parameters
    ----------
    supervision: Supervision
        The type of supervision this anomaly detector requires.
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
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
    BaseNeuralForecastingDetector: Use a neural network to forecast the time
        series, and detect anomalies by measuring the difference with the
        actual observations.
    BaseNeuralReconstructionDetector: Use a neural network to reconstruct
        windows in the time series, and detect anomalies as windows that
        are incorrectly reconstructed.
    """

    _OPTIMIZERS: dict[_OPTIMIZER_TYPE, type[torch.optim.Optimizer]] = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    _ACTIVATION_FUNCTIONS: dict[_ACTIVATION_FUNCTION_TYPE, type[torch.nn.Module]] = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
    }

    # Preprocessing related parameters
    window_size: int | str
    stride: int
    standard_scaling: bool
    # Data loading related parameters
    batch_size: int
    data_loader_kwargs: dict[str, any] | None
    # Optimizer related parameters
    optimizer: (
        _OPTIMIZER_TYPE | Callable[[_MODEL_PARAMETERS_TYPE], torch.optim.Optimizer]
    )
    learning_rate: float
    # Model compilation
    compile_model: bool
    compile_mode: _COMPILE_MODE_TYPE
    # Training related parameters
    n_epochs: int
    loss_function: torch.nn.Module
    # General parameters
    device: str
    seed: int | None

    # Learned parameters
    window_size_: int
    optimizer_: torch.optim.Optimizer
    neural_network_: torch.nn.Module

    def __init__(
        self,
        supervision: Supervision,
        window_size: str | int,
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
        super().__init__(supervision)

        # Check preprocessing related parameters
        check_is_valid_window_size(window_size)
        if not isinstance(stride, int) or isinstance(stride, bool):
            raise TypeError("`stride` should be an integer")
        if stride < 1:
            raise ValueError("`stride` should be strictly positive")
        if not isinstance(standard_scaling, bool):
            raise TypeError("`standard_scaling` should be a bool")

        # Check the data related parameters
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("`batch_size` should be an integer")
        if batch_size < 1:
            raise ValueError("`batch_size` should be strictly positive")
        if data_loader_kwargs is not None:
            if not isinstance(data_loader_kwargs, dict):
                raise TypeError("`data_loader_kwargs` should be a dictionary")

        # Check the optimizer related parameters
        if not (isinstance(optimizer, str) or callable(optimizer)):
            raise TypeError("`optimizer` should be a string or callable")
        if optimizer not in self._OPTIMIZERS and not callable(optimizer):
            raise ValueError(
                f"Invalid value for `optimizer` given: '{optimizer}'. Valid options are {list(self._OPTIMIZERS.keys())}"
            )
        if not isinstance(learning_rate, (float, int)) or isinstance(
            learning_rate, bool
        ):
            raise TypeError("`learning_rate` should be numerical")
        if learning_rate <= 0:
            raise ValueError("`learning_rate` should be strictly positive")

        # Check the training related parameters
        if not isinstance(loss_function, torch.nn.Module):
            raise TypeError("`loss_function` should be a torch.nn.Module")
        if not isinstance(n_epochs, int) or isinstance(n_epochs, bool):
            raise TypeError("`n_epochs` should be an integer")
        if n_epochs < 1:
            raise ValueError("`n_epochs` should be strictly positive")

        # Check model compilation parameters
        if not isinstance(compile_model, bool):
            raise TypeError("`compile_model` should be a bool")
        if not isinstance(compile_mode, str):
            raise TypeError("`compile_mode` should be a string")
        if compile_mode not in [
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ]:
            raise ValueError(
                f"Invalid value for `compile_mode` given: '{compile_mode}'. Valid options are ['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']"
            )

        # Check the device
        if not isinstance(device, str):
            raise TypeError("`device` should be a string")
        # Check CUDA availability if it's a CUDA device
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError(
                    f"Cuda-device given ('{device}'), but no cuda is available!"
                )
            device_index = int(device.split(":")[1]) if ":" in device else None
            if device_index is not None and device_index >= torch.cuda.device_count():
                raise ValueError(
                    f"Cuda-index given ('{device_index}'), but only {torch.cuda.device_count()} are available!"
                )
        try:
            torch.device(device)  # Try to initialize a device
        except RuntimeError:  # Raise Value error instead for consistency
            raise ValueError(f"Invalid input device: {device}")

        # Initialize the variables
        self.window_size = window_size
        self.stride = stride
        self.standard_scaling = standard_scaling
        self.batch_size = batch_size
        self.data_loader_kwargs = data_loader_kwargs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.n_epochs = n_epochs
        self.device = device
        self.seed = seed

        # Test building the optimizer and the data loader
        self._build_data_loader(torch.utils.data.TensorDataset(torch.empty((10, 3))))
        self._build_optimizer(
            [torch.nn.Parameter(torch.randn(3, 3, requires_grad=True))]
        )

    @abc.abstractmethod
    def _build_dataset(self, X: np.ndarray) -> torch.utils.data.Dataset:
        """Abstract method to build the dataset."""

    @abc.abstractmethod
    def _build_architecture(self, n_attributes: int) -> torch.nn.Module:
        """Abstract method to build the architecture."""

    @abc.abstractmethod
    def _train_batch(self, batch: list[torch.Tensor]) -> float:
        """Abstract method to train the network on a single batch."""

    @abc.abstractmethod
    def _evaluate_batch(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """Abstract method to evaluate the network on a single batch."""

    def _set_seed(self) -> None:
        if self.seed is not None:
            torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _build_data_loader(
        self, dataset: torch.utils.data.Dataset, shuffle: bool = None
    ) -> torch.utils.data.DataLoader:
        kwargs = (
            {} if self.data_loader_kwargs is None else self.data_loader_kwargs.copy()
        )
        kwargs["batch_size"] = self.batch_size
        if shuffle is not None:
            kwargs["shuffle"] = shuffle
        return torch.utils.data.DataLoader(dataset, **kwargs)

    @staticmethod
    def _build_activation_function(
        activation_function: _ACTIVATION_FUNCTION_TYPE,
    ) -> torch.nn.Module:
        if activation_function in BaseNeuralDetector._ACTIVATION_FUNCTIONS:
            return BaseNeuralDetector._ACTIVATION_FUNCTIONS[activation_function]()
        raise ValueError(
            f"Invalid activation function given: '{activation_function}'. Valid options are {list(BaseNeuralDetector._ACTIVATION_FUNCTIONS.keys())}"
        )

    def _build_optimizer(
        self, model_parameters: _MODEL_PARAMETERS_TYPE
    ) -> torch.optim.Optimizer:
        if callable(self.optimizer):
            return self.optimizer(model_parameters)
        if self.optimizer in self._OPTIMIZERS:
            return self._OPTIMIZERS[self.optimizer](
                model_parameters, lr=self.learning_rate
            )
        raise ValueError(
            f"Invalid optimizer given: '{self.optimizer}'. Value values are {list(self._OPTIMIZERS.keys())} or a callable."
        )

    def _train(self, data_loader: torch.utils.data.DataLoader) -> None:
        # Set in train mode
        self.neural_network_.train(True)

        # Initialize variables to keep track of the state
        best_epoch_loss = torch.inf
        best_state_dict = None

        # Iterate over the epochs
        for epoch in range(self.n_epochs):

            # Iterate over the batches
            epoch_loss = 0
            for batch in data_loader:

                epoch_loss += self._train_batch(batch)

            # Update the best model so far
            if epoch_loss <= best_epoch_loss:
                best_epoch_loss = epoch_loss
                best_state_dict = self.neural_network_.state_dict()

        # Load the best model again
        self.neural_network_.load_state_dict(best_state_dict)

    def _evaluate(self, data_loader: torch.utils.data.DataLoader) -> np.array:
        # Set in evaluate mode
        self.neural_network_.eval()

        # Initialize array for the decision scores
        decision_scores = np.empty(len(data_loader.dataset))

        # Turn off the gradients
        with torch.no_grad():

            # Compute the decision score for each batch
            idx = 0
            for batch in data_loader:
                batch_scores = self._evaluate_batch(batch).cpu().numpy()
                decision_scores[idx : idx + batch_scores.shape[0]] = batch_scores
                idx += batch_scores.shape[0]

        # Return the computed decision score
        return decision_scores

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        # Set the seed
        self._set_seed()

        # Compute the window size
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)

        # Build the neural network
        data_loader = self._build_data_loader(self._build_dataset(X))
        self.neural_network_ = self._build_architecture(
            n_attributes=utils.get_dimension(X)
        ).to(self.device)
        self.optimizer_ = self._build_optimizer(
            model_parameters=self.neural_network_.parameters()
        )

        # Compile the model
        if self.compile_model:
            self.neural_network_.compile(mode=self.compile_mode)

        # Train the network
        self._train(data_loader)

    def _decision_function(self, X: np.ndarray) -> np.array:
        # Build the neural network
        data_loader = self._build_data_loader(self._build_dataset(X), shuffle=False)

        # Evaluate the model
        decision_scores = self._evaluate(data_loader)

        # Format the decision scores
        decision_scores = reverse_sliding_window(
            decision_scores, self.window_size_, self.stride, X.shape[0]
        )

        # Return the decision scores
        return decision_scores
