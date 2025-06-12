import abc
from typing import Literal, Optional

import numpy as np
import torch
from sklearn.preprocessing import scale

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import (
    check_is_valid_window_size,
    compute_window_size,
    reverse_sliding_window,
    sliding_window,
)

_OPTIMIZER_TYPE = Literal["adam", "sgd", "adamgrad", "rmsprop"]
_COMPILE_MODE_TYPE = Literal[
    "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
]
_ACTIVATION_FUNCTION_TYPE = Literal["linear", "relu", "sigmoid", "tanh"]


class BaseNeuralDetector(BaseDetector, abc.ABC):
    """
    Base class for neural networks. This class includes the functionality
    for initializing models with pytorch, and includes the general configuration
    of the neural networks such as preprocessing, data loading, optimizer, etc.
    Essentially all the functionality except for the architecture.
    """

    _OPTIMIZERS: dict[_OPTIMIZER_TYPE, type[torch.optim.Optimizer]] = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adagrad": torch.optim.Adagrad,
        "rmsprop": torch.optim.RMSprop,
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
    optimizer: _OPTIMIZER_TYPE
    learning_rate: float
    optimizer_kwargs: dict[str, any] | None
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

    def __init__(
        self,
        supervision: Supervision,
        window_size: str | int,
        stride: int,
        standard_scaling: bool,
        batch_size: int,
        data_loader_kwargs: dict[str, any] | None,
        optimizer: _OPTIMIZER_TYPE,
        learning_rate: float,
        optimizer_kwargs: dict[str, any] | None,
        compile_model: bool,
        compile_mode: _COMPILE_MODE_TYPE,
        n_epochs: int,
        loss_function: torch.nn.Module,
        device: str,
        seed: int | None,
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
        if not isinstance(optimizer, str):
            raise TypeError("`optimizer` should be a string")
        if optimizer not in self._OPTIMIZERS:
            raise ValueError(
                f"Invalid value for `optimizer` given: '{optimizer}'. Valid options are {list(self._OPTIMIZERS.keys())}"
            )
        if not isinstance(learning_rate, (float, int)) or isinstance(
            learning_rate, bool
        ):
            raise TypeError("`learning_rate` should be numerical")
        if learning_rate <= 0:
            raise ValueError("`learning_rate` should be strictly positive")
        if optimizer_kwargs is not None:
            if not isinstance(optimizer_kwargs, dict):
                raise TypeError("`optimizer_kwargs` should be a dictionary")

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
        self.optimizer_kwargs = optimizer_kwargs
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

    def _build_optimizer(self, model_parameters) -> torch.optim.Optimizer:
        if self.optimizer in self._OPTIMIZERS:
            kwargs = (
                {} if self.optimizer_kwargs is None else self.optimizer_kwargs.copy()
            )
            kwargs["lr"] = self.learning_rate
            return self._OPTIMIZERS[self.optimizer](model_parameters, **kwargs)
        raise ValueError(
            f"Invalid optimizer given: '{self.optimizer}'. Value values are {list(self._OPTIMIZERS.keys())}"
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

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
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


class TimeSeriesDataset(torch.utils.data.Dataset):

    X: np.ndarray
    window_size: int
    stride: int
    standard_scaling: bool
    device: str

    def __init__(
        self,
        X: np.ndarray,
        window_size: int,
        stride: int,
        standard_scaling: bool,
        device: str,
    ):
        self.X = X
        self.window_size = window_size
        self.stride = stride
        self.standard_scaling = standard_scaling
        self.device = device

    def _scale(self, window: np.ndarray) -> np.ndarray:
        if self.standard_scaling:
            window = scale(window)
        return window


class ForecastDataset(TimeSeriesDataset):

    forecast_length: int

    def __init__(
        self,
        X: np.ndarray,
        window_size: int,
        stride: int,
        standard_scaling: bool,
        device: str,
        forecast_length: int,
    ):
        super().__init__(X, window_size, stride, standard_scaling, device)
        self.forecast_length = forecast_length

    def __len__(self) -> int:
        return (
            int(
                np.ceil(
                    (self.X.shape[0] - self.forecast_length - self.window_size)
                    / self.stride
                )
            )
            + 1
        )

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        # Find the indices
        start = idx * self.stride
        end = start + self.window_size + self.forecast_length

        # Handle the last window
        if end >= self.X.shape[0]:
            end = self.X.shape[0]
            start = end - self.window_size - self.forecast_length

        # Retrieve the window
        window = self._scale(self.X[start:end])

        # Split in history and future
        history = window[: self.window_size].reshape(-1)
        future = window[-self.forecast_length :].reshape(-1)

        # Return the data
        return torch.Tensor(history).to(self.device), torch.Tensor(future).to(
            self.device
        )


class ReconstructionDataset(TimeSeriesDataset):

    def __len__(self) -> int:
        return int(np.ceil((self.X.shape[0] - self.window_size) / self.stride)) + 1

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        start = idx * self.stride
        end = start + self.window_size

        # Handle the last window
        if end >= self.X.shape[0]:
            end = self.X.shape[0]
            start = end - self.window_size

        # Retrieve the window
        window = self._scale(self.X[start:end]).reshape(-1)

        # Return the data
        return [torch.Tensor(window).to(self.device)]
