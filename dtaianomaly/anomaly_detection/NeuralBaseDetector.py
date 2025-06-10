import abc
from typing import Any, Iterator, List, Optional, Union

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import (
    check_is_valid_window_size,
    compute_window_size,
    reverse_sliding_window,
    sliding_window,
)


class NeuralBaseDetector(BaseDetector, abc.ABC):
    """
    Base class for neural networks. This class includes the functionality
    for initializing models with pytorch, and includes the general configuration
    of the neural networks such as preprocessing, data loading, optimizer, etc.
    Essentially all the functionality except for the architecture.
    """

    window_size: Union[str, int]
    stride: int
    scaling_method: Optional[str]
    batch_size: int
    data_loader_kwargs: Optional[dict]
    validation_size: float
    optimizer: str
    learning_rate: float
    optimizer_kwargs: Optional[dict]
    loss_function: torch.nn.Module
    n_epochs: int
    device: str

    window_size_: int
    optimizer_: torch.optim.Optimizer
    scaler_: Any
    neural_network_: torch.nn.Module

    def __init__(
        self,
        supervision: Supervision,
        window_size: Union[str, int],
        stride: int,
        scaling_method: str,
        batch_size: int,
        data_loader_kwargs: Optional[dict],
        validation_size: float,
        optimizer: str,
        learning_rate: float,
        optimizer_kwargs: Optional[dict],
        n_epochs: int,
        loss_function: torch.nn.Module,  # Forced to provide by the child
        device: str,
    ):
        super().__init__(supervision)

        # Check window related parameters
        check_is_valid_window_size(window_size)
        if not isinstance(stride, int) or isinstance(stride, bool):
            raise TypeError("`stride` should be an integer")
        if stride < 1:
            raise ValueError("`stride` should be strictly positive")

        # Check preprocessing related parameters
        if scaling_method is not None:
            if not isinstance(scaling_method, str):
                raise TypeError("`scaling_method` should be a string")
            if scaling_method not in ["standard-scaling", "min-max-scaling"]:
                raise ValueError(
                    f"Invalid value for 'scaling_method' given: '{scaling_method}'. Valid options are ['standard-scaling', 'min-max-scaling']"
                )

        # Check data loading related parameters
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("`batch_size` should be an integer")
        if batch_size < 1:
            raise ValueError("`batch_size` should be strictly positive")
        if data_loader_kwargs is not None:
            if not isinstance(data_loader_kwargs, dict):
                raise TypeError("`data_loader_kwargs` should be a dictionary")
        _initialize_data_loader(
            torch.utils.data.TensorDataset(torch.empty((10, 3)), torch.empty((10,))),
            batch_size,
            data_loader_kwargs,
        )
        if not isinstance(validation_size, (float, int)) or isinstance(
            validation_size, bool
        ):
            raise TypeError("`validation_size` should be an integer")
        if not 0 <= validation_size < 1:
            raise ValueError("`validation_size` should be in the range [0, 1[")

        # Check the optimizer related parameters
        if not isinstance(optimizer, str):
            raise TypeError("`optimizer` should be a string")
        if not isinstance(learning_rate, (float, int)) or isinstance(
            learning_rate, bool
        ):
            raise TypeError("`learning_rate` should be numerical")
        if learning_rate <= 0:
            raise ValueError("`learning_rate` should be strictly positive")
        if optimizer_kwargs is not None:
            if not isinstance(optimizer_kwargs, dict):
                raise TypeError("`optimizer_kwargs` should be a dictionary")
        _initialize_optimizer(
            [torch.nn.Parameter(torch.randn(3, 3, requires_grad=True))],
            optimizer,
            learning_rate,
            optimizer_kwargs,
        )  # Check if the optimizer can be initialized

        # Check the training related parameters
        if not isinstance(loss_function, torch.nn.Module):
            raise TypeError("`loss_function` should be a torch.nn.Module")
        if not isinstance(n_epochs, int) or isinstance(n_epochs, bool):
            raise TypeError("`n_epochs` should be an integer")
        if n_epochs < 1:
            raise ValueError("`n_epochs` should be strictly positive")

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
            print(device_index)
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
        self.scaling_method = scaling_method
        self.batch_size = batch_size
        self.data_loader_kwargs = data_loader_kwargs
        self.validation_size = validation_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.optimizer_kwargs = optimizer_kwargs
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        self.device = device

    @abc.abstractmethod
    def _initialize_dataset(self, X: np.ndarray) -> torch.utils.data.Dataset:
        """Initialize the dataset-object."""

    @abc.abstractmethod
    def _initialize_architecture(self, input_size: int) -> torch.nn.Module:
        """Initialize the architecture."""

    @abc.abstractmethod
    def _compute_decision_scores(
        self, data_loader: torch.utils.data.DataLoader
    ) -> np.array:
        """Compute the decision scores using the neural network."""

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:

        # Retrieve windows from the time series
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)
        windows = sliding_window(X, self.window_size_, self.stride)

        # Process the data
        windows_ = self._preprocess(windows)

        # Train validate split
        train_validate_split = int(windows.shape[0] * self.validation_size)
        train_loader = _initialize_data_loader(
            data_set=self._initialize_dataset(windows_[:train_validate_split]),
            batch_size=self.batch_size,
            data_loader_kwargs=self.data_loader_kwargs,
        )
        validation_loader = _initialize_data_loader(
            data_set=self._initialize_dataset(windows_[train_validate_split:]),
            batch_size=self.batch_size,
            data_loader_kwargs=self.data_loader_kwargs,
        )

        # Initialize the arhitecture
        self.neural_network_ = self._initialize_architecture(
            input_size=windows_.shape[1]
        ).to(self.device)

        # Initialize the optimizer
        self.optimizer_ = _initialize_optimizer(
            model_parameters=self.neural_network_.parameters(),
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
        )

        # Train the model
        # TODO this loop will probably need to be different (maybe in a function?)
        best_loss = float("inf")
        best_state_dict = None
        for epoch in range(self.n_epochs):

            # Iterate over all batches
            loss_per_batch = []
            for data, data_idx in train_loader:
                data = data.to(self.device).float()
                loss = self.loss_function(data, self.neural_network_(data))

                self.neural_network_.zero_grad()
                loss.backward()
                self.optimizer_.step()
                loss_per_batch.append(loss.item())

            # track the best model so far
            mean_loss_epoch = np.mean(loss_per_batch)
            if mean_loss_epoch <= best_loss:
                best_loss = mean_loss_epoch
                best_state_dict = self.neural_network_.state_dict()

        # Load the best model again
        self.neural_network_.load_state_dict(best_state_dict)

    def _decision_function(self, X: np.ndarray) -> np.array:

        # Retrieve windows from the time series
        windows = sliding_window(X, self.window_size_, self.stride)

        # Format the data
        windows_ = self._preprocess(windows)
        data_loader_kwargs = (
            self.data_loader_kwargs
        )  # Make a copy to not modify this object
        data_loader_kwargs["shuffle"] = False  # Ensure that the data is not shuffled
        data_loader = _initialize_data_loader(
            dataset=self._initialize_dataset(windows_),
            batch_size=self.batch_size,
            data_loader_kwargs=self.data_loader_kwargs,
        )

        # Enable evaluation mode
        self.neural_network_.eval()

        # Compute the decision scores
        decision_scores = self._compute_decision_scores(data_loader)

        # Format the decision scores
        decision_scores = reverse_sliding_window(
            decision_scores, self.window_size_, self.stride, X.shape[0]
        )

        # Return the decision scores
        return decision_scores

    def _preprocess(self, X: np.ndarray) -> np.ndarray:

        if self.scaling_method is None:
            return X

        if not hasattr(self, "scaler_"):
            if self.scaling_method == "standard-scaling":
                self.scaler_ = StandardScaler()
            elif self.scaling_method == "min-max-scaling":
                self.scaler_ = MinMaxScaler()
            else:
                raise ValueError(
                    f"Invalid scaling method given: '{self.scaling_method}'. Valid options are ['standard-scaling', 'min-max-scaling']"
                )
            self.scaler_.fit(X.T)

        return self.scaler_.transform(X.T).T


def _initialize_data_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    data_loader_kwargs: Optional[dict],
) -> torch.utils.data.DataLoader:

    # Initialize the data loader kwargs
    if data_loader_kwargs is None:
        data_loader_kwargs = {}
    else:
        # Make a copy to avoid side effects
        data_loader_kwargs = data_loader_kwargs.copy()

    # Overwrite the batch size
    data_loader_kwargs["batch_size"] = batch_size

    # Initialize the data loader
    return torch.utils.data.DataLoader(dataset, **data_loader_kwargs)


def _initialize_optimizer(
    model_parameters: Iterator[torch.nn.Parameter],
    optimizer: str,
    learning_rate: float,
    optimizer_kwargs: Optional[dict],
) -> torch.optim.Optimizer:
    # Initialize the optimizer kwargs if None is given
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    else:
        # Make a copy to avoid side effects
        optimizer_kwargs = optimizer_kwargs.copy()

    # Overwrite the learning rate
    optimizer_kwargs["lr"] = learning_rate

    # Initialize the optimizer
    if optimizer == "Adam":
        return torch.optim.Adam(model_parameters, **optimizer_kwargs)
    elif optimizer == "SGD":
        return torch.optim.SGD(model_parameters, **optimizer_kwargs)
    elif optimizer == "Adagrad":
        return torch.optim.Adagrad(model_parameters, **optimizer_kwargs)
    elif optimizer == "RMSprop":
        return torch.optim.RMSprop(model_parameters, **optimizer_kwargs)

    raise ValueError(
        f"Invalid optimizer given: '{optimizer}'. Value values are ['Adam', 'SGD', 'Adagrad', 'RMSprop']"
    )


def _initialize_activation_function(activation_name: str) -> torch.nn.Module:
    _ACTIVATIONS = {
        "linear": torch.nn.Identity(),
        "relu": torch.nn.ReLU(),
        "sigmoid": torch.nn.Sigmoid(),
        "tanh": torch.nn.Tanh(),
        "leaky-relu": torch.nn.LeakyReLU(),
    }
    if activation_name in _ACTIVATIONS.keys():
        return _ACTIVATIONS[activation_name]
    else:
        raise ValueError(
            f"Invalid activation given: '{activation_name}'. Valid options are {list(_ACTIVATIONS.keys())}."
        )


def _initialize_mlp(
    input_dimension: int,
    output_dimension: int,
    hidden_layer_dimension: List[int],
    activation_function: Union[str, List[str]],
    batch_normalization: Union[bool, List[bool]],
    batch_normalization_first_layer: bool,
    batch_normalization_last_layer: bool,
    dropout_rate: Union[float, List[float]],
    dropout_first_layer: bool,
    dropout_last_layer: bool,
) -> torch.nn.Module:

    # Initialize the model
    mlp = torch.nn.Sequential()

    # Configure the parameters
    nb_layers = len(hidden_layer_dimension) + 1
    layer_inputs = [input_dimension, *hidden_layer_dimension]
    layer_outputs = [*hidden_layer_dimension, output_dimension]
    if isinstance(activation_function, str):
        activation_function = [activation_function for _ in range(nb_layers)]
    elif len(activation_function) != nb_layers:
        raise ValueError(
            f"Invalid number of activation functions given! Expected {nb_layers} but received {len(activation_function)}"
        )
    if isinstance(batch_normalization, bool):
        batch_normalization = [batch_normalization for _ in range(nb_layers)]
    elif len(batch_normalization) != nb_layers:
        raise ValueError(
            f"Invalid number of batch normalization flags given! Expected {nb_layers} but received {len(batch_normalization)}"
        )
    if isinstance(dropout_rate, float):
        dropout_rate = [dropout_rate for _ in range(nb_layers)]
    elif len(dropout_rate) != nb_layers:
        raise ValueError(
            f"Invalid number of dropout rates given! Expected {nb_layers} but received {len(dropout_rate)}"
        )

    # Add all the layers to the sequential model
    for i in range(nb_layers):
        first_layer = i == 0
        last_layer = i == nb_layers - 1

        # Add the linear layer
        mlp.add_module(
            f"linear-{i} ({layer_inputs[i]}->{layer_outputs[i]})",
            torch.nn.Linear(layer_inputs[i], layer_outputs[i]),
        )

        # Add a batch normalization per layer
        if batch_normalization[i]:
            if first_layer and not batch_normalization_first_layer:
                pass  # Nothing should be done.
            elif last_layer and not batch_normalization_last_layer:
                pass  # Nothing should be done.
            else:
                mlp.add_module(
                    f"batch-norm-{i}", torch.nn.BatchNorm1d(layer_outputs[i])
                )

        # Add the activation function
        mlp.add_module(
            f"activation-{i}-{activation_function[i]}",
            _initialize_activation_function(activation_function[i]),
        )

        # Add the dropout layer
        if dropout_rate[i] > 0:
            if first_layer and not dropout_first_layer:
                pass  # Nothing should be done.
            elif last_layer and not dropout_last_layer:
                pass  # Nothing should be done.
            else:
                mlp.add_module(f"dropout-{i}", torch.nn.Dropout(dropout_rate[i]))

    # Return the mlp
    return mlp
