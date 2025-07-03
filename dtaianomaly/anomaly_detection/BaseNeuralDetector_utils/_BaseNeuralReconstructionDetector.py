import abc
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.BaseNeuralDetector import (
    _COMPILE_MODE_TYPE,
    _MODEL_PARAMETERS_TYPE,
    _OPTIMIZER_TYPE,
    BaseNeuralDetector,
)
from dtaianomaly.anomaly_detection.BaseNeuralDetector_utils._TimeSeriesDataSet import (
    ReconstructionDataset,
)


class BaseNeuralReconstructionDetector(BaseNeuralDetector, abc.ABC):
    """
    Base class for reconstruction-based neural anomaly detectors.

    Reconstruction-based anomaly detection detect anomalies by learning
    to reconstruct the data. Specifically, the neural network takes as
    input a sliding window of the time series, and learns to output the
    exactly same data. Given a normal time series enable to learn the
    normal behaviors, and as a consequence it is possible to accurately
    reconstruct the data. However, anomalous subsequences, which were
    not seen during training, can not be accurately reconstructed, and
    will have a larger reconstruction error as a consequence.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    supervision: Supervision, default=Supervision.SEMI_SUPERVISED
        The type of supervision this anomaly detector requires.
    error_metric: {"mean-absolute-error", "mean-squared-error"}, default="mean-absolute-error"
        The error measure between the reconstructed window and the original window.
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
    AutoEncoder: An implementation of this class using an feed-forward auto encoder.
    """

    error_metric: Literal["mean-absolute-error", "mean-squared-error"]

    def __init__(
        self,
        window_size: str | int,
        supervision: Supervision = Supervision.SEMI_SUPERVISED,
        error_metric: Literal[
            "mean-absolute-error", "mean-squared-error"
        ] = "mean-absolute-error",
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
            supervision=supervision,
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

        if not isinstance(error_metric, str):
            raise TypeError("`error_metric` should be a string")
        if error_metric not in ["mean-absolute-error", "mean-squared-error"]:
            raise ValueError(
                f"Unknown error_metric '{error_metric}'. Valid options are ['mean-absolute-error', 'mean-squared-error']"
            )

        self.error_metric = error_metric

    def _build_dataset(self, X: np.ndarray) -> torch.utils.data.Dataset:
        return ReconstructionDataset(
            X=X,
            window_size=self.window_size_,
            stride=self.stride,
            standard_scaling=self.standard_scaling,
            device=self.device,
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
            return torch.mean(
                (reconstructed - data) ** 2, dim=tuple(range(1, reconstructed.ndim))
            )
        if self.error_metric == "mean-absolute-error":
            return torch.mean(
                torch.abs(reconstructed - data), dim=tuple(range(1, reconstructed.ndim))
            )

        # Raise an error if invalid metric is given
        raise ValueError(
            f"Unknown error_metric '{self.error_metric}'. Valid options are ['mean-squared-error', 'mean-absolute-error']"
        )
