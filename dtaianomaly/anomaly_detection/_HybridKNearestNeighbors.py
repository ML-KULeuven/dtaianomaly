import random
from typing import Literal

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from dtaianomaly.anomaly_detection._BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection._BaseNeuralDetector import (
    ACTIVATION_FUNCTION_TYPE,
    ACTIVATION_FUNCTIONS,
    BaseNeuralDetector,
)
from dtaianomaly.type_validation import (
    FloatAttribute,
    IntegerAttribute,
    ListAttribute,
    LiteralAttribute,
    NoneAttribute,
    WindowSizeAttribute,
)
from dtaianomaly.windowing import (
    WINDOW_SIZE_TYPE,
    compute_window_size,
    reverse_sliding_window,
    sliding_window,
)

__all__ = ["HybridKNearestNeighbors"]


class HybridKNearestNeighbors(BaseDetector):
    """
    Anomaly detection based on a hybrid K-NN with AutoEncoder embedding :cite:`song2017hybrid`.

    Combine an autoencoder model to learn a latent space representation of the subsequences
    with an ensemble of K-NN instances. At training, an autoencoder is fitted using subsequences
    from the training time series to embed them into a latent space. Then, the latent space
    embeddings are split into multiple subsets in a bagging-like manner. For each subset, a
    K-NN instance is initialized, and the average K-th nearest neighbor distance of each sample
    is computed across all the subsets. At prediction time, the autoencoder creates the latent
    space embedding of the sequences, and the average K-th nearest neighbor distance of each
    test-sequence across all subsets is computed. The anomaly score of a test-sequence in regard
    to a subset is then computed as the proportion of samples in the subset that have
    a smaller average distance. The final anomaly score equals the average anomaly score
    across all subsets.

    Parameters
    ----------
    window_size : int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride : int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    n_neighbors : int, default=5
        The number of neighbors to use for the nearest neighbor queries.
    n_estimators : int, default=100
        The number of K-NN instance and consequently subsets to use.
    max_samples : int or float, default='auto'
        The number of samples to draw for each subset:

        - if ``int``: Draw at most ``max_samples`` samples.
        - if ``float``: Draw at most ``max_samples`` percentage of the samples.
        - if ``'auto'``: Set ``max_samples=n_windows/n_estimators``.

    metric : str, default='euclidean'
        Distance metric for distance computations. Any metric of scikit-learn and
        scipy.spatial.distance can be used.
    hidden_layer_dimensions : list of ints, default=[64]
        The number of neurons in each hidden layer of the encoder and decoder. The given list
        equals the ordered sequence of neurons in the encoder. The layers in the decoder has
        the same dimensions but mirrored.
    latent_space_dimension : int default=16
        The dimension of the latent space.
    activation_function : {"linear", "relu", "sigmoid", "tanh"} default="relu"
        The activation function to use at the end of each layer.
    batch_size : int, default=32
        The size of the batches to feed to the network.
    n_epochs : int, default=5
        The number of epochs for which the neural network should be trained.
    learning_rate : float, default=1e-3
        The learning rate to use for training the network.
    device : str, default="cpu"
        The device on which te neural network should be trained.
        For more information, see: https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-device.
    seed : int, default=None
        The seed used for training the autoencoder and sampling the subsets.

    Attributes
    ----------
    window_size_ : int
        The effectively used window size for this anomaly detector
    nearest_neighbors_ : list[NearestNeighbors]
        The scikit-learn nearest neighbor instances for each subset
    g_ : list[np.ndarray]
        For each subset, a vector containing the average distance of each
        sample within the subset to its K-th nearest neighbor across all
        other subsets.
    auto_encoder_ : torch.nn.Module
        The auto encoder used to embed the windows in the time series.

    Notes
    -----
    - :cite:t:`song2017hybrid` assigns a binary anomaly score for a test-sequence
      with regards to a subset by checking if the number of sequences within the
      subset that have a greater distance than the test-sequence exceeds some
      predefined threshold :math:`\\alpha` (Equation 10). We drop this part to
      reduce the number of parameters and to allow for a more fine-grained anomaly
      score computation.
    - Currently, a very simple feed-forward auto encoder is implemented. If you want
      to use a more advanced model, you can extend this class and overwrite the
      :py:meth:`~dtaianomaly.anomaly_detection.HybridKNearestNeighbors.build_auto_encoder`
      which returns a torch.nn.Module with ``fit(windows)`` and a ``encode(windows)``
      methods. The given windows are those computed by :py:meth:`~dtaianomaly.windowing.sliding_window`.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import HybridKNearestNeighbors
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> hybrid_knn = HybridKNearestNeighbors(64, seed=0).fit(x)
    >>> hybrid_knn.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.12284644, 0.38202247, 0.43220974, ..., 0.83470662, 0.81722846,
           0.85243446]...)
    """

    window_size: WINDOW_SIZE_TYPE
    stride: int
    n_neighbors: int
    n_estimators: int
    max_samples: float | int | Literal["auto"]
    metric: str
    hidden_layer_dimensions: list[int]
    latent_space_dimension: int
    activation_function: ACTIVATION_FUNCTION_TYPE
    n_epochs: int
    learning_rate: float
    device: str
    seed: int | None

    window_size_: int
    nearest_neighbors_: list[NearestNeighbors]
    g_: list[np.ndarray]
    auto_encoder_: torch.nn.Module

    attribute_validation = {
        "window_size": WindowSizeAttribute(),
        "stride": IntegerAttribute(1),
        "n_neighbors": IntegerAttribute(minimum=1),
        "n_estimators": IntegerAttribute(minimum=1),
        "max_samples": IntegerAttribute(minimum=1)
        | FloatAttribute(0.0, 1.0, inclusive_minimum=False)
        | LiteralAttribute("auto"),
        "hidden_layer_dimensions": ListAttribute(IntegerAttribute(minimum=1)),
        "latent_space_dimension": IntegerAttribute(minimum=1),
        "activation_function": LiteralAttribute(ACTIVATION_FUNCTIONS),
        "batch_size": IntegerAttribute(minimum=1),
        "n_epochs": IntegerAttribute(minimum=1),
        "learning_rate": FloatAttribute(minimum=0.0, inclusive_minimum=False),
        "seed": IntegerAttribute() | NoneAttribute(),
    }

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        stride: int = 1,
        n_neighbors: int = 5,
        n_estimators: int = 3,
        max_samples: float | int = "auto",
        metric: str = "euclidean",
        hidden_layer_dimensions: list[int] = (64,),
        latent_space_dimension: int = 16,
        activation_function: ACTIVATION_FUNCTION_TYPE = "relu",
        batch_size: int = 32,
        n_epochs: int = 5,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        seed: int = None,
    ):
        super().__init__(Supervision.SEMI_SUPERVISED)
        self.window_size = window_size
        self.stride = stride
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.metric = metric
        self.hidden_layer_dimensions = list(hidden_layer_dimensions)
        self.latent_space_dimension = latent_space_dimension
        self.activation_function = activation_function
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.seed = seed

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        # Compute the window size
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)
        windows = sliding_window(X, self.window_size_, self.stride)

        self.auto_encoder_ = self.build_auto_encoder(windows)
        self.auto_encoder_.fit(windows)
        X_ = self.auto_encoder_.encode(windows)

        # Create the subsets
        subsets = self._create_subsets(X_)

        # Fit the nearest neighbor instances
        self.nearest_neighbors_ = [
            NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(
                subset
            )
            for subset in subsets
        ]

        # Compute the g-array
        self.g_ = [self._compute_g(subset).reshape(1, -1) for subset in subsets]

    def _decision_function(self, X: np.ndarray) -> np.array:
        # Compute the windows
        windows = sliding_window(X, self.window_size_, self.stride)
        X_ = self.auto_encoder_.encode(windows)

        # Compute the G-values
        g = self._compute_g(X_).reshape(-1, 1)

        # Compute the p-values (Equation 9)
        p = np.array(
            [
                # The paper states less_equal, but then detects anomalies based on the smallest values.
                # We compute greater to have higher scores for more anomalous points.
                np.greater(g, self.g_[i]).mean(axis=1)
                for i in range(self.n_estimators)
            ]
        )

        # Apply reverse sliding window
        return reverse_sliding_window(
            p.mean(axis=0), self.window_size_, self.stride, X.shape[0]
        )

    def _create_subsets(self, X: np.ndarray) -> list[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        if self.max_samples == "auto":
            nb_samples = X.shape[0] / self.n_estimators
        elif isinstance(self.max_samples, float):
            nb_samples = self.max_samples * X.shape[0]
        else:
            nb_samples = self.max_samples

        return [
            X[rng.choice(X.shape[0], size=int(nb_samples), replace=False)]
            for _ in range(self.n_estimators)
        ]

    def _compute_g(self, X: np.ndarray) -> np.array:
        # Equation 8
        return np.array(
            [
                nearest_neighbors.kneighbors(X)[0][:, -1]
                for nearest_neighbors in self.nearest_neighbors_
            ]
        ).mean(axis=0)

    def build_auto_encoder(self, windows: np.ndarray) -> torch.nn.Module:
        """
        Build an auto encoder.

        Build an auto encoder module that takes as input the given windows
        and learns to reconstruct them.

        Parameters
        ----------
        windows : array-like of shape (n_windows, n_attributes x window_size)
            The windows that will be fed to the auto encoder.

        Returns
        -------
        torch.nn.Module
            Returns a torch neural network module which will take as input
            the windows and learns to reconstruct them. The torch module
            has a ``.fit(windows)`` method to learn the weights and a
            ``.encode(windows)`` method to transform the windows into a
            latent space embedding.
        """
        return _AutoEncoder(
            input_size=windows.shape[1],
            hidden_layer_dimensions=self.hidden_layer_dimensions,
            latent_space_dimension=self.latent_space_dimension,
            activation_function=self.activation_function,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            device=self.device,
            seed=self.seed,
        )


class _AutoEncoder(torch.nn.Module):

    encoder: torch.nn.Sequential
    decorator: torch.nn.Sequential
    batch_size: int
    n_epochs: int
    learning_rate: float
    device: str
    seed: int | None

    def __init__(
        self,
        input_size: int,
        hidden_layer_dimensions: list[int],
        latent_space_dimension: int,
        activation_function: ACTIVATION_FUNCTION_TYPE,
        batch_size: int,
        n_epochs: int,
        learning_rate: float,
        device: str,
        seed: int | None,
    ):
        super().__init__()
        self.seed = seed
        self._set_seed()

        encoder_layers = []
        prev_d = input_size
        for d in hidden_layer_dimensions:
            encoder_layers.append(torch.nn.Linear(prev_d, d))
            encoder_layers.append(
                BaseNeuralDetector._build_activation_function(activation_function)
            )
            prev_d = d
        encoder_layers.append(
            torch.nn.Linear(hidden_layer_dimensions[-1], latent_space_dimension)
        )
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_d = latent_space_dimension
        for d in reversed(hidden_layer_dimensions):
            decoder_layers.append(torch.nn.Linear(prev_d, d))
            decoder_layers.append(
                BaseNeuralDetector._build_activation_function(activation_function)
            )
            prev_d = d
        decoder_layers.append(torch.nn.Linear(hidden_layer_dimensions[0], input_size))
        self.decoder = torch.nn.Sequential(*decoder_layers)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.to(device)

    def _set_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, windows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.Tensor(windows).to(self.device)
            encoded = self.encoder(x)
        return encoded.cpu().numpy()

    def fit(self, windows: np.ndarray):
        self._set_seed()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(torch.Tensor(windows))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        self.train()
        for epoch in range(self.n_epochs):
            for batch in dataloader:
                batch = batch[0].to(self.device)
                self.zero_grad()
                loss = criterion(self.forward(batch), batch)
                loss.backward()
                optimizer.step()
        self.eval()
