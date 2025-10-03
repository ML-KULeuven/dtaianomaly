import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample
from sktime.transformations.panel.rocket import Rocket

from dtaianomaly.anomaly_detection._BaseDetector import BaseDetector, Supervision
from dtaianomaly.type_validation import (
    BoolAttribute,
    IntegerAttribute,
    NoneAttribute,
    WindowSizeAttribute,
)
from dtaianomaly.utils import get_dimension
from dtaianomaly.windowing import (
    WINDOW_SIZE_TYPE,
    compute_window_size,
    reverse_sliding_window,
)

__all__ = ["ROCKAD"]


class ROCKAD(BaseDetector):
    """
    Detect anomalies in time series subsequences with ROCKAD :cite:`theissler2023rockad`.

    ROCKAD uses the ROCKET transformation :cite:`dempster2020rocket` as an unsupervised
    feature extractor from time series subsequences. Then, a bagging-based ensemble of
    k-NN models using the ROCKET-features is used to detect anomalous time series
    subsequences, in which the anomaly score of each individual instance is computed as
    the distance to the k-th nearest neighbor within each bagging subset. As discussed
    by :cite:t:`theissler2023rockad`, first applying a power-transform and then standard
    scaling the ROCKET features improves separation of the normal and anomalous sequences.

    Parameters
    ----------
    window_size : int or str
        The window size, the length of the subsequences that will be detected as anomalies. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride : int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    n_kernels : int, default=100,
        The number of kernels to use in the ROCKET-transformation.
    power_transform : bool, default=True
        Whether to perform a power-transformation or not.
    n_estimators : int, default=10
        The number of k-NN estimators to include in the detection ensemble.
    n_neighbors : int, default=5
        The number of neighbors to use for the nearest neighbor queries.
    metric : str, default='euclidean'
        Distance metric for distance computations. any metric of scikit-learn and
        scipy.spatial.distance can be used.
    n_jobs : int, default=1
        The number of jobs to use, which is passed to the scikit-learn components.
    seed : int, default=None
        The random seed used to split the data and initialise the kernels.

    Attributes
    ----------
    window_size_ : int
        The effectively used window size for detecting anomalies.
    rocket_ : Rocket
        The ``sktime`` Rocket transformer object.
    power_transformer_ : PowerTransformer
        The ``sklearn`` power transformer object. The object will only be fitted if
        ``power_transform=True``.
    nearest_neighbors_ : list of NearestNeighbors
        The fitted nearest neighbor instances on a different subset of the instances.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import ROCKAD
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> rockad = ROCKAD(64, seed=0).fit(x)
    >>> rockad.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([5.30759668, 5.25451016, 4.80149563, ..., 3.40483896, 3.72443581,
           3.74599171])
    """

    window_size: WINDOW_SIZE_TYPE
    stride: int
    n_kernels: int
    power_transform: bool
    n_estimators: int
    n_neighbors: int
    metric: str
    n_jobs: int
    seed: int | None

    window_size_: int
    rocket_: Rocket
    power_transformer_: PowerTransformer
    nearest_neighbors_: list[NearestNeighbors]

    attribute_validation = {
        "window_size": WindowSizeAttribute(),
        "stride": IntegerAttribute(1),
        "n_kernels": IntegerAttribute(1),
        "n_estimators": IntegerAttribute(1),
        "n_neighbors": IntegerAttribute(1),
        "n_jobs": IntegerAttribute(1),
        "power_transform": BoolAttribute(),
        "seed": IntegerAttribute() | NoneAttribute(),
    }

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        stride: int = 1,
        n_kernels: int = 100,
        power_transform: bool = True,
        n_estimators: int = 10,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        n_jobs: int = 1,
        seed: int = None,
    ):
        super().__init__(Supervision.UNSUPERVISED)
        self.window_size = window_size
        self.stride = stride
        self.n_kernels = n_kernels
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        self.power_transform = power_transform
        self.seed = seed

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)
        windows = self._sliding_window(X)

        # Apply ROCKET
        self.rocket_ = Rocket(
            num_kernels=self.n_kernels, n_jobs=self.n_jobs, random_state=self.seed
        )
        features = self.rocket_.fit_transform(windows)

        # Apply power-transformation
        self.power_transformer_ = PowerTransformer(standardize=True)
        if self.power_transform:
            features = self.power_transformer_.fit_transform(features)

        # Train the ensemble of nearest neighbor models.
        self.nearest_neighbors_ = []
        for i in range(self.n_estimators):

            # Define a seed for this estimator
            seed_ = self.seed
            if self.seed is not None:
                seed_ += i

            # Initialize the NN object
            nearest_neighbors = NearestNeighbors(
                n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=self.n_jobs
            )

            # Sample a subset to bootstrap
            resample(
                features,
                replace=True,
                n_samples=None,
                random_state=seed_,
                stratify=None,
            )

            # Fit the nearest neighbor instance on the sample
            nearest_neighbors.fit(features)
            self.nearest_neighbors_.append(nearest_neighbors)

    def _decision_function(self, X: np.ndarray) -> np.array:

        # Create the sliding windows
        windows = self._sliding_window(X)

        # Extract the ROCKET features
        features = self.rocket_.transform(windows)

        # Apply power transform
        if self.power_transform:
            features = self.power_transformer_.transform(features)

        # Compute the k-th nearest neighbor distance to each ensemble item
        nearest_neighbors_distances = np.empty(
            shape=(windows.shape[0], self.n_estimators)
        )
        for i, nearest_neighbors in enumerate(self.nearest_neighbors_):
            nearest_neighbors_distances[:, i] = nearest_neighbors.kneighbors(features)[
                0
            ][:, -1]

        # Aggregate the scores
        decision_scores = nearest_neighbors_distances.mean(axis=1)
        return reverse_sliding_window(
            decision_scores, self.window_size_, self.stride, X.shape[0]
        )

    def _sliding_window(self, X: np.ndarray) -> np.ndarray:
        """Custom method to format the windows according to sktime format."""
        X = X.reshape(X.shape[0], get_dimension(X))
        windows = [
            X[t : t + self.window_size_, :].T
            for t in range(0, X.shape[0] - self.window_size_, self.stride)
        ]
        windows.append(X[-self.window_size_ :].T)
        return np.array(windows)
