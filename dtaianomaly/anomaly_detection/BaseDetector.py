import abc
import enum
import os.path
import pickle
from pathlib import Path

import numpy as np
import scipy

from dtaianomaly.thresholding import ContaminationRateThreshold
from dtaianomaly.type_validation import AttributeValidationMixin
from dtaianomaly.utils import (
    CheckIsFittedMixin,
    PrintConstructionCallMixin,
    is_valid_array_like,
)


class Supervision(enum.Enum):
    """
    An enum for the different supervision types for anomaly detectors.
    Valid supervision types are:

    - ``Unsupervised``: the anomaly detector does not need any labels or training data.
    - ``Semi-supervised``: The anomaly detector requires *normal* training data, but no training labels.
    - ``Supervised``: The anomaly detector requires both training data and training labels. The training data may contain anomalies.
    """

    UNSUPERVISED = 1
    SEMI_SUPERVISED = 2
    SUPERVISED = 3


class BaseDetector(
    PrintConstructionCallMixin, CheckIsFittedMixin, AttributeValidationMixin
):
    """
    Abstract base class for time series anomaly detection.

    This base class defines method signatures to build
    specific anomaly detectors. User-defined detectors
    can be used throughout the ``dtaianomaly`` by extending
    this base class.

    Parameters
    ----------
    supervision: Supervision
        The type of supervision this anomaly detector requires.
    """

    supervision: Supervision

    def __init__(self, supervision: Supervision):
        if not isinstance(supervision, Supervision):
            raise TypeError("'supervision' should be a valid 'Supervision' type!")
        self.supervision = supervision

    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> "BaseDetector":
        """
        Abstract method, fit this detector to the given data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.
        y: array-like, default=None
            Ground-truth information.

        Returns
        -------
        self: BaseDetector
            Returns the instance itself.
        """
        # Check the input
        if not is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")

        # Fit the detector
        self._fit(np.asarray(X), y, **kwargs)

        # Return self
        return self

    @abc.abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        """Effectively fit this detector."""

    def decision_function(self, X: np.ndarray) -> np.array:
        """
        Abstract method, compute anomaly scores.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.

        Returns
        -------
        decision_scores: array-like of shape (n_samples)
            The computed anomaly scores.
        """
        # Check input
        if not is_valid_array_like(X):
            raise ValueError(f"Input must be numerical array-like")

        # Check if fitted
        self.check_is_fitted()

        # Compute decision scores
        return self._decision_function(np.asarray(X))

    @abc.abstractmethod
    def _decision_function(self, X: np.ndarray) -> np.array:
        """Effectively compute the decision function."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly probabilities

        Estimate the probability of a sample of `X` being anomalous,
        based on the anomaly scores obtained from `decision_function`
        by rescaling them to the range of [0, 1] via min-max scaling.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.

        Returns
        -------
        anomaly_scores: array-like of shape (n_samples)
            1D array with the same length as `X`, with values
            in the interval [0, 1], in which a higher value
            implies that the instance is more likely to be
            anomalous.

        Raises
        ------
        ValueError
            If `scores` is not a valid array.
        ValueError
            If the prediction scores from 'decision_function' are constant, but not
            in the interval [0, 1], because these values can not unambiguously be
            transformed to an anomaly probability.
        """
        if not is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")

        raw_scores = self.decision_function(X)

        min_score = np.nanmin(raw_scores)
        max_score = np.nanmax(raw_scores)
        if min_score == max_score:
            if not (0.0 <= min_score <= 1.0):
                raise ValueError(
                    "The predicted anomaly scores are constant, but not in the interval [0, 1]. "
                    "It is not clear how to transform these unambiguously to anomaly-probabilities!"
                )
            return raw_scores

        else:
            return (raw_scores - min_score) / (max_score - min_score)

    def predict_confidence(
        self,
        X: np.ndarray,
        X_train: np.ndarray = None,
        contamination: float = 0.05,
        decision_scores_given: bool = False,
    ):
        """
        Predict the confidence of the anomaly scores on the test given test data.

        This method implements ExCeeD [perini2020quantifying]_ (Example-wise Confidence
        of anomaly Detectors) to estimate the confidence. ExCeed transforms the predicted
        decision scores to probability estimates using a Bayesian approach, which enables
        to assign a confidence score to each prediction which captures the uncertainty
        of the anomaly detector in that prediction.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            The test time series for which the confidence of anomaly scores
            should be predicted.
        X_train: array-like of shape (n_samples_train, n_attributes), default=None
            The training time series, which can be used as reference. If
            ``X_train=None``, the test set is used as reference set.
        contamination: float, default=0.05
            The (estimated) contamination rate for the data, i.e., the expected
            percentage of anomalies.
        decision_scores_given: bool, default=False
            Whether the given ``X`` and ``X_train`` represent time series data
            or decision scores. If ``decision_scores_given=False`` (default),
            then the given arrays are interpreted as time series. Otherwise,
            they are interpreted as decision scores, as computed by
            ``decision_function()``.

        Returns
        -------
        confidence: array-like of shape (n_samples)
            The confidence of this anomaly detector in each prediction in the
            given test time series.

        References
        ----------
        .. [perini2020quantifying] Perini, L., Vercruyssen, V., Davis, J. Quantifying
           the Confidence of Anomaly Detectors in Their Example-Wise Predictions. In:
           Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2020.
           Springer, Cham, doi: `10.1007/978-3-030-67664-3_14 <https://doi.org/10.1007/978-3-030-67664-3_14>`_.
        """
        # Set the decision scores
        if decision_scores_given:
            if len(X.shape) > 1:
                raise ValueError(
                    "In the 'predict_confidence()' method, it was indicated that the decision scores are provided "
                    "as X (decision_scores_given=True), but the shape of X does not correspond to the shape of decision"
                    f"scores: {X.shape}!"
                )
            if X_train is not None and len(X_train.shape) > 1:
                raise ValueError(
                    "In the 'predict_confidence()' method, it was indicated that the decision scores are provided "
                    "as X (decision_scores_given=True), but the shape of X_train does not correspond to the shape of decision"
                    f"scores: {X.shape}!"
                )
            decision_scores = X
            decision_scores_train = X_train if X_train is not None else decision_scores

        else:
            # Compute the decision scores
            decision_scores = self.decision_function(X)
            decision_scores_train = (
                self.decision_function(X_train)
                if X_train is not None
                else decision_scores
            )

        # Convert the decision scores to binary predictions
        prediction = ContaminationRateThreshold(
            contamination_rate=contamination
        ).threshold(decision_scores)

        # Apply the ExCeed method (https://github.com/Lorenzo-Perini/Confidence_AD/blob/master/ExCeeD.py)
        n = decision_scores.shape[0]

        count_instances = np.vectorize(
            lambda x: np.count_nonzero(decision_scores_train <= x)
        )
        n_instances = count_instances(decision_scores)

        prob_func = np.vectorize(lambda x: (1 + x) / (2 + n))
        posterior_prob = prob_func(
            n_instances
        )  # Outlier probability according to ExCeeD

        conf_func = np.vectorize(
            lambda p: 1 - scipy.stats.binom.cdf(n - int(n * contamination), n, p)
        )
        exWise_conf = conf_func(posterior_prob)
        np.place(
            exWise_conf, prediction == 0, 1 - exWise_conf[prediction == 0]
        )  # if the example is classified as normal, use 1 - confidence.

        return exWise_conf

    def save(self, path: str | Path) -> None:
        """
        Save detector to disk as a pickle file with extension `.dtai`. If the given
        path consists of multiple subdirectories, then the not existing subdirectories
        are created.

        Parameters
        ----------
        path: str or Path
            Location where to store the detector.
        """
        # Add the '.dtai' extension
        if Path(path).suffix != ".dtai":
            path = f"{path}.dtai"

        # Create the subdirectory, if it doesn't exist
        if not os.path.exists(Path(path).parent):
            os.makedirs(Path(path).parent)

        # Effectively write the anomaly detector to disk
        with open(path, "wb") as f:
            pickle.dump(self, f)


def load_detector(path: str | Path) -> BaseDetector:
    """
    Load a detector from disk.

    Warning: method relies on pickle. Only load trusted files!

    Parameters
    ----------
    path: str or Path
        Location of the stored detector.

    Returns
    -------
    detector: BaseDetector
        The loaded detector.
    """
    with open(path, "rb") as f:
        detector = pickle.load(f)
    return detector
