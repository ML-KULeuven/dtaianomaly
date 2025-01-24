import numpy as np
from typing import Optional, List
from dtaianomaly.utils import is_valid_array_like, is_univariate, get_dimension
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision


class DataSet:
    """
    A class for time series anomaly detection data sets. These
    consist of the raw data for training and testing anomaly
    detectors, as well as the respective ground truth labels.

    Parameters
    ----------
    X_test: array-like of shape (n_samples_test, n_attributes)
        The test time series data.
    y_test: array-like of shape (n_samples_test)
        The ground truth anomaly labels of the test data.
    X_train: array-like of shape (n_samples_train, n_attributes), optional
        The train time series. If not given, then the test data will
        be used for training and the data is only compatible with
        unsupervised anomaly detectors.
    y_train: array-like of shape (n_samples_train), optional
        The ground truth anomaly labels of the training data. If not given,
        either the train data should not be given either, or the train
        data is assumed to consist of only normal data.
    """

    X_test: np.ndarray
    y_test: np.ndarray
    X_train: np.ndarray = None
    y_train: np.ndarray = None

    def __init__(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
    ):
        self.check_is_valid(X_test, y_test, X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train

    @staticmethod
    def check_is_valid(
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: Optional[np.ndarray],
        y_train: Optional[np.ndarray],
    ) -> None:
        """
        Checks if the given elements refer o a valid ``DataSet``. If the elements
        would not give a valid ``DataSet``, then a ``ValueError`` is raised.

        Parameters
        ----------
        X_test: array-like of shape (n_samples_test, n_attributes)
            The test time series data.
        y_test: array-like of shape (n_samples_test)
            The ground truth anomaly labels of the test data.
        X_train: array-like of shape (n_samples_train, n_attributes) or ``None``
            The train time series data. Note that, even though ``X_train`` can
            be ``None``, it must be provided.
        y_train: array-like of shape (n_samples_train) or ``None``.
            The ground truth anomaly labels of the train data. Note that, even
            though ``y_train`` can be ``None``, it must be provided.

        Raises
        ------
        ValueError:
            If the given variables would not lead to a valid ``DataSet``. This is the
            case if:

            - If ``X_test`` or ``y_test`` are not valid array-like.
            - If ``y_test`` is not univariate and has a value different from 0 or 1.
            - If ``X_test`` and ``y_test`` consist of a different number of samples.
            - If ``X_train`` is not ``None``, but it is not a valid array-like.
            - If ``X_train`` is not ``None`` and consists of a different number of
              attributes than ``X_test``.
            - If ``y_train`` is not ``None`` but ``X_train`` is ``None``.
            - If ``y_train`` is not ``None`` but it is not a valid array-like.
            - If ``y_train`` is not ``None``, but it is not univariate and has a .
              value different from 0 or 1.
            - If ``y_train`` is not ``None`` but consists of a different number of
              samples than ``X_train``.
        """
        # Check test data
        if not is_valid_array_like(X_test):
            raise ValueError("The test data must be a valid array like!")

        # Check test labels
        if not is_valid_array_like(y_test):
            raise ValueError("The test labels must be a valid array like!")
        if not is_univariate(y_test):
            raise ValueError(
                "There can only be one label for each observation in the test data!"
            )
        if not np.all(np.isin(y_test, [0, 1])):
            raise ValueError("The test labels must be binary!")
        if not y_test.shape[0] == X_test.shape[0]:
            raise ValueError(
                "The test data and labels must consist of the same number of observations!"
            )

        # Check the train data
        if X_train is not None:
            if not is_valid_array_like(X_train):
                raise ValueError("The train data must be a valid array like!")
            if get_dimension(X_test) != get_dimension(X_train):
                raise ValueError(
                    "The test and train data must consist of the same number of features!"
                )

        # Check the train data
        if y_train is not None:
            if X_train is None:
                raise ValueError(
                    "There can not be any train labels if there is no train data!"
                )
            if not is_valid_array_like(y_train):
                raise ValueError("The train labels must be a valid array like!")
            if not is_univariate(y_train):
                raise ValueError(
                    "There can only be one label for each observation in the train data!"
                )
            if not np.all(np.isin(y_train, [0, 1])):
                raise ValueError("The test labels must be binary!")
            if not X_train.shape[0] == y_train.shape[0]:
                raise ValueError(
                    "The train data and labels must consist of the same number of observations!"
                )

    def is_valid(self) -> bool:
        """
        Checks whether this ``DataSet`` is valid or not.

        Returns
        -------
        is_valid: bool
            True if and only if this instance is valid, i.e., if the attributes
            ``X_test``, ``y_test``, ``X_train`` and ``y_train`` of this instance
            pass all the checks of :py:meth:`~dtaianomaly.data.DataSet.check_is_valid`.
        """
        try:
            self.check_is_valid(
                X_test=self.X_test,
                y_test=self.y_test,
                X_train=self.X_train,
                y_train=self.y_train,
            )
            return True

        except ValueError:
            return False

    def compatible_supervision(self) -> List[Supervision]:
        """
        Get the compatible supervision types for this data set.

        Returns
        -------
        compatible_types: list of Supervision
            A list containing the compatible types for this dataset. The following
            suprvision types can be compatible:

            - ``Supervision.UNSUPERVISED``: Always compatible.
            - ``Supervision.SEMI_SUPERVISED``: Compatible if and only if there
              is some training data given (which is assumed to be normal).
            - ``Supervision.SUPERVISED``: Only compatible if both training data
              and training labels are provided.
        """
        # If there is no train data given at all, then only unsupervised detectors are compatible
        if self.X_train is None and self.y_train is None:
            return [Supervision.UNSUPERVISED]
        # If train data is given but no train labels, then either unsupervised or semi-supervised detectors are compatible
        elif self.X_train is not None and self.y_train is None:
            return [Supervision.UNSUPERVISED, Supervision.SEMI_SUPERVISED]
        # If the train data and train labels are given, then all detectors are compatible.
        else:
            return [
                Supervision.UNSUPERVISED,
                Supervision.SEMI_SUPERVISED,
                Supervision.SUPERVISED,
            ]

    def is_compatible(self, detector: BaseDetector) -> bool:
        """
        Checks if the given anomaly detector is compatible with this ``DataSet``.

        Parameters
        ----------
        detector: BaseDetector
            The anomaly detector to check if it is compatible with this ``DataSet``.

        Returns
        -------
        is_compatible: bool
            True if and only if the given anomaly detector is compatible with
            this ``DataSet``. The detector is compatible if

            - This ``DataSet`` does not contain any training data or training labels,
              only unsupervised anomaly detectors are compatible
            - This ``DataSet`` contains training data but no training labels, then
              unsupervised and semi-supervised anomaly detectors are compatible.
            - This ``DataSet`` contains training data and labels, then supervised,
              unsupervised and semi-supervised anomaly detectors are compatible.
        """
        return detector.supervision in self.compatible_supervision()
