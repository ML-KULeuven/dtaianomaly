from typing import List, Optional

import numpy as np

from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.utils import (
    get_dimension,
    is_univariate,
    is_valid_array_like,
    is_valid_list,
)


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
    X_train: array-like of shape (n_samples_train, n_attributes), default=None
        The train time series. If not given, then the test data will
        be used for training and the data is only compatible with
        unsupervised anomaly detectors.
    y_train: array-like of shape (n_samples_train), default=None
        The ground truth anomaly labels of the training data. If not given,
        either the train data should not be given either, or the train
        data is assumed to consist of only normal data.
    feature_names: list of str, default=None
        The name of each feature in the data. The number of names must be
        identical to the number of actual features. If None, then the data
        is assumed to be unnamed.
    time_steps_test: array-like of shape (n_samples_test), default=None
        The time steps corresponding to the test data. If ``None``, then no
        time steps are known.
    time_steps_train: array-like of shape (n_samples_train), default=None
        The time steps corresponding to the train data. If ``None``, then no
        time steps are known. Can only be provided if there is actually some
        training data given (``X_train` != None``).
    """

    X_test: np.ndarray
    y_test: np.array
    X_train: Optional[np.ndarray]
    y_train: Optional[np.array]
    feature_names: Optional[List[str]]
    time_steps_test: Optional[np.array]
    time_steps_train: Optional[np.array]

    def __init__(
        self,
        X_test: np.ndarray,
        y_test: np.array,
        X_train: np.ndarray = None,
        y_train: np.array = None,
        feature_names: List[str] = None,
        time_steps_test: np.array = None,
        time_steps_train: np.array = None,
    ):
        # Check actual data
        self.check_is_valid(X_test, y_test, X_train, y_train)

        # Check feature names
        if feature_names is not None:
            if not is_valid_list(feature_names, str):
                raise ValueError("The given feature_names are not a valid list!")
            if len(feature_names) != get_dimension(X_test):
                raise ValueError(
                    "The number of features do not correspond to the given number of feature names!"
                )

        # Check time steps for the test data
        if time_steps_test is not None:
            if not is_valid_array_like(time_steps_test):
                raise ValueError("The given time_steps_test is not a valid array-like!")
            if time_steps_test.shape[0] != X_test.shape[0]:
                raise ValueError(
                    "The number of test time steps do not correspond to the actual number of observations in the test data!"
                )

        # Check time steps for the train data
        if time_steps_train is not None:
            if X_train is None:
                raise ValueError(
                    "There have been time steps given for the training data, but no training data!"
                )
            if not is_valid_array_like(time_steps_train):
                raise ValueError(
                    "The given time_steps_train is not a valid array-like!"
                )
            if time_steps_train.shape[0] != X_train.shape[0]:
                raise ValueError(
                    "The number of train time steps do not correspond to the actual number of observations in the train data!"
                )

        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.time_steps_test = time_steps_test
        self.time_steps_train = time_steps_train

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
