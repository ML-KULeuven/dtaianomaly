
import numpy as np
from typing import Optional
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector


class DataSet:
    """
    A class for time series anomaly detection data sets. These
    consist of the raw data for training and testing anomaly
    detectors, as well as the respective ground truth labels.

    Parameters
    ----------
    X_test: array-like of shape (n_samples_test, n_attributes)
        The test time series data
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

    @property
    def x(self) -> np.ndarray:
        """ Temporary method to be compatible with existing code. """
        return self.X_test

    @property
    def y(self) -> np.ndarray:
        """ Temporary method to be compatible with existing code. """
        return self.y_test

    def is_valid(self) -> bool:
        # Check if this DataSet object is valid (correct dimensions, whether train data is given, ...)
        return True

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

        # If there is no train data given at all, then only unsupervised detectors are compatible
        if self.X_train is None and self.y_train is None:
            return True

        # If train data is given but no train labels, then either unsupervised or semi-supervised detectors are compatible
        elif self.X_train is not None and self.y_train is None:
            return True

        # If the train data and train labels are given, then all detectors are compatible.
        else:
            return True
