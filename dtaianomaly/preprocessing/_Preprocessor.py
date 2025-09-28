import abc

import numpy as np

from dtaianomaly.type_validation import AttributeValidationMixin
from dtaianomaly.utils import (
    CheckIsFittedMixin,
    PrintConstructionCallMixin,
    is_valid_array_like,
)

__all__ = ["Preprocessor"]


def _check_preprocessing_inputs(X: np.ndarray, y: np.ndarray = None) -> None:
    """
    Check the inputs for preprocessing.

    Check if the given `X` and `y` arrays are valid, i.e., if they
    are valid array-likes and have the same length.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_attributes)
        Raw time series
    y : array-like, default=None
        Ground-truth information

    Raises
    ------
    ValueError
        If inputs are not valid numeric arrays
    ValueError
        If inputs have a different size in the first dimension (n_samples)
    """
    if not is_valid_array_like(X):
        raise ValueError("`X` is not a valid array")
    if y is not None and not is_valid_array_like(y):
        raise ValueError("`y` is not  valid array")
    if y is not None:
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("`X` and `y` have a different number of samples")


class Preprocessor(
    PrintConstructionCallMixin, AttributeValidationMixin, CheckIsFittedMixin
):
    """
    Base preprocessor class.

    Class to preprocess data. This is useful for applying transformations on
    the data such that anomalies are more clearly visible or such that the
    data has a standard form (e.g., scaling).
    """

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "Preprocessor":
        """
        Fit this preprocessor.

        First checks the inputs with :py:meth:`~dtaianomaly.preprocessing.Preprocessor.check_preprocessing_inputs`,
        and then fits this preprocessor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_attributes)
            Raw time series.
        y : array-like, default=None
            Ground-truth information.

        Returns
        -------
        Preprocessor
            Returns the fitted instance self.
        """
        _check_preprocessing_inputs(X, y)
        return self._fit(np.asarray(X), y if y is None else np.asarray(y))

    @abc.abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "Preprocessor":
        """Effectively fit this preprocessor, without checking the inputs."""

    def transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        """
        Transform the given time series.

        First checks the inputs with :py:meth:`~dtaianomaly.preprocessing.Preprocessor.check_preprocessing_inputs`,
        and then transforms (i.e., preprocesses) the given time series.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_attributes)
            Raw time series.
        y : array-like of shape (n_samples), default=None
            Ground-truth information.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_attributes)
            Preprocessed raw time series.
        y_transformed : np.ndarray of shape (n_samples)
            The transformed ground truth. If no ground truth was provided (`y=None`),
            then None will be returned as well.
        """
        self.check_is_fitted()
        _check_preprocessing_inputs(X, y)
        return self._transform(np.asarray(X), y if y is None else np.asarray(y))

    @abc.abstractmethod
    def _transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        """Effectively transform the given data, without checking the inputs."""

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        """
        Fit this preprocessor and transform the given time series.

        First checks the inputs with :py:meth:`~dtaianomaly.preprocessing.Preprocessor.check_preprocessing_inputs`,
        and then chains the fit and transform methods on the given data, i.e.,
        first fit this preprocessor on the given `X` and `y`, after which the
        given `X` and `y` will be transformed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_attributes)
            Raw time series.
        y : array-like of shape (n_samples), default=None
            Ground-truth information.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_attributes)
            Preprocessed raw time series.
        y_transformed : np.ndarray of shape (n_samples)
            The transformed ground truth. If no ground truth was provided (`y=None`),
            then None will be returned as well.
        """
        return self.fit(X, y).transform(X, y)

    def piped_str(self) -> str:
        return self.__str__()
