
import abc
import numpy as np
from typing import Tuple, Optional

from dtaianomaly import utils
from dtaianomaly.PrettyPrintable import PrettyPrintable


def check_preprocessing_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """
    Check if the given `X` and `y` arrays are valid.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_attributes)
        Raw time series
    y: array-like, default=None
        Ground-truth information

    Raises
    ------
    ValueError
        If inputs are not valid numeric arrays
    ValueError
        If inputs have a different size in the first dimension (n_samples)
    """
    if not utils.is_valid_array_like(X):
        raise ValueError('`X` is not a valid array')
    if y is not None and not utils.is_valid_array_like(y):
        raise ValueError('`y` is not  valid array')
    if y is not None:
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError('`X` and `y` have a different number of samples')


class Preprocessor(PrettyPrintable):
    """
    Base preprocessor class.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'Preprocessor':
        """
        First checks the inputs with :py:meth:`~dtaianomaly.preprocessing.Preprocessor.check_preprocessing_inputs`,
        and then fits this preprocessor.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Raw time series
        y: array-like, default=None
            Ground-truth information

        Returns
        -------
        self: Preprocessor
            Returns the fitted instance self.
        """
        check_preprocessing_inputs(X, y)
        return self._fit(np.asarray(X), y if y is None else np.asarray(y))

    @abc.abstractmethod
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'Preprocessor':
        """ Effectively fit this preprocessor, without checking the inputs. """

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        First checks the inputs with :py:meth:`~dtaianomaly.preprocessing.Preprocessor.check_preprocessing_inputs`,
        and then transforms (i.e., preprocesses) the given time series.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Raw time series
        y: array-like of shape (n_samples), default=None
            Ground-truth information

        Returns
        -------
        X_transformed: np.ndarray of shape (n_samples, n_attributes)
            Preprocessed raw time series
        y_transformed: np.ndarray of shape (n_samples)
            The transformed ground truth. If no ground truth was provided (`y=None`),
            then None will be returned as well.
        """
        check_preprocessing_inputs(X, y)
        return self._transform(np.asarray(X), y if y is None else np.asarray(y))

    @abc.abstractmethod
    def _transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """ Effectively transform the given data, without checking the inputs. """

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        First checks the inputs with :py:meth:`~dtaianomaly.preprocessing.Preprocessor.check_preprocessing_inputs`,
        and then chains the fit and transform methods on the given data, i.e.,
        first fit this preprocessor on the given `X` and `y`, after which the
        given `X` and `y` will be transformed.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Raw time series
        y: array-like of shape (n_samples), default=None
            Ground-truth information

        Returns
        -------
        X_transformed: np.ndarray of shape (n_samples, n_attributes)
            Preprocessed raw time series
        y_transformed: np.ndarray of shape (n_samples)
            The transformed ground truth. If no ground truth was provided (`y=None`),
            then None will be returned as well.
        """
        return self.fit(X, y).transform(X, y)


class Identity(Preprocessor):
    """
    Identity preprocessor. A dummy preprocessor which does not do any processing at all.
    """
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'Preprocessor':
        return self

    def _transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return X, y
