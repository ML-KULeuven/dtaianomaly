import numpy as np

from dtaianomaly.preprocessing._Preprocessor import (
    Preprocessor,
    _check_preprocessing_inputs,
)
from dtaianomaly.type_validation import ListAttribute, ObjectAttribute

__all__ = ["ChainedPreprocessor"]


class ChainedPreprocessor(Preprocessor):
    """
    Wrapper chaining multiple :py:class:`~dtaianomaly.preprocessing.Preprocessor` objects.

    Given an ordered list of base :py:class:`~dtaianomaly.preprocessing.Preprocessor`
    objects, the :py:class:`~dtaianomaly.preprocessing.ChainedPreprocessor` will
    apply each preprocessing step in sequence on a given time series.

    Parameters
    ----------
    *base_preprocessors : list of `Preprocessor` objects
        The preprocessors to chain. These preprocessors can be passed as a single
        list argument or as multiple independent arguments to the constructor.

    Examples
    --------
    >>> from dtaianomaly.preprocessing import ChainedPreprocessor, Differencing, StandardScaler
    >>> from dtaianomaly.data import demonstration_time_series
    >>> X, y = demonstration_time_series()
    >>> preprocessor = ChainedPreprocessor(Differencing(order=1), StandardScaler())
    >>> X_, y_ = preprocessor.fit_transform(X, y)
    """

    base_preprocessors: list[Preprocessor]

    attribute_validation = {
        "base_preprocessors": ListAttribute(
            ObjectAttribute(Preprocessor), minimum_length=1
        )
    }

    def __init__(self, *base_preprocessors: Preprocessor | list[Preprocessor]):
        if len(base_preprocessors) == 1 and isinstance(base_preprocessors[0], list):
            self.base_preprocessors = base_preprocessors[0]
        else:
            self.base_preprocessors = list(base_preprocessors)

    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "Preprocessor":
        for preprocessor in self.base_preprocessors:
            preprocessor._fit(X, y)
            X, y = preprocessor._transform(X, y)
        return self

    def _transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        for preprocessor in self.base_preprocessors:
            X, y = preprocessor._transform(X, y)
        return X, y

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        """
        Fit each component of this ChainedPreprocessor and transform the time series.

        First checks if the given input is valid, and then iteratively fit each individual
        preprocessor within the chain and transform the data. The transformed data is uses
        to fit the next preprocessor, and so on.

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
        _check_preprocessing_inputs(X, y)
        for preprocessor in self.base_preprocessors:
            preprocessor._fit(X, y)
            X, y = preprocessor._transform(X, y)
        return X, y

    def piped_str(self) -> str:
        """
        Return this preprocessor as a pipe-representation.

        Return the string representation of each preprocessor in the
        chain, and combine them with a '->' symbol.

        Returns
        -------
        str
            The piped representation of this preprocessor.
        """
        return "->".join(map(str, self.base_preprocessors))
