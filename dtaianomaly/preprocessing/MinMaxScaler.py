import numpy as np
from typing import Optional, Tuple
from sklearn.exceptions import NotFittedError

from dtaianomaly.preprocessing.Preprocessor import Preprocessor


class MinMaxScaler(Preprocessor):
    """
    Rescale raw time series to a [0, 1] via min-max scaling. The
    minimum and maximum is computed on a training set, after which
    these values can be used to transform a new time series. Therefore,
    there is no guarantee that the values of the transformed test set
    will actually be in the range [0, 1].

    For multivariate time series, each attribute will be normalized
    independently, i.e., the minimum and maximum of each attribute in
    the transformed time series will 0 and 1, respectively.

    If the minimum and maximum of an attribute is the same (the time
    series consists of only one value), then the transformation will
    not do anything.

    Attributes
    ----------
    min_: array-like of shape (n_attributes)
        The minimum value in each attribute of the training data.
    max_: array-like of shape (n_attributes)
        The maximum value in each attribute of the training data.

    Raises
    ------
    NotFittedError
        If the `transform` method is called before fitting this MinMaxScaler.
    """

    min_: np.array
    max_: np.array

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MinMaxScaler":
        if len(X.shape) == 1 or X.shape[1] == 1:
            # univariate case
            self.min_ = np.array([np.nanmin(X)])
            self.max_ = np.array([np.nanmax(X)])
        else:
            # multivariate case
            self.min_ = np.nanmin(X, axis=0)
            self.max_ = np.nanmax(X, axis=0)

        # Adjust to deal with constant attributes
        constant_attributes = self.min_ == self.max_
        self.min_ = np.where(constant_attributes, 0, self.min_)
        self.max_ = np.where(constant_attributes, 1, self.max_)

        return self

    def _transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not (hasattr(self, "min_") and hasattr(self, "max_")):
            raise NotFittedError(f"Call `fit` before using transform on {str(self)}")
        if not (
            (len(X.shape) == 1 and self.min_.shape[0] == 1)
            or X.shape[1] == self.min_.shape[0]
        ):
            raise AttributeError(
                f"Trying to min max scale a time series with {X.shape[0]} attributes while it was fitted on {self.min_.shape[0]} attributes!"
            )

        X_ = (X - self.min_) / (self.max_ - self.min_)
        return X_, y
