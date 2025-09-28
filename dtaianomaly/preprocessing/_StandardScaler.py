import numpy as np

from dtaianomaly.preprocessing._Preprocessor import Preprocessor
from dtaianomaly.type_validation import FloatAttribute

__all__ = ["StandardScaler"]


class StandardScaler(Preprocessor):
    """
    Standard scale the data: rescale to zero mean, unit variance.

    Rescale to zero mean and unit variance. A mean value and standard
    deviation is computed on a training set, after which these values
    can be used to transform a new time series. Therefore, there is no
    guarantee that the values of the transformed test set will actually
    have zero mean and unit variance. For multivariate time series, each
    attribute will be normalized independently, i.e., the mean and std of
    each attribute in the transformed time series will 1.0 and 0.0, respectively.

    Parameters
    ----------
    min_std : float, default = 1e-9
        The minimum std required to actually Z-normalize an attribute.
        If the standard deviation is below this value, then no normalization
        will be applied. This prevents amplifying noise in the data.

    Attributes
    ----------
    mean_ : array-like of shape (n_attributes)
        The mean value in each attribute of the training data.
    std_ : array-like of shape (n_attributes)
        The standard deviation in each attribute of the training data.

    Raises
    ------
    NotFittedError
        If the `transform` method is called before fitting this StandardScaler.

    Examples
    --------
    >>> from dtaianomaly.preprocessing import StandardScaler
    >>> from dtaianomaly.data import demonstration_time_series
    >>> X, y = demonstration_time_series()
    >>> preprocessor = StandardScaler()
    >>> X_, y_ = preprocessor.fit_transform(X, y)
    """

    min_std: float
    mean_: np.array
    std_: np.array

    attribute_validation = {
        "min_std": FloatAttribute(minimum=0.0, inclusive_minimum=False),
    }

    def __init__(self, min_std: float = 1e-9):
        self.min_std = min_std

    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "StandardScaler":
        if len(X.shape) == 1 or X.shape[1] == 1:
            # univariate case
            self.mean_ = np.array([np.nanmean(X)])
            self.std_ = np.array([np.nanstd(X)])
        else:
            # multivariate case
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0)

        return self

    def _transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        if not (
            (len(X.shape) == 1 and self.mean_.shape[0] == 1)
            or X.shape[1] == self.mean_.shape[0]
        ):
            raise AttributeError(
                f"Trying to standard scale a time series with {X.shape[0]} attributes while it was fitted on {self.mean_.shape[0]} attributes!"
            )

        # If the std of all attributes is 0, then no transformation happens
        if np.all((self.std_ < self.min_std)):
            X_ = X

        # Else, each attribute is normalized independently, except for the
        # attributes with 0 std.
        else:
            X_ = (X - self.mean_) / self.std_
            for i, (std) in enumerate(self.std_):
                if std < self.min_std:
                    X_[:, i] = X[:, i]

        return X_, y
