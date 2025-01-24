import numpy as np
from typing import Optional, Tuple
from sklearn.exceptions import NotFittedError

from dtaianomaly.utils import get_dimension
from dtaianomaly.preprocessing.Preprocessor import Preprocessor


class RobustScaler(Preprocessor):
    """
    Scale the time series using robust statistics.

    The :py:class:`~dtaianomaly.preprocessing.RobustScaler` is similar to
    :py:class:`~dtaianomaly.preprocessing.StandardScaler`, but uses robust
    statistics rather than mean and standard deviation. The center of the data
    is computed via the median, and the scale is computed as the range between
    two quantiles (by default uses the IQR). This ensures that scaling is less
    affected by outliers.

    For a time series :math:`x`, center :math:`c` and scale :math:`s`, observation
    :math:`x_i` is scaled to observation :math:`y_i` using the following equation:

    .. math::

       y_i = \\frac{x_i - c}{s}

    Notice the similarity with the formula for standard scaling. For multivariate
    time series, each attribute is scaled independently, each with an independent
    scale and center.

    Parameters
    ----------
    quantile_range: tuple of (float, float), default = (25.0, 75.0)
        Quantile range used to compute the ``scale_`` of the robust scaler.
        By default, this is equal to the Inter Quantile Range (IQR). The first
        value of the quantile range corresponds to the smallest quantile, the
        second value corresponds to the larger quantile. If the first value is
        not smaller than the second value, an error will be thrown. The values
        must also both be in the range [0, 100].

    Attributes
    ----------
    center_: array-like of shape (n_attributes)
        The median value in each attribute of the training data.
    scale_: array-like of shape (n_attributes)
        The quantile range for each attribute of the training data.

    Raises
    ------
    NotFittedError
        If the `transform` method is called before fitting this StandardScaler.
    """

    quantile_range: (float, float)
    center_: np.array
    scale_: np.array

    def __init__(self, quantile_range: (float, float) = (25.0, 75.0)):
        if not isinstance(quantile_range, tuple):
            raise TypeError("`quantile_range` should be tuple")
        if len(quantile_range) != 2:
            raise ValueError(
                "'quantile_range' should consist of exactly two values (length of 2)"
            )
        if not isinstance(quantile_range[0], (float, int)) or isinstance(
            quantile_range[0], bool
        ):
            raise TypeError(
                "The first element `quantile_range` should be a float or int"
            )
        if not isinstance(quantile_range[1], (float, int)) or isinstance(
            quantile_range[1], bool
        ):
            raise TypeError(
                "The second element `quantile_range` should be a float or int"
            )
        if quantile_range[0] < 0.0:
            raise ValueError(
                "the first element in 'quantile_range' must be at least 0.0"
            )
        if quantile_range[1] > 100.0:
            raise ValueError(
                "the second element in 'quantile_range' must be at most 100.0"
            )
        if not quantile_range[0] < quantile_range[1]:
            raise ValueError(
                "the first element in 'quantile_range' must be at smaller than the second element in 'quantile_range'"
            )
        self.quantile_range = quantile_range

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RobustScaler":
        if get_dimension(X) == 1:
            # univariate case
            self.center_ = np.array([np.nanmedian(X)])
            q_min = np.percentile(X, q=self.quantile_range[0])
            q_max = np.percentile(X, q=self.quantile_range[1])
            self.scale_ = np.array([q_max - q_min])
        else:
            # multivariate case
            self.center_ = np.nanmedian(X, axis=0)
            q_min = np.percentile(X, q=self.quantile_range[0], axis=0)
            q_max = np.percentile(X, q=self.quantile_range[1], axis=0)
            self.scale_ = q_max - q_min
        return self

    def _transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not (hasattr(self, "center_") and hasattr(self, "scale_")):
            raise NotFittedError(f"Call `fit` before using transform on {str(self)}")
        if not (
            (len(X.shape) == 1 and self.center_.shape[0] == 1)
            or X.shape[1] == self.center_.shape[0]
        ):
            raise AttributeError(
                f"Trying to robust scale a time series with {X.shape[0]} attributes while it was fitted on {self.center_.shape[0]} attributes!"
            )

        X_ = (X - self.center_) / self.scale_
        return np.where(np.isnan(X_), X, X_), y
