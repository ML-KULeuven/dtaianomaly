import numpy as np

from dtaianomaly.preprocessing._Preprocessor import Preprocessor
from dtaianomaly.type_validation import FloatAttribute
from dtaianomaly.utils import get_dimension

__all__ = ["RobustScaler"]


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
    lower_quantile: float, default=25.0
        The lower quantile used to compute the scale. Must be in range [0.0, 100.0].
    upper_quantile: float, default=75.0
        The upper quantile used to compute the scale. Must be in range [0.0, 100.0].

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

    Examples
    --------
    >>> from dtaianomaly.preprocessing import RobustScaler
    >>> from dtaianomaly.data import demonstration_time_series
    >>> X, y = demonstration_time_series()
    >>> preprocessor = RobustScaler()
    >>> X_, y_ = preprocessor.fit_transform(X, y)
    """

    lower_quantile: float
    upper_quantile: float
    center_: np.array
    scale_: np.array

    attribute_validation = {
        "lower_quantile": FloatAttribute(minimum=0.0, maximum=100.0),
        "upper_quantile": FloatAttribute(minimum=0.0, maximum=100.0),
    }

    def __init__(self, lower_quantile: float = 25.0, upper_quantile: float = 75.0):
        if lower_quantile > upper_quantile:
            raise ValueError(
                f"Attribute 'lower_quantile' must be smaller than attribute 'upper_quantile' in 'RobustScaler', but received '{lower_quantile}' and '{upper_quantile}' "
            )
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "RobustScaler":
        if get_dimension(X) == 1:
            # univariate case
            self.center_ = np.array([np.nanmedian(X)])
            q_min = np.percentile(X, q=self.lower_quantile)
            q_max = np.percentile(X, q=self.upper_quantile)
            self.scale_ = np.array([q_max - q_min])
        else:
            # multivariate case
            self.center_ = np.nanmedian(X, axis=0)
            q_min = np.percentile(X, q=self.lower_quantile, axis=0)
            q_max = np.percentile(X, q=self.upper_quantile, axis=0)
            self.scale_ = q_max - q_min
        return self

    def _transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        if not (
            (len(X.shape) == 1 and self.center_.shape[0] == 1)
            or X.shape[1] == self.center_.shape[0]
        ):
            raise AttributeError(
                f"Trying to robust scale a time series with {X.shape[0]} attributes while it was fitted on {self.center_.shape[0]} attributes!"
            )

        X_ = (X - self.center_) / self.scale_
        return np.where(np.isnan(X_), X, X_), y
