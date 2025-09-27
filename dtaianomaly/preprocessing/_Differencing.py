import numpy as np

from dtaianomaly.preprocessing._Preprocessor import Preprocessor
from dtaianomaly.type_validation import IntegerAttribute

__all__ = ["Differencing"]


class Differencing(Preprocessor):
    """
    Applies differencing to the given time series.

    For a time series :math:`x` and given seasonality :math:`m`, the difference
    :math:`y` is computed as:

    .. math::

       y_t = x_t - x_{t-m}

    This differencing process can be applied a given order of times, recursively.

    Parameters
    ----------
    order : int
        The number of times the differencing procedure should be applied. If the
        order is 0, then no differencing will be applied.
    seasonality : int, default=1
        The seasonality used for computing the difference.

    Examples
    --------
    >>> from dtaianomaly.preprocessing import Differencing
    >>> from dtaianomaly.data import demonstration_time_series
    >>> X, y = demonstration_time_series()
    >>> preprocessor = Differencing(order=1, seasonality=1)
    >>> X_, y_ = preprocessor.fit_transform(X, y)
    """

    order: int
    seasonality: int

    attribute_validation = {
        "order": IntegerAttribute(minimum=1),
        "seasonality": IntegerAttribute(minimum=1),
    }

    def __init__(self, order: int, seasonality: int = 1):
        self.order = order
        self.seasonality = seasonality

    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "Preprocessor":
        return self

    def _transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        X_ = X
        for _ in range(self.order):
            concat = np.concatenate([X_[: self.seasonality], X_])
            X_ = concat[self.seasonality :] - concat[: -self.seasonality]
        return X_, y
