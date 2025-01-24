from typing import Optional, Tuple

import numpy as np

from dtaianomaly.preprocessing.Preprocessor import Preprocessor


class Differencing(Preprocessor):
    """
    Applies differencing to the given time series. For a time series :math:`x`
    and given season :math:`m`, the difference :math:`y` is computed as:

    .. math::

       y_t = x_t - x_{t-m}

    This differencing process can be applied a given order of times, recursively.

    Parameters
    ----------
    order: int
        The number of times the differencing procuder should be applied. If the
        order is 0, then no differencing will be applied.
    window_size: int, default=1
        The decaying factor to be used in the exponential moving average.
    """

    order: int
    window_size: int

    def __init__(self, order: int, window_size: int = 1):
        super().__init__()

        if not isinstance(order, int) or isinstance(order, bool):
            raise TypeError("`order` should be an integer")
        if order < 0:
            raise ValueError("'order' must be positive!")

        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise TypeError("`window_size` should be an integer")
        if window_size < 1:
            raise ValueError("'window_size' must be strictly positive!")

        self.order = order
        self.window_size = window_size

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Preprocessor":
        return self

    def _transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_ = X
        for _ in range(self.order):
            concat = np.concatenate([X_[: self.window_size], X_])
            X_ = concat[self.window_size :] - concat[: -self.window_size]
        return X_, y
