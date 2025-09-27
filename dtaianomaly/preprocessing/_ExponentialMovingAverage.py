import numpy as np

from dtaianomaly.preprocessing._Preprocessor import Preprocessor
from dtaianomaly.type_validation import FloatAttribute

__all__ = ["ExponentialMovingAverage"]


class ExponentialMovingAverage(Preprocessor):
    """
    Compute exponential moving average.

    For a given input :math:`x`, the exponential moving average :math:`y` is computed as

    .. math::

       y_0 &= x_0 \\\\
       y_t &= \\alpha \\cdot x_t + (1 - \\alpha) \\cdot y_{t-1}

    with :math:`0 < \\alpha < 1` the smoothing factor. Higher values of
    :math:`\\alpha` result in more smoothing.

    Parameters
    ----------
    alpha : float
        The decaying factor to be used in the exponential moving average.

    Examples
    --------
    >>> from dtaianomaly.preprocessing import ExponentialMovingAverage
    >>> from dtaianomaly.data import demonstration_time_series
    >>> X, y = demonstration_time_series()
    >>> preprocessor = ExponentialMovingAverage(alpha=0.5)
    >>> X_, y_ = preprocessor.fit_transform(X, y)
    """

    alpha: float

    attribute_validation = {
        "alpha": FloatAttribute(
            minimum=0.0, maximum=1.0, inclusive_minimum=False, inclusive_maximum=False
        )
    }

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "ExponentialMovingAverage":
        return self

    def _transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        X_ = (
            np.frompyfunc(lambda a, b: self.alpha * a + (1 - self.alpha) * b, 2, 1)
            .accumulate(X)
            .astype(dtype=float)
        )
        return X_, y
