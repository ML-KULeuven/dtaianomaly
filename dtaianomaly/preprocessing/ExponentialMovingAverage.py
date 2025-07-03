import numpy as np

from dtaianomaly.preprocessing.Preprocessor import Preprocessor


class ExponentialMovingAverage(Preprocessor):
    """
    Compute exponential moving average. For a given input :math:`x`,
    the exponential moving average :math:`y` is computed as

    .. math::

       y_0 &= x_0 \\\\
       y_t &= \\alpha \\cdot x_t + (1 - \\alpha) \\cdot y_{t-1}

    with :math:`0 < \\alpha < 1` the smoothing factor. Higher values of
    :math:`\\alpha` result in more smoothing.

    Parameters
    ----------
    alpha: float
        The decaying factor to be used in the exponential moving average.
    """

    alpha: float

    def __init__(self, alpha: float) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("Alpha must be in the open interval ]0, 1[")
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
