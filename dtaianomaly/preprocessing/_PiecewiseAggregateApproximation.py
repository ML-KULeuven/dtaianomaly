import numpy as np

from dtaianomaly.preprocessing._Preprocessor import Preprocessor
from dtaianomaly.type_validation import IntegerAttribute

__all__ = ["PiecewiseAggregateApproximation"]


class PiecewiseAggregateApproximation(Preprocessor):
    """
    Performs piecewise aggregate approximation :cite:`keogh2001dimensionality`.

    Piecewise Aggregate Approximation (PAA) is a
    form of dimensionality reduction of time series, originally proposed for
    fast indexing of time series in large databases. Given a value for :math:`n`,
    PAA divides the time series in :math:`n` equi-sized frames. Next, each frame
    is replaced by its mean value. Specifically, for a time series :math:`x` of
    length :math:`N`, position :math:`i` in the transformed time series :math:`y`
    equals:

    .. math::

       y_i = \\frac{n}{N} \\displaystyle\\sum_{j=N/N(i-1)+1}^{(n/N)i} x_j

    For multivariate time series, the dimension of each attribute is reduced
    independently, but the same frames are used.

    Parameters
    ----------
    n: int
        The number of equi-sized frames to generate.

    Examples
    --------
    >>> from dtaianomaly.preprocessing import PiecewiseAggregateApproximation
    >>> from dtaianomaly.data import demonstration_time_series
    >>> X, y = demonstration_time_series()
    >>> preprocessor = PiecewiseAggregateApproximation(n=8)
    >>> X_, y_ = preprocessor.fit_transform(X, y)
    """

    n: int
    attribute_validation = {"n": IntegerAttribute(minimum=1)}

    def __init__(self, n: int):
        self.n = n

    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "Preprocessor":
        return self

    def _transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        if X.shape[0] <= self.n:
            return X, y

        X_ = paa(X, self.n)
        if y is None:
            return X_, y
        else:
            return X_, np.where(paa(y, self.n) < 0.5, 0, 1)


def paa(x: np.ndarray, n: int) -> np.ndarray:
    indices = np.linspace(0, x.shape[0], n + 1, dtype=int, endpoint=True)
    return np.array([np.mean(x[s:e], axis=0) for s, e in zip(indices, indices[1:])])
