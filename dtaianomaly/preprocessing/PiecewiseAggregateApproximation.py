
import numpy as np
from typing import Optional, Tuple
from dtaianomaly.preprocessing.Preprocessor import Preprocessor


class PiecewiseAggregateApproximation(Preprocessor):
    """
    Performs piecewise aggregate approximation.

    Piecewise Aggregate Approximation (PAA) [keogh2001dimensionality]_ is a
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

    References
    ----------
    .. [keogh2001dimensionality] Keogh, E., Chakrabarti, K., Pazzani, M. et al.
       Dimensionality Reduction for Fast Similarity Search in Large Time Series
       Databases. Knowledge and Information Systems 3, 263–286 (2001).
       doi: `10.1007/PL00011669 <https://doi.org/10.1007/PL00011669>`_.
    """
    n: int

    def __init__(self, n: int):
        super().__init__()

        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("`n` should be an integer")
        if n <= 0:
            raise ValueError("'n' must be strictly positive!")

        self.n = n

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'Preprocessor':
        return self

    def _transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if X.shape[0] <= self.n:
            return X, y

        X_ = paa(X, self.n)
        if y is None:
            return X_, y
        else:
            return X_, np.where(paa(y, self.n) < 0.5, 0, 1)


def paa(x: np.ndarray, n: int) -> np.ndarray:
    indices = np.linspace(0, x.shape[0], n + 1, dtype=int, endpoint=True)
    print(x, indices)
    return np.array([np.mean(x[s:e], axis=0) for s, e in zip(indices, indices[1:])])
