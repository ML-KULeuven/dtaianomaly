# Based on: https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4
import numba as nb
import numpy as np

from dtaianomaly import utils
from dtaianomaly.anomaly_detection import BaseDetector, Supervision


class LocalPolynomialApproximation(BaseDetector):
    """
    Perform anomaly detection based on local polynomial approximations :cite:`li2007unifying`.

    The local polynomial approximation learns a forward estimate :math:`\\hat{x}_t^{(f)} = f(x_{t-1})`
    and a backward estimate :math:`\\hat{x}_t^{(b)} = b(x_{t+1})` for each observation in the time series.
    Both :math:`f` and :math:`b` are polynomials of a specified degree, and are estimated using a neighborhood
    before and after :math:`x_t`, respectively. The error of both estimators is computed, and the maximum
    error is taken as anomaly score, under the assumption anomalies have both a large forward and
    backward error.

    Parameters
    ----------
    neighborhood: int
        The size of the neighborhood for estimating the polynomials.
        The window size, the length of the subsequences that will be detected as anomalies. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    power: int, default=1
        The power of the polynomial to fit. While any strictly positive value is valid, :cite:`li2007unifying`
        indicate that the local polynomial fit with odd order is better than that with an even order.
    normalize_variance: bool, default=false
        Wether to normalize the forward and backward estimates based on the variance of the forward
        and backward neighborhood, respectively.
    buffer: int, default=16
        A buffer at the start and end of the time series, used to ensure that sufficient data is available
        for fitting each polynomial. The buffer must be at least 3 to ensure that the .

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import LocalPolynomialApproximation
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> poly = LocalPolynomialApproximation(neighborhood=50).fit(x)
    >>> poly.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0., 0., 0., ..., 0., 0., 0.]...)

    Notes
    -----
    - LocalPolynomialApproximation only handles univariate time series.
    - The original version of :cite:t:`li2007unifying` normalizes the forward and backward scores. Their
      approach requires two additional parameters. Therefore, we did not implement this and leave normalizatin
      of the (aggregate) scores to post-processing.
    """

    neighborhood: int | str
    power: int
    normalize_variance: bool
    buffer: int

    def __init__(
        self,
        neighborhood: int,
        power: int = 1,
        normalize_variance: bool = False,
        buffer: int = 16,
    ):
        super().__init__(Supervision.UNSUPERVISED)

        if not isinstance(neighborhood, int) or isinstance(neighborhood, bool):
            raise TypeError("`neighborhood` should be integer")
        if neighborhood < 2:
            raise ValueError("`neighborhood` should be at least 2")

        if not isinstance(power, int) or isinstance(power, bool):
            raise TypeError("`power` should be integer")
        if power < 1:
            raise ValueError("`power` should be strictly positive")

        if not isinstance(normalize_variance, bool):
            raise TypeError("'normalize_variance' should be a boolean")

        if not isinstance(buffer, int) or isinstance(buffer, bool):
            raise TypeError("`buffer` should be integer")
        if buffer < 3:
            raise ValueError("`buffer` must be at least 3")

        self.neighborhood = neighborhood
        self.power = power
        self.normalize_variance = normalize_variance
        self.buffer = buffer

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        if not utils.is_univariate(X):
            raise ValueError(
                "'LocalPolynomialApproximation' must receive univariate data!"
            )

    def _decision_function(self, X: np.ndarray) -> np.array:
        if not utils.is_univariate(X):
            raise ValueError(
                "'LocalPolynomialApproximation' must receive univariate data!"
            )

        return _local_poly_nb_parallel(
            X=X.squeeze(),
            neighborhood=self.neighborhood,
            power=self.power,
            buffer=self.buffer,
            normalize_variance=self.normalize_variance,
        )


@nb.njit(fastmath=True)
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0], deg + 1))
    const = np.ones_like(x)
    mat_[:, 0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_


@nb.njit(fastmath=True)
def _fit_x(a, b):
    return np.linalg.lstsq(a, b)[0]


@nb.njit(fastmath=True)
def fit_poly(x, y, deg):
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]


@nb.njit(fastmath=True)
def eval_polynomial(P, x):
    """
    Compute polynomial P(x) where P is a vector of coefficients, highest
    order coefficient at P[0]. Uses Horner's Method.
    """
    result = 0
    for coeff in P:
        result = x * result + coeff
    return result


@nb.njit(parallel=True)
def _local_poly_nb_parallel(
    X, neighborhood=64, power=1, buffer=16, normalize_variance=False
):
    decision_scores = np.zeros(shape=(X.shape[0]))

    for t in nb.prange(buffer + 1, X.shape[0] - buffer):

        # Fit the forward estimator
        forward_neighborhood = X[max(0, t - neighborhood) : t]
        forward_polynomial = fit_poly(
            forward_neighborhood[:-1], forward_neighborhood[1:], power
        )

        # Fit the backward estimator
        backward_neighborhood = X[t + 1 : t + neighborhood + 1]
        backward_polynomial = fit_poly(
            backward_neighborhood[1:], backward_neighborhood[:-1], power
        )

        # Compute the scores
        forward_score = (X[t] - eval_polynomial(forward_polynomial, X[t - 1])) ** 2
        backward_score = (X[t] - eval_polynomial(backward_polynomial, X[t + 1])) ** 2

        # Normalize the variance, if requested
        if normalize_variance:
            forward_score /= np.var(forward_neighborhood) ** 2
            backward_score /= np.var(backward_neighborhood) ** 2

        decision_scores[t] = max(forward_score, backward_score)

    return decision_scores
