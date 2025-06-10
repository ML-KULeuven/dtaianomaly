import warnings
from typing import Any, Iterable, List, Optional, Tuple

import numba as nb
import numpy as np
from numba.typed import List as nb_List
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.covariance import EmpiricalCovariance

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision


class DWT_MLEAD(BaseDetector):
    """
    Anomaly detection based on the Discrete Wavelet Transform :cite:`thill2017time`.

    DWT-MLEAD (Discrete Wavelet Transform and Maximum Likelihood Estimation
    for Anomaly Detection) first performs multilevel DWT
    using Haar wavelets. Next, for each window in the obtained coefficients,
    a likelihood is estimated using a Guassian distribution. A boundary on
    the likelihood is computed within each DWT-level based on the quantiles,
    and the likelihood estimates that are below the boundary are flagged as
    anomalous. The final anomaly score is then computed as the number of
    times an observation was in an anomalous window.

    Parameters
    ----------
    start_level: int, default=3
        The first level for computing the Discrete Wavelet Transform.
    quantile_boundary_type: {'percentile'}, default='percentile'
        Method for putting a boundary on the likelihood estimates within each DWT-level.
        ``'percentile'`` will consider a ``quantile_epsilon`` of the windows as anomalous.
    quantile_epsilon: float, default=0.01
        The percentile used as threshold on the likelihood estimates.
    padding_mode: {'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'symmetric', 'wrap', 'empty'}, default='wrap'
        Mode for padding the time series, which is passed to `numpy.pad <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`_.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import DWT_MLEAD
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> dwt_mlead = DWT_MLEAD()  # No fitting is necessary
    >>> dwt_mlead.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([ 0.,  0.,  0., ..., 12., 12., 12.]...)

    Notes
    -----
    The implementation is based on `aeon <https://github.com/aeon-toolkit/aeon/blob/main/aeon/anomaly_detection/distribution_based/_dwt_mlead.py>`_
    and `TimeEval <https://github.com/TimeEval/TimeEval-algorithms/blob/main/dwt_mlead/dwt_mlead.py>`_.
    These made the following modifications compared to original paper [thill2017time]_:

    - We use window sizes for the DWT coefficients that decrease with the level number
      because otherwise we would have too few items to slide the window over.
    - We exclude the highest level coefficients because they contain only a single entry
      and are, thus, not suitable for sliding a window of length 2 over it.
    - We have not implemented the Monte Carlo quantile boundary type yet.
    - We do not perform the anomaly clustering step to determine the anomaly centers.
      Instead, we return the anomaly scores for each timestep in the original time
      series.

    In addition, we add the following extension:

    - aeon uses ``'wrap'`` padding and TimeEval uses ``'periodic'`` padding. Initial
      experiments show that different values may lead to quite different anomaly scores.
      Therefore, we included the padding as a parameter of DWT-MLEAD.
    """

    start_level: int
    quantile_boundary_type: str
    quantile_epsilon: float
    padding_mode: str

    def __init__(
        self,
        start_level: int = 3,
        quantile_boundary_type: str = "percentile",
        quantile_epsilon: float = 0.01,
        padding_mode: str = "wrap",
    ):
        super().__init__(Supervision.UNSUPERVISED)

        # Check start level
        if not isinstance(start_level, int) or isinstance(start_level, bool):
            raise TypeError("`start_level` should be numeric")
        if start_level < 0:
            raise ValueError("`start_level` must be >= 0")

        # Check quantile boundary type
        if not isinstance(quantile_boundary_type, str):
            raise TypeError("'quantile_boundary_type' should be a string")
        if quantile_boundary_type not in ["percentile", "monte-carlo"]:
            raise ValueError(
                "`quantile_boundary_type` must be 'percentile' or 'monte-carlo', "
                f"but is {quantile_boundary_type}"
            )
        if quantile_boundary_type in ["monte-carlo"]:
            raise NotImplementedError(
                f"The quantile boundary type '{quantile_boundary_type}' "
                "is not implemented yet!"
            )

        # Check quantile epsilon
        if not isinstance(quantile_epsilon, (float, int)) or isinstance(
            quantile_epsilon, bool
        ):
            raise TypeError("`quantile_epsilon` should be numeric")
        if quantile_epsilon < 0 or quantile_epsilon > 1:
            raise ValueError("quantile_epsilon must be in [0, 1]")

        # Check quantile epsilon
        if not isinstance(padding_mode, str):
            raise TypeError("'padding_mode' should be a string")
        np.pad(np.empty(10), (0, 3), mode=padding_mode)

        # Set the parameters
        self.start_level = start_level
        self.quantile_boundary_type = quantile_boundary_type
        self.quantile_epsilon = quantile_epsilon
        self.padding_mode = padding_mode

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """Should not do anything."""

    def _decision_function(self, X: np.ndarray) -> np.array:

        # Make sure that X is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")
        X = X.squeeze().astype(float)

        # Format the data
        X, n, m = _pad_series(X, self.padding_mode)
        max_level = int(np.log2(m))

        # Check if the start level is not too big
        if self.start_level >= max_level:
            raise ValueError(
                f"start_level ({self.start_level}) must be less than "
                f"log_2(n_timepoints) ({max_level})"
            )

        # perform multilevel DWT and capture coefficients
        levels, approx_coeffs, detail_coeffs = self._multilevel_dwt(X, max_level)

        # extract anomalies in each level
        window_sizes = np.array(
            [
                max(2, max_level - level - self.start_level + 1)
                for level in range(max_level)
            ],
            dtype=np.int_,
        )
        coef_anomaly_counts = []
        for x, level in zip(
            _combine_alternating(detail_coeffs, approx_coeffs), levels.repeat(2, axis=0)
        ):
            w = window_sizes[level]
            windows = sliding_window_view(x, w)

            p = self._estimate_gaussian_likelihoods(windows)
            a = self._mark_anomalous_windows(p)
            xa = self._reverse_windowing(a, window_length=w, full_length=x.shape[0])
            coef_anomaly_counts.append(xa)

        # aggregate anomaly counts (leaf counters)
        point_anomaly_scores = self._push_anomaly_counts_down_to_points(
            coef_anomaly_counts, m, n
        )

        return point_anomaly_scores

    def _multilevel_dwt(
        self, X: np.ndarray, max_level: int
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        ls_ = np.arange(self.start_level - 1, max_level - 1, dtype=np.int_) + 1
        as_, ds_ = _multilevel_haar_transform(X, max_level - 1)
        as_ = as_[self.start_level :]
        ds_ = ds_[self.start_level - 1 :]
        return ls_, as_, ds_

    @staticmethod
    def _estimate_gaussian_likelihoods(x_windows: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=UserWarning)

            # fit gaussion distribution with mean and covariance
            estimator = EmpiricalCovariance(assume_centered=False)
            estimator.fit(x_windows)

            # compute log likelihood for each window x in x_view
            n_windows = x_windows.shape[0]
            p = np.empty(shape=n_windows)
            for i in range(n_windows):
                p[i] = estimator.score(x_windows[i].reshape(1, -1))
        return p

    def _mark_anomalous_windows(self, p: np.ndarray) -> np.ndarray:
        if self.quantile_boundary_type == "percentile":

            # Surpress the warning
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    category=RuntimeWarning,
                    message="invalid value encountered in subtract",
                )
                z_eps = np.percentile(p, self.quantile_epsilon * 100)

        else:
            raise ValueError(
                f"The quantile boundary type '{self.quantile_boundary_type}' "
                "is not implemented yet!"
            )

        return p < z_eps

    @staticmethod
    def _reverse_windowing(
        x: np.ndarray, window_length: int, full_length: int
    ) -> np.ndarray:
        mapped = np.full(shape=(full_length, window_length), fill_value=0)
        mapped[: x.shape[0], 0] = x

        for w in range(1, window_length):
            mapped[:, w] = np.roll(mapped[:, 0], w)

        return np.sum(mapped, axis=1)

    @staticmethod
    def _push_anomaly_counts_down_to_points(
        coef_anomaly_counts: List[np.ndarray], m: int, n: int
    ) -> np.ndarray:
        # sum up counters of detail coeffs (orig. D^l) and approx coeffs (orig. C^l)
        anomaly_counts = coef_anomaly_counts[0::2] + coef_anomaly_counts[1::2]

        # extrapolate anomaly counts to the original series' points
        counter = np.zeros(m)
        for ac in anomaly_counts:
            counter += ac.repeat(m // ac.shape[0], axis=0)
        # set event counters with count < 2 to 0
        counter[counter < 2] = 0
        return counter[:n]


def _pad_series(x: np.ndarray, padding_mode) -> Tuple[np.ndarray, int, int]:
    """Pad input signal to the next power of 2."""
    n = x.shape[0]
    exp = np.ceil(np.log2(n))
    m = int(np.power(2, exp))
    return np.pad(x, (0, m - n), mode=padding_mode), n, m


def _combine_alternating(xs: List[Any], ys: List[Any]) -> Iterable[Any]:
    """Combine two lists by alternating their elements."""
    for x, y in zip(xs, ys):
        yield x
        yield y


def _multilevel_haar_transform(
    x: np.ndarray, levels: int = 1
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Perform the multilevel discrete Haar wavelet transform on a given signal.

    Captures the approximate and detail coefficients per level. The approximate
    coefficients contain one more element than the detail coefficients.

    Parameters
    ----------
    x : np.ndarray
        The input signal.
    levels : int
        The number of levels to perform the Haar wavelet transform.

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        The approximate and detail coefficients per level.
    """
    N = len(x)
    max_levels = np.floor(np.log2(N))
    if levels > max_levels:
        raise ValueError(
            f"The level ({levels}) must be less than log_2(N) ({max_levels})."
        )

    res_approx, res_detail = _haar_transform_iterative(x, levels)
    return res_approx, res_detail


@nb.njit(cache=True, fastmath=True)
def _haar_transform_iterative(
    x: np.ndarray, levels: int
) -> Tuple[nb_List[np.ndarray], nb_List[np.ndarray]]:
    # initialize
    l_approx = nb_List()
    l_approx.append(x)
    l_detail = nb_List()

    for _ in range(1, levels + 1):
        approx = l_approx[-1]
        l_approx.append(_haar_approx_coefficients(approx))
        l_detail.append(_haar_detail_coefficients(approx))

    return l_approx, l_detail


@nb.njit(cache=True, fastmath=True)
def _haar_approx_coefficients(arr: np.ndarray) -> np.ndarray:
    """Get the approximate coefficients at a given level."""
    if len(arr) == 1:
        return np.array([arr[0]])

    N = int(np.floor(len(arr) / 2))
    new = np.empty(N, dtype=arr.dtype)
    for i in range(N):
        new[i] = (arr[2 * i] + arr[2 * i + 1]) / np.sqrt(2)
    return new


@nb.njit(cache=True, fastmath=True)
def _haar_detail_coefficients(arr: np.ndarray) -> np.ndarray:
    """Get the detail coefficients at a given level."""
    if len(arr) == 1:
        return np.array([arr[0]])

    N = int(np.floor(len(arr) / 2))
    new = np.empty(N, dtype=arr.dtype)
    for i in range(N):
        new[i] = (arr[2 * i] - arr[2 * i + 1]) / np.sqrt(2)
    return new
