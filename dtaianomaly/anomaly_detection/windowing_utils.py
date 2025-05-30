import math
from typing import Union

import numpy as np
import scipy
from statsmodels.tsa.stattools import acf

from dtaianomaly import utils


def sliding_window(X: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Constructs a sliding window for the given time series.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_attributes)
        The time series
    window_size: int
        The window size for the sliding windows.
    stride: int
        The stride, i.e., the step size for the windows.

    Returns
    -------
    windows: np.ndarray of shape ((n_samples - window_size)/stride + 1, n_attributes * window_size)
        The windows as a 2D numpy array. Each row corresponds to a
        window. For windows of multivariate time series are flattened
        to form a 1D array of length the number of attributes multiplied
        by the window size.
    """
    windows = [
        X[t : t + window_size].ravel()
        for t in range(0, X.shape[0] - window_size, stride)
    ]
    windows.append(X[-window_size:].ravel())
    return np.array(windows)


def reverse_sliding_window(
    per_window_anomaly_scores: np.ndarray,
    window_size: int,
    stride: int,
    length_time_series: int,
) -> np.ndarray:
    """
    Reverses the sliding window, to convert the per-window anomaly
    scores into per-observation anomaly scores.

    For non-overlapping sliding windows, it is trivial to convert
    the per-window anomaly scores to per-observation scores, because
    each observation is linked to only one window. For overlapping
    windows, certain observations are linked to one or more windows
    (depending on the window size and stride), obstructing simply
    copying the corresponding per-window anomaly score to each window.
    In the case of multiple overlapping windows, the anomaly score
    of the observation is set to the mean of the corresponding
    per-window anomaly scores.

    Parameters
    ----------
    per_window_anomaly_scores: array-like of shape (n_windows)
    window_size: int
        The window size used for creating windows
    stride: int
        The stride, i.e., the step size used for creating windows
    length_time_series: int
        The original length of the time series.

    Returns
    -------
    anomaly_scores: np.ndarray of shape (length_time_series)
        The per-observation anomaly scores.
    """
    # Convert to array
    scores_time = np.empty(length_time_series)

    start_window_index = 0
    min_start_window = 0
    end_window_index = 0
    min_end_window = 0
    for t in range(length_time_series - window_size):
        while min_start_window + window_size <= t:
            start_window_index += 1
            min_start_window += stride
        while t >= min_end_window:
            end_window_index += 1
            min_end_window += stride
        scores_time[t] = np.mean(
            per_window_anomaly_scores[start_window_index:end_window_index]
        )

    for t in range(length_time_series - window_size, length_time_series):
        while min_start_window + window_size <= t:
            start_window_index += 1
            min_start_window += stride
        scores_time[t] = np.mean(per_window_anomaly_scores[start_window_index:])

    return scores_time


def check_is_valid_window_size(window_size: Union[int, str]) -> None:
    """
    Checks if the given window size is valid or not. If the window size
    is not valid, a ValueError will be raised. Valid window sizes include:

    - a strictly positive integer
    - a string from the set {``'fft'``, ``'acf'``, ``'mwf'``, ``'suss'``}

    Parameters
    ----------
    window_size: int or string
        The valid to check if it is valid or not.

    Raises
    ------
    ValueError
        If the given ``window_size`` is not a valid window size.
    """
    if isinstance(window_size, int):
        if isinstance(window_size, bool):
            raise ValueError("The window size can not be a boolean value!")
        if window_size <= 0:
            raise ValueError("An integer window size should be strictly positive.")

    elif window_size not in ["fft", "acf", "mwf", "suss"]:
        raise ValueError(f"Invalid window_size given: '{window_size}'.")


def compute_window_size(
    X: np.ndarray,
    window_size: Union[int, str],
    lower_bound: int = 10,
    relative_lower_bound: float = 0.0,
    upper_bound: int = 1000,
    relative_upper_bound: float = 1.0,
    threshold: float = 0.89,
    default_window_size: int = None,
) -> int:
    """
    Compute the window size of the given time series :cite:`ermshaus2023window`.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_attributes)
        Input time series.

    window_size: int or str
        The method by which a window size should be computed. Valid options are:

        - ``int``: Simply return the given window size.
        - ``'fft'``: Compute the window size by selecting the dominant Fourier frequency.
        - ``'acf'``: Compute the window size as the leg with the highest autocorrelation.
        - ``'mwf'``: Computes the window size using the Multi-Window-Finder method :cite:`imani2021multi`.
        - ``'suss'``: Computes the window size using the Summary Statistics Subsequence method :cite:`ermshaus2023clasp`.

    lower_bound: int, default=10
        The lower bound on the automatically computed window size. Only used if ``window_size``
        equals ``'fft'``, ``'acf'``, ``'mwf'`` or ``'suss'``.

    relative_lower_bound: float, default=0.0
        The lower bound on the automatically computed window size, relative to the
        length of the given time series. Only used if ``window_size`` equals ``'fft'``,
        ``'acf'``, ``'mwf'`` or ``'suss'``.

    upper_bound: int, default=1000
        The lower bound on the automatically computed window size. Only used if ``window_size``
        equals ``'fft'``, ``'acf'``, or ``'mwf'``.

    relative_upper_bound: float, default=1.0
        The upper bound on the automatically computed window size, relative to the
        length of the given time series. Only used if ``window_size`` equals ``'fft'``,
        ``'acf'``, or ``'mwf'``.

    threshold: float, default=0.89
        The threshold for selecting the optimal window size using ``'suss'``.

    default_window_size: int, default=None
        The default window size, in case an invalid automatic window size was computed.
        By default, the value is set to None, which means that an error is thrown.

    Returns
    -------
    window_size_: int
        The computed window size.
    """
    # Check the input
    check_is_valid_window_size(window_size)
    if not utils.is_valid_array_like(X):
        raise ValueError("X must be a valid, numerical array-like")

    # Initialize the variable
    window_size_ = -1

    # Compute the upper and lower bound
    lower_bound = max(lower_bound, int(relative_lower_bound * X.shape[0]))
    upper_bound = min(upper_bound, int(relative_upper_bound * X.shape[0]))

    # If an int is given, then we can simply return the given window size
    if isinstance(window_size, int):
        return window_size

    # Check if the time series is univariate (error should not be raise if given window size is an integer)
    elif not utils.is_univariate(X):
        raise ValueError(
            "It only makes sense to compute the window size in univariate time series."
        )

    # If the upper and lower bound are invalid, then use the default value (if given)
    elif not (0 <= lower_bound < upper_bound <= X.shape[0]):
        pass

    # Use the fft to compute a window size
    elif window_size == "fft":
        window_size_ = _dominant_fourier_frequency(
            X, lower_bound=lower_bound, upper_bound=upper_bound
        )

    # Use the acf to compute a window size
    elif window_size == "acf":
        window_size_ = _highest_autocorrelation(
            X, lower_bound=lower_bound, upper_bound=upper_bound
        )

    elif window_size == "mwf":
        window_size_ = _mwf(X, lower_bound=lower_bound, upper_bound=upper_bound)

    # Use SUSS to compute a window size
    elif window_size == "suss":
        window_size_ = _suss(X, lower_bound=lower_bound, threshold=threshold)

    # Check if a valid window size was computed, and raise an error if necessary
    if window_size_ == -1:
        if default_window_size is None:
            raise ValueError(
                f"Something went wrong when computing the window size using '{window_size}', "
                f"with lower bound {lower_bound} and upper bound {upper_bound} on a time series "
                f"with shape {X.shape}!"
            )
        else:
            return default_window_size
    else:
        return window_size_


def _dominant_fourier_frequency(
    X: np.ndarray, lower_bound: int, upper_bound: int
) -> int:
    # https://github.com/ermshaua/window-size-selection/blob/main/src/window_size/period.py#L10
    fourier = np.fft.fft(X)
    freq = np.fft.fftfreq(X.shape[0], 1)

    magnitudes = []
    window_sizes = []

    for coef, freq in zip(fourier, freq):
        if coef and freq > 0:
            window_size = int(1 / freq)
            mag = math.sqrt(coef.real * coef.real + coef.imag * coef.imag)

            if lower_bound <= window_size <= upper_bound:
                window_sizes.append(window_size)
                magnitudes.append(mag)

    if len(window_sizes) == 0:
        return -1

    return window_sizes[np.argmax(magnitudes)]


def _highest_autocorrelation(X: np.ndarray, lower_bound: int, upper_bound: int):
    # https://github.com/ermshaua/window-size-selection/blob/main/src/window_size/period.py#L29
    acf_values = acf(X, fft=True, nlags=int(X.shape[0] / 2))

    peaks, _ = scipy.signal.find_peaks(acf_values)
    peaks = peaks[np.logical_and(peaks >= lower_bound, peaks < upper_bound)]
    corrs = acf_values[peaks]

    if peaks.shape[0] == 0:
        return -1

    return peaks[np.argmax(corrs)]


def _mwf(X: np.ndarray, lower_bound: int, upper_bound: int) -> int:
    # https://github.com/ermshaua/window-size-selection/blob/main/src/window_size/mwf.py#L16

    def moving_mean(time_series: np.ndarray, w: int):
        moving_avg = np.cumsum(time_series, dtype=float)
        moving_avg[w:] = moving_avg[w:] - moving_avg[:-w]
        return moving_avg[w - 1 :] / w

    all_averages = []
    window_sizes = list(range(lower_bound, upper_bound))

    for window_size in window_sizes:
        all_averages.append(np.array(moving_mean(X, window_size)))

    moving_average_residuals = []
    for i in range(len(window_sizes)):
        moving_avg = all_averages[i][: len(all_averages[-1])]
        moving_avg_residual = np.log(abs(moving_avg - moving_avg.mean()).sum())
        moving_average_residuals.append(moving_avg_residual)

    b = (np.diff(np.sign(np.diff(moving_average_residuals))) > 0).nonzero()[
        0
    ] + 1  # local min

    if len(b) == 0:
        return -1
    if len(b) < 3:
        return window_sizes[b[0]]

    w = np.mean([window_sizes[b[i]] / (i + 1) for i in range(3)])
    return int(w)


def _suss(X: np.ndarray, lower_bound: int, threshold: float) -> int:
    # https://github.com/ermshaua/window-size-selection/blob/main/src/window_size/suss.py#L25
    # Implementation has been changed to remove pandas dependencies (in `suss_score`)

    def suss_score(time_series: np.ndarray, w: int):

        # Compute the statistics in each window
        windows = np.lib.stride_tricks.sliding_window_view(time_series, w)
        local_stats = np.array(
            [
                windows.mean(axis=1) - global_mean,
                windows.std(axis=1) - global_std,
                (windows.max(axis=1) - windows.min(axis=1)) - global_min_max,
            ]
        )

        # Compute Euclidean distance between local and global stats
        stats_diff = np.sqrt(np.sum(np.square(local_stats), axis=0)) / np.sqrt(w)
        return np.mean(stats_diff)

    if X.max() > X.min():
        X = (X - X.min()) / (X.max() - X.min())

    global_mean = np.mean(X)
    global_std = np.std(X)
    global_min_max = np.max(X) - np.min(X)

    max_suss_score = suss_score(X, 1)
    min_suss_score = suss_score(X, X.shape[0] - 1)
    if min_suss_score == max_suss_score:
        return -1

    # exponential search (to find window size interval)
    exp = 0
    while True:
        window_size = 2**exp

        if window_size < lower_bound:
            exp += 1
            continue

        score = 1 - (suss_score(X, window_size) - min_suss_score) / (
            max_suss_score - min_suss_score
        )

        if score > threshold:
            break

        exp += 1

    lbound, ubound = max(lower_bound, 2 ** (exp - 1)), min(2**exp + 1, X.shape[0] - 1)

    # binary search (to find window size in interval)
    while lbound <= ubound:
        window_size = int((lbound + ubound) / 2)
        score = 1 - (suss_score(X, window_size) - min_suss_score) / (
            max_suss_score - min_suss_score
        )

        if score < threshold:
            lbound = window_size + 1
        elif score > threshold:
            ubound = window_size - 1
        else:
            lbound = window_size
            break

    return 2 * lbound
