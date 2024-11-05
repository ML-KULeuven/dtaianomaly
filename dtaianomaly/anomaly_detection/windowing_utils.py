
import math
import numpy as np
from typing import Union
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
    windows = [X[t:t+window_size].ravel() for t in range(0, X.shape[0] - window_size, stride)]
    windows.append(X[-window_size:].ravel())
    return np.array(windows)


def reverse_sliding_window(per_window_anomaly_scores: np.ndarray, window_size: int, stride: int, length_time_series: int) -> np.ndarray:
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
        scores_time[t] = np.mean(per_window_anomaly_scores[start_window_index:end_window_index])

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
    - a string from the set {``'fft'``}

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
            raise ValueError('The window size can not be a boolean value!')
        if window_size <= 0:
            raise ValueError('An integer window size should be strictly positive.')

    elif window_size not in ['fft']:
        raise ValueError(f"Invalid window_size given: '{window_size}'.")


def compute_window_size(
        X: np.ndarray,
        window_size: Union[int, str],
        lower_bound: int = 10,
        upper_bound: int = 1000) -> int:
    """
    Compute the window size of the given time series [ermshaus2023window]_.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_attributes)
        Input time series.

    window_size: int or str
        The method by which a window size should be computed. Valid options are:

        - ``int``: Simply return the given window size.
        - ``'fft'``: Compute the window size by selecting the dominant Fourier frequency.

    lower_bound: int, default=10
        The lower bound on the automatically computed window size. Only used if ``window_size``
        equals ``'fft'``.

    upper_bound: int, default=1000
        The lower bound on the automatically computed window size. Only used if ``window_size``
        equals ``'fft'``

    Returns
    -------
    window_size_: int
        The computed window size.

    References
    ----------
    .. [ermshaus2023window] Ermshaus, Arik, Patrick Schäfer, and Ulf Leser. "Window
       size selection in unsupervised time series analytics: A review and benchmark."
       International Workshop on Advanced Analytics and Learning on Temporal Data.
       Springer, Cham, 2023, doi: `10.1007/978-3-031-24378-3_6 <https://doi.org/10.1007/978-3-031-24378-3_6>`_
    """
    # Check the input
    check_is_valid_window_size(window_size)
    if not utils.is_valid_array_like(X):
        raise ValueError("X must be a valid, numerical array-like")

    # If an int is given, then we can simply return the given window size
    if isinstance(window_size, int):
        return window_size

    # Check if the time series is univariate (error should not be raise if given window size is an integer)
    if not utils.is_univariate(X):
        raise ValueError('It only makes sens to compute the window size in univariate time series.')

    # Use the fft to compute a window size
    elif window_size == 'fft':
        return _dominant_fourier_frequency(X, lower_bound=lower_bound, upper_bound=upper_bound)

    else:
        raise ValueError(f"Invalid window_size given: '{window_size}'.")


def _dominant_fourier_frequency(time_series: np.ndarray, lower_bound: int = 10, upper_bound: int = 1000) -> int:
    # https://github.com/ermshaua/window-size-selection/blob/main/src/window_size/period.py#L10
    fourier = np.fft.fft(time_series)
    freq = np.fft.fftfreq(time_series.shape[0], 1)

    magnitudes = []
    window_sizes = []

    for coef, freq in zip(fourier, freq):
        if coef and freq > 0:
            window_size = int(1 / freq)
            mag = math.sqrt(coef.real * coef.real + coef.imag * coef.imag)

            if lower_bound <= window_size <= upper_bound:
                window_sizes.append(window_size)
                magnitudes.append(mag)

    return window_sizes[np.argmax(magnitudes)]
