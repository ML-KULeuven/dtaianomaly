import numpy as np

__all__ = ["reverse_sliding_window"]


def reverse_sliding_window(
    per_window_anomaly_scores: np.ndarray,
    window_size: int,
    stride: int,
    length_time_series: int,
) -> np.ndarray:
    """
    Reverses the sliding window.

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

    Examples
    --------
    >>> from windowing import reverse_sliding_window
    >>> per_window_anomaly_scores = [0.2, 0.3, 0.5, 0.8, 0.9, 0.6, 0.2, 0.1]
    >>> reverse_sliding_window(per_window_anomaly_scores, 3, 1, 10)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.2     , 0.25    , 0.333..., 0.533..., 0.733...,
           0.766..., 0.566..., 0.3     , 0.15    , 0.1     ])
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
