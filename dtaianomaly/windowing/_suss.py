import numpy as np

__all__ = ["summary_statistics_subsequences"]


def summary_statistics_subsequences(
    X: np.ndarray,
    lower_bound: int = 10,
    upper_bound: int = 1000,
    threshold: float = 0.89,
) -> int:
    """
    Compute the window size using the Summary Statistics Subsequence method :cite:`ermshaus2023clasp`.

    Compare multiple summary statistics (mean, standard deviation, range of values)
    within subsequences to those of the complete time series. The assumption is that
    for a proper subsequence length, the local summary statistics within the subsequences
    will be highly similar to the global statistics across the complete time series.
    Hence, the subsequence length such that the summarized statistics within the subsequences
    is highly similar to the statistics of the time series is returned as computed
    window size.

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Input time series.
    lower_bound : int, default=10
        The lower bound on the automatically computed window size.
    upper_bound : int, default=1000
        The (exclusive) upper bound on the automatically computed window size.
    threshold : float, default=0.89
        The threshold for selecting the optimal window size.

    Returns
    -------
    int
        The computed window size, or -1 if no window size within the bounds
        meets the threshold.

    Warnings
    --------
    Automatically computing the windwow size only works for univariate time series!

    Examples
    --------
    >>> from dtaianomaly.data import demonstration_time_series
    >>> from dtaianomaly.windowing import summary_statistics_subsequences
    >>> X, _ = demonstration_time_series()
    >>> summary_statistics_subsequences(X)
    62
    """
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

    # The window sizes can not exceed the length of the time series
    upper_bound = min(upper_bound, X.shape[0] - 1)
    if lower_bound >= upper_bound:
        return -1

    # exponential search (to find window size interval)
    exp = 0
    while True:
        window_size = min(2**exp, upper_bound - 1)

        if window_size < lower_bound:
            exp += 1
            continue

        score = 1 - (suss_score(X, window_size) - min_suss_score) / (
            max_suss_score - min_suss_score
        )

        if score > threshold:
            break

        # No window size within the bounds meets the threshold
        if window_size == upper_bound - 1:
            return -1

        exp += 1

    lbound, ubound = max(lower_bound, 2 ** (exp - 1)), min(2**exp + 1, upper_bound - 1)

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

    # Check if the window size remains within the bounds
    if 2 * lbound >= upper_bound:
        return -1

    return 2 * lbound
