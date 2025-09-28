import numpy as np

__all__ = ["multi_window_finder"]


def multi_window_finder(
    X: np.ndarray, lower_bound: int = 10, upper_bound: int = 1000
) -> int:
    """
    Compute the window size using the Multi-Window-Finder method :cite:`imani2021multi`.

    A subsequence-based approach which assumes that the variance in the moving averages is
    small given an appropriate window size. This window size then captures the global pattern
    that repeats throught the time series. Multi-Window-Finder will compute the variance of
    the moving average for a number of window candidates, and select the one that minimizes
    the variance.

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Input time series.
    lower_bound : int, default=10
        The lower bound on the automatically computed window size.
    upper_bound : int, default=1000
        The lower bound on the automatically computed window size.

    Returns
    -------
    int
        The computed window size.

    Warnings
    --------
    Automatically computing the windwow size only works for univariate time series!

    Examples
    --------
    >>> from dtaianomaly.data import demonstration_time_series
    >>> from dtaianomaly.windowing import multi_window_finder
    >>> X, _ = demonstration_time_series()
    >>> multi_window_finder(X)
    112
    """

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
