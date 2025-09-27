import numpy as np
import scipy
from statsmodels.tsa.stattools import acf

__all__ = ["highest_autocorrelation"]


def highest_autocorrelation(
    X: np.ndarray, lower_bound: int = 10, upper_bound: int = 1000
):
    """
    Compute the window size as the leg with the highest autocorrelation.

    The autocorrelation of a time series equals the correlation of that time series
    with a lagged version of itself. It thus shows how similar the observations in the
    time series are to the observations a specific number of lags before. When the
    autocorrelation function is high, the time series is highly similar to the lagged
    version. Consequently, the window size can be computed as the number of lags for
    which the autocorrelation is maximized.

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
    >>> from dtaianomaly.windowing import highest_autocorrelation
    >>> X, _ = demonstration_time_series()
    >>> highest_autocorrelation(X)
    112
    """
    # https://github.com/ermshaua/window-size-selection/blob/main/src/window_size/period.py#L29
    acf_values = acf(X, fft=True, nlags=int(X.shape[0] / 2))

    peaks, _ = scipy.signal.find_peaks(acf_values)
    peaks = peaks[np.logical_and(peaks >= lower_bound, peaks < upper_bound)]
    corrs = acf_values[peaks]

    if peaks.shape[0] == 0:
        return -1

    return int(peaks[np.argmax(corrs)])
