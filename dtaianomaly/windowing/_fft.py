import math

import numpy as np

__all__ = ["dominant_fourier_frequency"]


def dominant_fourier_frequency(
    X: np.ndarray, lower_bound: int = 10, upper_bound: int = 1000
) -> int:
    """
    Compute the window size by selecting the dominant Fourier frequency.

    The Fourier transform decomposes a time series into a weighted sum of sine-waves.
    These weights are called the fourier coefficients and are related to a specific
    frequency or period length. The assumption is that the greatest coefficient
    corresponds to the dominant sine-wave which captures the time series characteristics
    best. The window size is therefore computed as the period of this dominant
    sine-wave.

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
    >>> from dtaianomaly.windowing import dominant_fourier_frequency
    >>> X, _ = demonstration_time_series()
    >>> dominant_fourier_frequency(X)
    107
    """
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
