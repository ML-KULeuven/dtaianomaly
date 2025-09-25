import numpy as np

from dtaianomaly.utils import is_univariate, is_valid_array_like
from dtaianomaly.windowing._acf import highest_autocorrelation
from dtaianomaly.windowing._fft import dominant_fourier_frequency
from dtaianomaly.windowing._mwf import multi_window_finder
from dtaianomaly.windowing._suss import summary_statistics_subsequences

__all__ = ["compute_window_size"]


def compute_window_size(
    X: np.ndarray,
    window_size: int | str,
    lower_bound: int = 10,
    relative_lower_bound: float = 0.0,
    upper_bound: int = 1000,
    relative_upper_bound: float = 1.0,
    threshold: float = 0.89,
    default_window_size: int = None,
) -> int:
    """
    Compute the window size of the given time series :cite:`ermshaus2023window`.

    Given a time series, automatically compute the window size. This can be done
    either automatically, using a number of procedures for estimating the window
    size, or based on a user-defined value.

    Warnings
    --------
    Automatically computing the windwow size only works for univariate time series!

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

    See Also
    --------
    dominant_fourier_frequency: Compute the window size based on the fft
    highest_autocorrelation: Compute the window size based on the acf
    multi_window_finder: Compute the window size with mwf
    summary_statistics_subsequences: Compute the window size with suss

    Examples
    --------
    >>> from dtaianomaly.data import demonstration_time_series
    >>> from dtaianomaly.windowing import compute_window_size
    >>> X, _ = demonstration_time_series()
    >>> compute_window_size(X, 64)
    64
    >>> compute_window_size(X, 'fft')
    107
    >>> compute_window_size(X, 'acf')
    112
    >>> compute_window_size(X, 'mwf')
    112
    >>> compute_window_size(X, 'suss')
    62
    """
    # Check the input
    if not is_valid_array_like(X):
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
    elif not is_univariate(X):
        raise ValueError(
            "It only makes sense to compute the window size in univariate time series."
        )

    # If the upper and lower bound are invalid, then use the default value (if given)
    elif not (0 <= lower_bound < upper_bound <= X.shape[0]):
        pass

    # Use the fft to compute a window size
    elif window_size == "fft":
        window_size_ = dominant_fourier_frequency(
            X, lower_bound=lower_bound, upper_bound=upper_bound
        )

    # Use the acf to compute a window size
    elif window_size == "acf":
        window_size_ = highest_autocorrelation(
            X, lower_bound=lower_bound, upper_bound=upper_bound
        )

    elif window_size == "mwf":
        window_size_ = multi_window_finder(
            X, lower_bound=lower_bound, upper_bound=upper_bound
        )

    # Use SUSS to compute a window size
    elif window_size == "suss":
        window_size_ = summary_statistics_subsequences(
            X, lower_bound=lower_bound, threshold=threshold
        )

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
