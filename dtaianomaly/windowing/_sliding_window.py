import numpy as np

__all__ = ["sliding_window"]


def sliding_window(X: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Construct a sliding window for the given time series.

    Convert the given time series into sliding windows of given size,
    using the given stride.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_attributes)
        The time series.
    window_size : int
        The window size for the sliding windows.
    stride : int
        The stride, i.e., the step size for the windows.

    Returns
    -------
    np.ndarray of shape ((n_samples - window_size)/stride + 1, n_attributes * window_size)
        The windows as a 2D numpy array. Each row corresponds to a
        window. For windows of multivariate time series are flattened
        to form a 1D array of length the number of attributes multiplied
        by the window size.

    Examples
    --------
    >>> import numpy as np
    >>> from dtaianomaly.windowing import sliding_window
    >>> X = np.array([0.2, 0.3, 0.5, 0.8, 0.9, 0.6, 0.2, 0.1])
    >>> sliding_window(X, 2, 1)
    array([[0.2, 0.3],
           [0.3, 0.5],
           [0.5, 0.8],
           [0.8, 0.9],
           [0.9, 0.6],
           [0.6, 0.2],
           [0.2, 0.1]])
    >>> sliding_window(X, 3, 1)
    array([[0.2, 0.3, 0.5],
           [0.3, 0.5, 0.8],
           [0.5, 0.8, 0.9],
           [0.8, 0.9, 0.6],
           [0.9, 0.6, 0.2],
           [0.6, 0.2, 0.1]])
    >>> sliding_window(X, 3, 2)
    array([[0.2, 0.3, 0.5],
           [0.5, 0.8, 0.9],
           [0.9, 0.6, 0.2],
           [0.6, 0.2, 0.1]])
    """
    windows = [
        X[t : t + window_size].ravel()
        for t in range(0, X.shape[0] - window_size, stride)
    ]
    windows.append(X[-window_size:].ravel())
    return np.array(windows)
