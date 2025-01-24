import numpy as np
from typing import Optional, Tuple

from dtaianomaly.preprocessing.Preprocessor import Preprocessor


class MovingAverage(Preprocessor):
    """
    Computes the moving average of a time series. This is the unweighted
    average of the observations within a window.

    To compute the moving average at time :math:`t`, the window is centered at
    position :math:`t`. For an odd window size, the number of measurements taken
    before and after :math:`t` is equal (namely ``(window_size - 1 ) / 2``. For an
    even window size, there is one additional observation taken before :math:`t`,
    to ensure a correct window size.

    For multivariate time series, the moving average is computed within
    each attribute independently.

    Parameters
    ----------
    window_size: int
        Length of the window in which the average should be computed.
    """

    window_size: int

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("Window size must be strictly positive")
        self.window_size = window_size

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MovingAverage":
        return self

    def _transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Add nan values at the beginning and end of the given array
        extend_front = np.full(self.window_size // 2, np.nan)
        extend_back = np.full(
            self.window_size // 2 - (self.window_size % 2 == 0), np.nan
        )
        if len(X.shape) == 2:
            extend_front = np.repeat(extend_front, X.shape[1]).reshape(-1, X.shape[1])
            extend_back = np.repeat(extend_back, X.shape[1]).reshape(-1, X.shape[1])
        X_extended = np.concatenate([extend_front, X, extend_back], axis=0)
        # Compute the average within each window
        X_ = np.array(
            [
                np.nanmean(window, axis=-1)
                for window in np.lib.stride_tricks.sliding_window_view(
                    X_extended, self.window_size, axis=0
                )
            ]
        )
        # Return the results
        return X_, y
