import numpy as np

from dtaianomaly.preprocessing._Preprocessor import Preprocessor
from dtaianomaly.type_validation import IntegerAttribute

__all__ = ["MovingAverage"]


class MovingAverage(Preprocessor):
    """
    Computes the moving average of a time series.

    To compute the moving average at time :math:`t`, i.e., the unweighted
    average of the observations within a window. The window is centered at
    position :math:`t`. For an odd window size, the number of measurements taken
    before and after :math:`t` is equal (namely ``(window_size - 1 ) / 2``. For an
    even window size, there is one additional observation taken before :math:`t`,
    to ensure a correct window size. For multivariate time series, the moving
    average is computed within each attribute independently.

    Parameters
    ----------
    window_size: int
        Length of the window in which the average should be computed.

    Examples
    --------
    >>> from dtaianomaly.preprocessing import MovingAverage
    >>> from dtaianomaly.data import demonstration_time_series
    >>> X, y = demonstration_time_series()
    >>> preprocessor = MovingAverage(window_size=16)
    >>> X_, y_ = preprocessor.fit_transform(X, y)
    """

    window_size: int
    attribute_validation = {
        "window_size": IntegerAttribute(minimum=1),
    }

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size

    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "MovingAverage":
        return self

    def _transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
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
