import numpy as np

from dtaianomaly import utils
from dtaianomaly.anomaly_detection._BaseDetector import BaseDetector, Supervision
from dtaianomaly.type_validation import IntegerAttribute, NoneAttribute

__all__ = ["MedianMethod"]


class MedianMethod(BaseDetector):
    """
    Anomaly detection based on the Two-sided Median Method :cite:`basu2007automatic`.

    The Median Method computes the deviation of a sample
    compared to its neighborhood. This neighborhood is computed as a window
    around the sample. The deviation is consequently measured as the number
    of standard deviations the observations deviates from the mean of its
    neighborhood.

    In contrast to the original paper, this implementation allows to define a
    different neighborhood size before and after the sample, to fine tune how
    much lookahead is allowed. In the ultimate case, if ``neighborhood_size_after = 0``,
    then the Median Method is a purely online anomaly detector. Note, however,
    that this case differs from the One-Sided Median Method discussed in the
    original paper, which also uses the first order difference to detect anomalies.

    Parameters
    ----------
    neighborhood_size_before : int
        The number of observations before the sample to include in the neighborhood.
    neighborhood_size_after : int, default=None
        The number of observations after the sample to include in the neighborhood.
        If None, the same value as ``window_size_before`` will be used.

    Notes
    -----
    MedianMethod only handles univariate time series.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import MedianMethod
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> median_method = MedianMethod(10)  # No fitting is necessary
    >>> median_method.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([1.1851476 , 0.68191703, 1.05125284, ..., 0.81373386, 1.86097851, 0.05852008]...)
    """

    neighborhood_size_before: int
    neighborhood_size_after: int | None

    attribute_validation = {
        "neighborhood_size_before": IntegerAttribute(1),
        "neighborhood_size_after": IntegerAttribute(0) | NoneAttribute(),
    }

    def __init__(
        self,
        neighborhood_size_before: int,
        neighborhood_size_after: int = None,
    ):
        super().__init__(Supervision.UNSUPERVISED)

        if not isinstance(neighborhood_size_before, int) or isinstance(
            neighborhood_size_before, bool
        ):
            raise TypeError("`neighborhood_size_before` should be an integer")
        if neighborhood_size_before < 1:
            raise ValueError("`neighborhood_size_before` should be strictly positive")

        if neighborhood_size_after is not None:
            if not isinstance(neighborhood_size_after, int) or isinstance(
                neighborhood_size_after, bool
            ):
                raise TypeError("`neighborhood_size_after` should be an integer")
            if neighborhood_size_after < 0:
                raise ValueError("`neighborhood_size_after` can not be negative!")

        self.neighborhood_size_before = neighborhood_size_before
        self.neighborhood_size_after = neighborhood_size_after

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        """Should not do anything."""

    def _decision_function(self, X: np.ndarray) -> np.array:
        # Make sure that X is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")
        X = X.squeeze().astype(float)

        # Set the neighborhood size after the observation
        if self.neighborhood_size_after is None:
            neighborhood_size_after = self.neighborhood_size_before
        else:
            neighborhood_size_after = self.neighborhood_size_after

        X_padded = np.pad(
            X,
            (self.neighborhood_size_before, neighborhood_size_after),
            constant_values=(np.nan,),
        )
        neighborhoods = np.lib.stride_tricks.sliding_window_view(
            X_padded,
            window_shape=(self.neighborhood_size_before + neighborhood_size_after + 1),
        )
        return np.nan_to_num(
            np.abs(X - np.nanmean(neighborhoods, axis=1))
            / np.nanstd(neighborhoods, axis=1),
            nan=0.0,
        )
