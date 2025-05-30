import abc

import numba as nb
import numpy as np


class FBetaBase(abc.ABC):
    """
    Base class for all F-Beta based metrics. Takes a beta value, checks if it
    is correct, and offers a method to compute the F-score for a given precision
    and recall.
    """

    beta: float

    def __init__(self, beta: (float, int)) -> None:
        if not isinstance(beta, (int, float)) or isinstance(beta, bool):
            raise TypeError("`beta` should be numeric")
        if beta <= 0.0:
            raise ValueError("`beta` should be strictly positive")
        self.beta = beta

    def _f_score(self, precision: float, recall: float) -> float:
        numerator = (1 + self.beta**2) * precision * recall
        denominator = self.beta**2 * precision + recall
        return 0.0 if denominator == 0 else numerator / denominator


@nb.njit(fastmath=True, cache=True)
def np_diff(x: np.array):
    diff = np.empty(shape=(x.shape[0] + 1))
    diff[1:-1] = x[1:] - x[:-1]
    diff[0] = x[0]
    diff[-1] = -x[-1]
    return diff


@nb.njit(fastmath=True, cache=True)
def np_any_axis0(x):
    """Numba compatible version of np.any(x, axis=0)."""
    out = np.zeros(x.shape[1], dtype=nb.bool)
    for i in range(x.shape[0]):
        out = np.logical_or(out, x[i, :])
    return out


@nb.njit(fastmath=True, cache=True)
def np_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=nb.bool)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out


@nb.njit(fastmath=True, cache=True)
def make_intervals(y: np.array) -> (np.array, np.array):
    y = (y > 0).astype(np.int8)
    change_points = np_diff(y)
    starts = np.where(change_points == 1)[0]
    ends = np.where(change_points == -1)[0] - 1
    return starts, ends
