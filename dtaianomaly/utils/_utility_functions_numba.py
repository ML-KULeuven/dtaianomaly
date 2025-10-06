import numba as nb
import numpy as np

__all__ = ["np_diff", "np_any_axis0", "np_any_axis1", "make_intervals"]


@nb.njit(fastmath=True, cache=True)
def np_diff(x):
    """
    Numba implementation for np.diff(x).

    Computes the difference of each pair of consecutive values, i.e.,
    the element at position :math:`i` in the output is computed as
    element :math:`i` in ``x`` subtracted from element :math:`i+1`
    in ``x``. Consequently, the output will have one element less
    than the given array.

    Parameters
    ----------
    x : np.array of shape (N,)
        The array on which the difference should be computed.

    Returns
    -------
    np.array of shape (N-1,)
        Identical as np.diff(x)

    Examples
    --------
    >>> import numpy as np
    >>> from dtaianomaly.utils import np_diff
    >>> np_diff(np.array([0, 3, 4, 2, 6, 1]))
    array([ 3,  1, -2,  4, -5])
    """
    return x[1:] - x[:-1]


@nb.njit(fastmath=True, cache=True)
def np_any_axis0(x):
    """
    Numba implementation for np.any(x, axis=0).

    Iterates over the columns of the given matrix and applies
    a logical OR to it. Thus, if any value in column :math:`j`
    is True, then the value at position :math:`j` in the
    output will be True.

    Parameters
    ----------
    x : np.ndarray of shape (N, M)
        The array on which the numpy-call should be applied.

    Returns
    -------
    np.array of shape (M,)
        Identical as np.any(x, axis=0).

    Examples
    --------
    >>> import numpy as np
    >>> from dtaianomaly.utils import np_any_axis0
    >>> np_any_axis0(np.array([[True, False, True, False], [False, False, True, True]]))
    array([ True, False,  True,  True])
    """
    out = np.zeros(x.shape[1], dtype=nb.bool)
    for i in range(x.shape[0]):
        out = np.logical_or(out, x[i, :])
    return out


@nb.njit(fastmath=True, cache=True)
def np_any_axis1(x):
    """
    Numba implementation for np.any(x, axis=1).

    Iterates over the rows of the given matrix and applies
    a logical OR to it. Thus, if any value in row :math:`i`
    is True, then the value at position :math:`i` in the
    output will be True.

    Parameters
    ----------
    x : np.ndarray of shape (N, M)
        The array on which the numpy-call should be applied.

    Returns
    -------
    np.array of shape (N,)
        Identical as np.any(x, axis=1).

    Examples
    --------
    >>> import numpy as np
    >>> from dtaianomaly.utils import np_any_axis1
    >>> np_any_axis1(np.array([[True, True], [True, False], [False, True], [False, False]]))
    array([ True,  True,  True, False])
    """
    out = np.zeros(x.shape[0], dtype=nb.bool)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out


@nb.njit(fastmath=True, cache=True)
def make_intervals(x: np.array) -> (np.array, np.array):
    """
    Numba implementation to convert an array into intervals.

    Given a binary array, i.e., an array consisting of only 0's and 1's, this
    method will extract all intervals of consecutive o1's from the array. The
    start-index corresponds to the index (0-based) of the first 1 in the interval,
    the end-index is the index of the last 1 in the interval. The method outputs
    two independent arrays, one for the start indices and one for the end incides.

    Parameters
    ----------
    x : np.array
        The array from which the intervals must be computed. The array should consist
        of solely 0's and 1's.

    Returns
    -------
    starts: np.array of shape (I,)
        The start index of each interval. In total, there are I intervals.
    ends: np.array of shape (I,)
        The end index of each interval. In total, there are I intervals.

    Examples
    --------
    >>> import numpy as np
    >>> from dtaianomaly.utils import make_intervals
    >>> starts, ends = make_intervals(np.array([0, 1, 1, 1, 0, 0, 1, 0]))
    >>> starts
    array([1, 6])
    >>> ends
    array([3, 6])
    """
    n = x.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    x = (x > 0).astype(np.int8)

    change_points = np.empty(shape=(n + 1))
    change_points[1:-1] = np_diff(x)
    change_points[0] = x[0]
    change_points[-1] = -x[-1]

    starts = np.where(change_points == 1)[0]
    ends = np.where(change_points == -1)[0] - 1
    return starts, ends
