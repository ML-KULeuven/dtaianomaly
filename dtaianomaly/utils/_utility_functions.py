from collections.abc import Sequence
from typing import TypeVar

import numpy as np

__all__ = [
    "is_valid_list",
    "is_valid_array_like",
    "is_univariate",
    "get_dimension",
    "convert_to_list",
]

T = TypeVar("T")


def is_valid_list(value, target_type) -> bool:
    """
    Check if the given list is a valid, with each instance being a member
    of the given type.

    Parameters
    ----------
    value: object
        The value to check if it is a valid list
    target_type: Type
        The type of each object in the given list

    Returns
    -------
    is_valid: bool
        True if and only if the given ``value`` is a list and all elements in
        the list are of type ``Type``, otherwise False.

    Examples
    --------
    >>> from dtaianomaly.utils import is_valid_list
    >>> is_valid_list([1, 2, 3, 4, 5], int)
    True
    >>> is_valid_list([1, 2, 3, 4, 5], str)
    False
    >>> is_valid_list(["1", "2", "3", "4", "5"], int)
    False
    >>> is_valid_list(["1", "2", "3", "4", "5"], str)
    True
    """
    return (isinstance(value, list) or isinstance(value, tuple)) and all(
        isinstance(item, target_type) for item in value
    )


def is_valid_array_like(array) -> bool:
    """
    Check if input is "array-like". Within ``dtaianomaly``, this is
    either a numpy array of numerical values or a python sequence of
    numerical values.

    Parameters
    ----------
    array: object
        The array to check if it is a valid array-like

    Returns
    -------
    is_valid: bool
        True if and only if the given array is either a numpy array
        or a python sequence, in which the type entirely consists of
        numerical values, otherwise False.

    Examples
    --------
    >>> from dtaianomaly.utils import is_valid_array_like
    >>> is_valid_array_like([1, 2, 3, 4, 5])
    True
    >>> is_valid_array_like([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    True
    >>> is_valid_array_like([1, 2, 3.0, 4, 5])
    True
    >>> is_valid_array_like([1, 2, "3", 4, 5])
    False
    >>> is_valid_array_like(["1", "2", "3", "4", "5"])
    False
    """
    # Check for valid numpy array
    if isinstance(array, np.ndarray):
        if array.size == 0:
            return True
        return (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.floating)
            or np.issubdtype(array.dtype, bool)
            or np.issubdtype(array.dtype, np.datetime64)
        )

    # Check for numerical sequence
    if isinstance(array, Sequence) and not isinstance(array, str):
        if len(array) == 0:
            return True

        if isinstance(array[0], Sequence) and not isinstance(array[0], str):
            # Multivariate case
            n_attributes = len(array[0])
            return all(
                isinstance(sample, Sequence)
                and not isinstance(sample, str)
                and len(sample) == n_attributes
                and all(
                    isinstance(item, (int, float, np.datetime64)) for item in sample
                )
                for sample in array
            )
        else:
            # Univariate case
            return all(isinstance(item, (int, float, np.datetime64)) for item in array)

    # Default case
    return False


def is_univariate(X: np.ndarray) -> bool:
    """
    Checks if the given array equals a univariate time series.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_attributes)
        The time series data to check if it is univariate.

    Returns
    -------
    is_univariate: bool
        True if and only if the given time series has only one dimension,
        or if the second dimension of the time series is of size 1.

    Examples
    --------
    >>> from dtaianomaly.utils import is_univariate
    >>> is_univariate([1, 2, 3, 4, 5])
    True
    >>> is_univariate([[1], [2], [3], [4], [5]])
    True
    >>> is_univariate([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    False
    """
    return get_dimension(X) == 1


def get_dimension(X: np.ndarray) -> int:
    """
    Get the dimension of the given array.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_attributes)
        The time series data to get the dimension from

    Returns
    -------
    n_attributes: int
        The number of attributes in the given time series.

    Examples
    --------
    >>> from dtaianomaly.utils import get_dimension
    >>> get_dimension([1, 2, 3, 4, 5])
    1
    >>> get_dimension([[1], [2], [3], [4], [5]])
    1
    >>> get_dimension([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    2
    >>> get_dimension([[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400], [5, 50, 500]])
    3
    """
    X = np.array(X)
    if len(X.shape) == 1:
        return 1
    else:
        return X.shape[1]


def convert_to_list(value: T | list[T]) -> list[T]:
    """
    Converts a given value to a list.

    Converts the given value in a list. If the value already is a list, then
    it will simply be returned. Otherwise, a list is returned with a single
    element as the given element.

    Parameters
    ----------
    value: list or other
        The value to convert to a list.

    Returns
    -------
    as_a_list: list
        The given value in a list.

    Examples
    --------
    >>> from dtaianomaly.utils import convert_to_list
    >>> convert_to_list(1)
    [1]
    >>> convert_to_list(3.14)
    [3.14]
    >>> convert_to_list([1, 3.14])
    [1, 3.14]
    """
    if not isinstance(value, list):
        return [
            value,
        ]
    return value
