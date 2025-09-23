from collections.abc import Sequence

import numpy as np

__all__ = ["is_valid_list", "is_valid_array_like", "is_univariate", "get_dimension"]


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
    """
    X = np.array(X)
    if len(X.shape) == 1:
        return 1
    else:
        return X.shape[1]
