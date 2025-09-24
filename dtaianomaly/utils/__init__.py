"""
This module contains all kinds of utility methods, and can be imported as follows:

>>> from dtaianomaly import utils
"""

from ._CheckIsFittedMixin import CheckIsFittedMixin
from ._discovery import all_classes
from ._PrintConstructionCallMixin import PrintConstructionCallMixin
from ._utility_functions import (
    get_dimension,
    is_univariate,
    is_valid_array_like,
    is_valid_list,
)
from ._utility_functions_numba import (
    make_intervals,
    np_any_axis0,
    np_any_axis1,
    np_diff,
)

__all__ = [
    "is_valid_list",
    "is_valid_array_like",
    "is_univariate",
    "get_dimension",
    "PrintConstructionCallMixin",
    "all_classes",
    "CheckIsFittedMixin",
    "np_diff",
    "np_any_axis0",
    "np_any_axis1",
    "make_intervals",
]
