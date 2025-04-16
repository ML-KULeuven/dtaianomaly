"""
This module contains all kinds of utility methods, and can be imported as follows:

>>> from dtaianomaly import utils
"""

from .PrettyPrintable import PrettyPrintable
from .utils import get_dimension, is_univariate, is_valid_array_like, is_valid_list

__all__ = [
    "is_valid_list",
    "is_valid_array_like",
    "is_univariate",
    "get_dimension",
    "PrettyPrintable",
]
