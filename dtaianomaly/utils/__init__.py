"""
This module contains all kinds of utility methods, and can be imported as follows:

>>> from dtaianomaly import utils
"""

from ._CheckIsFittedMixin import CheckIsFittedMixin
from .discovery import all_classes
from .PrettyPrintable import PrettyPrintable
from .utils import get_dimension, is_univariate, is_valid_array_like, is_valid_list

__all__ = [
    "is_valid_list",
    "is_valid_array_like",
    "is_univariate",
    "get_dimension",
    "PrettyPrintable",
    "all_classes",
    "CheckIsFittedMixin",
]
