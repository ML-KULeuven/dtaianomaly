"""
This is a pure-Python module that contains functionality for automatic type validation
on the attributes of a class. It can be imported as follows:

>>> from dtaianomaly import type_validation
"""

from ._AttributeValidationMixin import AttributeValidationMixin
from ._BaseAttributeValidation import BaseAttributeValidation, UnionAttribute
from ._BoolAttribute import BoolAttribute
from ._FloatAttribute import FloatAttribute
from ._IntegerAttribute import IntegerAttribute
from ._ListAttribute import ListAttribute
from ._LiteralAttribute import LiteralAttribute
from ._NoneAttribute import NoneAttribute
from ._ObjectAttribute import ObjectAttribute
from ._PathAttribute import PathAttribute
from ._WindowSizeAttribute import WindowSizeAttribute

__all__ = [
    "BaseAttributeValidation",
    "AttributeValidationMixin",
    "UnionAttribute",
    "NoneAttribute",
    "IntegerAttribute",
    "FloatAttribute",
    "BoolAttribute",
    "LiteralAttribute",
    "ListAttribute",
    "WindowSizeAttribute",
    "ObjectAttribute",
    "PathAttribute",
]
