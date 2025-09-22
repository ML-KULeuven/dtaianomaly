from dtaianomaly.type_validation._BaseAttributeValidation import (
    BaseAttributeValidation,
    UnionAttribute,
)
from dtaianomaly.type_validation._NoneAttribute import NoneAttribute

__all__ = ["ListAttribute"]


class ListAttribute(BaseAttributeValidation):
    """
    Validate if a given value is a list.

    Check whethr a given value is a valid list. Each elemnt in the list
    is also subjected to a specific type, which is also defined as a
    :py:class:`~dtaianomaly.type_validation.BaseAttributeValidation`.

    Parameters
    ----------
    validator: :py:class:`~dtaianomaly.type_validation.BaseAttributeValidation`
        The validator used to validate the individual elements within the list.

    Examples
    --------
    >>> from dtaianomaly.type_validation import ListAttribute, IntegerAttribute
    >>> list_of_ints = ListAttribute(IntegerAttribute(minimum=1))
    >>> list_of_ints.raise_error_if_invalid([1, 2, 3, 4, 5], "my_attribute", "MyClass")  # No error
    >>> list_of_ints.raise_error_if_invalid("not-a-list", "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'my_attribute' in class 'MyClass' must be of type list of int, but received 'not-a-list' of type <class 'str'>!
    >>> list_of_ints.raise_error_if_invalid(5, "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'my_attribute' in class 'MyClass' must be of type list of int, but received '5' of type <class 'int'>!
    >>> list_of_ints.raise_error_if_invalid([0, 1, 2, 3, 4, 5], "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    ValueError: Attribute 'my_attribute' in class 'MyClass' must be list of int greater than or equal to 1, but received '[0, 1, 2, 3, 4, 5]'!
    """

    _validator: BaseAttributeValidation
    _minimum_length: int

    def __init__(self, validator: BaseAttributeValidation, minimum_length: int = 0):
        if not isinstance(validator, BaseAttributeValidation):
            raise TypeError(
                f"Attribute 'validator' in class 'ListAttribute' must be of type BaseAttributeValidation, but received '{validator}' of type {type(validator)}!"
            )

        if not isinstance(minimum_length, int) or isinstance(minimum_length, bool):
            raise TypeError(
                f"Attribute 'minimum_length' in class 'ListAttribute' must be of type int, but received '{minimum_length}' of type {type(minimum_length)}!"
            )
        if minimum_length < 0:
            raise ValueError(
                "Attribute 'minimum_length' in class 'ListAttribute' must be larger than or equal to 0!"
            )
        self._validator = validator
        self._minimum_length = minimum_length

    @property
    def validator(self) -> BaseAttributeValidation:
        return self._validator

    @property
    def minimum_length(self) -> int:
        return self._minimum_length

    def _is_valid_type(self, value) -> bool:
        return isinstance(value, list) and all(
            self.validator._is_valid_type(element) for element in value
        )

    def _get_valid_type_description(self) -> str:
        return f"list of {self.validator._get_valid_type_description()}"

    def _is_valid_value(self, value) -> bool:
        return len(value) >= self.minimum_length and all(
            self.validator._is_valid_value(element) for element in value
        )

    def _get_valid_value_description(self) -> str:

        def _simple_description(validator: BaseAttributeValidation) -> str:
            if isinstance(validator, NoneAttribute):
                return "None"
            return f"{validator._get_valid_type_description()} {validator._get_valid_value_description()}"

        if isinstance(self.validator, UnionAttribute):
            values = list(map(_simple_description, self.validator.attribute_validators))
            return "list of " + ", ".join(values[:-1]) + " or " + values[-1]

        else:
            return f"list of {_simple_description(self.validator)}"
