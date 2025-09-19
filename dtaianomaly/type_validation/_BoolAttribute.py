from dtaianomaly.type_validation._BaseAttributeValidation import BaseAttributeValidation

__all__ = ["BoolAttribute"]


class BoolAttribute(BaseAttributeValidation):
    """
    Validate if a given value is a boolean.

    Examples
    --------
    >>> from dtaianomaly.type_validation import BoolAttribute
    >>> a_bool = BoolAttribute()
    >>> a_bool.raise_error_if_invalid(True, "my_attribute", "MyClass")   # No error
    >>> a_bool.raise_error_if_invalid(False, "my_attribute", "MyClass")  # No error
    >>> a_bool.raise_error_if_invalid("not-a-bool", "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'my_attribute' in class 'MyClass' must be of type bool, but received 'not-a-bool' of type <class 'str'>!
    """

    def _is_valid_type(self, value) -> bool:
        return isinstance(value, bool)

    def _get_valid_type_description(self) -> str:
        return "bool"

    def _is_valid_value(self, value) -> bool:
        return value is True or value is False

    def _get_valid_value_description(self) -> str:
        return "True or False"
