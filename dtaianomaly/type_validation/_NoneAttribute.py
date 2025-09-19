from dtaianomaly.type_validation._BaseAttributeValidation import BaseAttributeValidation

__all__ = ["NoneAttribute"]


class NoneAttribute(BaseAttributeValidation):
    """
    Validate if a given value is of none-type.

    Examples
    --------
    >>> from dtaianomaly.type_validation import NoneAttribute
    >>> a_none = NoneAttribute()
    >>> a_none.raise_error_if_invalid("not-a-none", "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'my_attribute' in class 'MyClass' must be of type None, but received 'not-a-none' of type <class 'str'>!
    """

    def _is_valid_type(self, value) -> bool:
        return value is None

    def _get_valid_type_description(self) -> str:
        return "None"

    def _is_valid_value(self, value) -> bool:
        return value is None

    def _get_valid_value_description(self) -> str:
        return "None"
