from dtaianomaly.type_validation._BaseAttributeValidation import BaseAttributeValidation

__all__ = ["LiteralAttribute"]


class LiteralAttribute(BaseAttributeValidation):
    """
    Validate if a given value is a literal.

    Check wether a given value is a valid literal. A literal is a value of type
    string but must be one of the predefined values.

    Parameters
    ----------
    *values: string or list of string
        The valid literals.

    Examples
    --------
    >>> from dtaianomaly.type_validation import LiteralAttribute
    >>> a_literal = LiteralAttribute("one", "two", "three")
    >>> a_literal.raise_error_if_invalid("one", "my_attribute", "MyClass")    # No error
    >>> a_literal.raise_error_if_invalid("two", "my_attribute", "MyClass")    # No error
    >>> a_literal.raise_error_if_invalid("three", "my_attribute", "MyClass")  # No error
    >>> a_literal.raise_error_if_invalid(0, "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'my_attribute' in class 'MyClass' must be of type string, but received '0' of type <class 'int'>!
    >>> a_literal.raise_error_if_invalid("four", "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    ValueError: Attribute 'my_attribute' in class 'MyClass' must be in {'one', 'three', 'two'}, but received 'four'!
    """

    _values: set[str]

    def __init__(self, *values: str | list[str]):

        # Handle lists as input
        if len(values) == 1 and isinstance(values[0], list):
            values = values[0]

        # Checks
        for value in values:
            if not isinstance(value, str):
                raise TypeError(
                    f"All elements of attribute 'values' in class 'LiteralAttribute' must be of type str, but received '{value}' of type {type(value)}!"
                )
        if len(values) == 0:
            raise ValueError(
                "Attribute 'values' in class 'LiteralAttribute' must consist of at least one element!"
            )

        self._values = set(values)

    @property
    def values(self) -> set[str]:
        return self._values

    def _is_valid_type(self, value) -> bool:
        return isinstance(value, str)

    def _get_valid_type_description(self) -> str:
        return "string"

    def _is_valid_value(self, value) -> bool:
        return value in self.values

    def _get_valid_value_description(self) -> str:
        values_sorted = ", ".join(f"'{v}'" for v in sorted(self.values))
        return f"in {{{values_sorted}}}"
