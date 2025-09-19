from dtaianomaly.type_validation._BaseAttributeValidation import BaseAttributeValidation

__all__ = ["IntegerAttribute"]


class IntegerAttribute(BaseAttributeValidation):
    """
    Validate if a given value is an int.

    Check wether a given value is a valid int. By default, any int is valid,
    but it is also possible to put an upper- and lower-bound on the valid range
    of floats.

    Parameters
    ----------
    minimum: int, default=None
        The minimum value a given int can have. If ``None``, there is no lower-bound.
    maximum: int, default=None
        The maximum value a given int can have. If ``None``, there is no upper-bound.

    Examples
    --------
    >>> from dtaianomaly.type_validation import IntegerAttribute
    >>> an_int = IntegerAttribute(minimum=0)
    >>> an_int.raise_error_if_invalid(0, "my_attribute", "MyClass")   # No error
    >>> an_int.raise_error_if_invalid(42, "my_attribute", "MyClass")  # No error
    >>> an_int.raise_error_if_invalid("not-an-int", "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'my_attribute' in class 'MyClass' must be of type int, but received 'not-an-int' of type <class 'str'>!
    >>> an_int.raise_error_if_invalid(-1, "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    ValueError: Attribute 'my_attribute' in class 'MyClass' must be greater than or equal to 0, but received '-1'!
    """

    _minimum: int | None
    _maximum: int | float | None

    def __init__(self, minimum: int = None, maximum: int = None):
        # Check minimum
        if minimum is not None:
            if not isinstance(minimum, int) or isinstance(minimum, bool):
                raise TypeError(
                    f"Attribute 'minimum' in class 'IntegerAttribute' must be of type int, but received '{minimum}' of type {type(minimum)}!"
                )

        # Check maximum
        if maximum is not None:
            if not isinstance(maximum, int) or isinstance(maximum, bool):
                raise TypeError(
                    f"Attribute 'maximum' in class 'IntegerAttribute' must be of type int, but received '{maximum}' of type {type(maximum)}!"
                )

        self._minimum = minimum
        self._maximum = maximum

    @property
    def minimum(self) -> int | None:
        return self._minimum

    @property
    def maximum(self) -> int | None:
        return self._maximum

    def _is_valid_type(self, value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def _get_valid_type_description(self) -> str:
        return "int"

    def _is_valid_value(self, value) -> bool:
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True

    def _get_valid_value_description(self) -> str:
        if self.minimum is None and self.maximum is not None:
            return f"less than or equal to {self.maximum}"
        if self.minimum is not None and self.maximum is None:
            return f"greater than or equal to {self.minimum}"
        if self.minimum is not None and self.maximum is not None:
            return f"in range [{self.minimum}, {self.maximum}]"
        return "int"
