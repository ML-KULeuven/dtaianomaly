from dtaianomaly.type_validation._BaseAttributeValidation import BaseAttributeValidation

__all__ = ["FloatAttribute"]


class FloatAttribute(BaseAttributeValidation):
    """
    Validator for floats.

    Check wether a given value is a valid float. By default, any float is valid,
    but it is also possible to put an upper- and lower-bound on the valid range
    of floats. Both upper- and lower-bound can be an inclusive or exclusive bound.

    Parameters
    ----------
    minimum : float, default=None
        The minimum value a given float can have. If ``None``, there is no lower-bound.
    maximum : float, default=None
        The maximum value a given float can have. If ``None``, there is no upper-bound.
    inclusive_minimum : bool, default=True
        Whether the lower-bound itself is valid or not.
    inclusive_maximum : bool, default=True
        Whether the upper-bound itself is valid or not.

    Examples
    --------
    >>> from dtaianomaly.type_validation import FloatAttribute
    >>> a_float = FloatAttribute(minimum=0.0, maximum=1.0, inclusive_maximum=False)
    >>> a_float.raise_error_if_invalid(0.0, "my_attribute", "MyClass")  # No error
    >>> a_float.raise_error_if_invalid(0.5, "my_attribute", "MyClass")  # No error
    >>> a_float.raise_error_if_invalid("not-a-float", "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'my_attribute' in class 'MyClass' must be of type float, but received 'not-a-float' of type <class 'str'>!
    >>> a_float.raise_error_if_invalid(1.0, "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    ValueError: Attribute 'my_attribute' in class 'MyClass' must be in range [0.0, 1.0[, but received '1.0'!
    """

    _minimum: float | None
    _inclusive_minimum: bool
    _maximum: float | None
    _inclusive_maximum: bool

    def __init__(
        self,
        minimum: float = None,
        maximum: float = None,
        inclusive_minimum: bool = True,
        inclusive_maximum: bool = True,
    ):
        if minimum is not None:
            if not isinstance(minimum, float) or isinstance(minimum, bool):
                raise TypeError(
                    f"Attribute 'minimum' in class 'FloatAttribute' must be of type float, but received '{minimum}' of type {type(minimum)}!"
                )

        if maximum is not None:
            if not isinstance(maximum, float) or isinstance(maximum, bool):
                raise TypeError(
                    f"Attribute 'maximum' in class 'FloatAttribute' must be of type float, but received '{maximum}' of type {type(maximum)}!"
                )

        if minimum is not None and maximum is not None:
            if minimum > maximum:
                raise ValueError(
                    f"Attribute 'minimum' must be smaller than or equal to attribute 'maximum' in class 'IntegerAttribute'!"
                )

        if not isinstance(inclusive_minimum, bool):
            raise TypeError(
                f"Attribute 'inclusive_minimum' in class 'FloatAttribute' must be of type bool, but received '{inclusive_minimum}' of type {type(inclusive_minimum)}!"
            )

        if not isinstance(inclusive_maximum, bool):
            raise TypeError(
                f"Attribute 'inclusive_maximum' in class 'FloatAttribute' must be of type bool, but received '{inclusive_maximum}' of type {type(inclusive_maximum)}!"
            )

        self._minimum = minimum
        self._maximum = maximum
        self._inclusive_minimum = inclusive_minimum
        self._inclusive_maximum = inclusive_maximum

    @property
    def minimum(self) -> float | None:
        return self._minimum

    @property
    def inclusive_minimum(self) -> bool:
        return self._inclusive_minimum

    @property
    def maximum(self) -> float | None:
        return self._maximum

    @property
    def inclusive_maximum(self) -> bool:
        return self._inclusive_maximum

    def _is_valid_type(self, value) -> bool:
        return isinstance(value, float) and not isinstance(value, bool)

    def _get_valid_type_description(self) -> str:
        return "float"

    def _is_valid_value(self, value) -> bool:
        if self.minimum is not None:
            if self.inclusive_minimum and value < self.minimum:
                return False
            if not self.inclusive_minimum and value <= self.minimum:
                return False

        if self.maximum is not None:
            if self.inclusive_maximum and value > self.maximum:
                return False
            if not self.inclusive_maximum and value >= self.maximum:
                return False

        return True

    def _get_valid_value_description(self) -> str:
        if self.minimum is None and self.maximum is not None:
            if self.inclusive_maximum:
                return f"less than or equal to {self.maximum}"
            else:
                return f"less than {self.maximum}"

        if self.minimum is not None and self.maximum is None:
            if self.inclusive_minimum:
                return f"greater than or equal to {self.minimum}"
            else:
                return f"greater than {self.minimum}"

        if self.minimum is not None and self.maximum is not None:
            result = "in range "
            result += "[" if self.inclusive_minimum else "]"
            result += f"{self.minimum}, {self.maximum}"
            result += "]" if self.inclusive_maximum else "["
            return result

        return "float"
