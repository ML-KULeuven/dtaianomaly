from dtaianomaly.type_validation._BaseAttributeValidation import BaseAttributeValidation
from dtaianomaly.type_validation._IntegerAttribute import IntegerAttribute
from dtaianomaly.type_validation._LiteralAttribute import LiteralAttribute
from dtaianomaly.windowing import AUTO_WINDOW_SIZE_COMPUTATION

__all__ = ["WindowSizeAttribute"]


class WindowSizeAttribute(BaseAttributeValidation):
    """
    Validator for a window size.

    Check wether a given value is a valid window size. A valid window size
    is either an int greater than or equal to 1, or a literal in {'fft',
    'suss', 'acf', 'mwf'}.

    Examples
    --------
    >>> from dtaianomaly.type_validation import WindowSizeAttribute
    >>> a_window_size = WindowSizeAttribute()
    >>> a_window_size.raise_error_if_invalid(1, "my_attribute", "MyClass")        # No error
    >>> a_window_size.raise_error_if_invalid(64, "my_attribute", "MyClass")       # No error
    >>> a_window_size.raise_error_if_invalid('fft', "my_attribute", "MyClass")    # No error
    >>> a_window_size.raise_error_if_invalid('suss', "my_attribute", "MyClass")   # No error
    >>> a_window_size.raise_error_if_invalid('acf', "my_attribute", "MyClass")    # No error
    >>> a_window_size.raise_error_if_invalid('mwf', "my_attribute", "MyClass")    # No error
    >>> a_window_size.raise_error_if_invalid(1.0, "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'my_attribute' in class 'MyClass' must be of type int or string, but received '1.0' of type <class 'float'>!
    >>> a_window_size.raise_error_if_invalid("not-a-window-size", "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    ValueError: Attribute 'my_attribute' in class 'MyClass' must be greater than or equal to 1 or in {'acf', 'fft', 'mwf', 'suss'}, but received 'not-a-window-size'!
    >>> a_window_size.raise_error_if_invalid(0, "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    ValueError: Attribute 'my_attribute' in class 'MyClass' must be greater than or equal to 1 or in {'acf', 'fft', 'mwf', 'suss'}, but received '0'!
    """

    validator: BaseAttributeValidation = IntegerAttribute(minimum=1) | LiteralAttribute(
        AUTO_WINDOW_SIZE_COMPUTATION
    )

    def _is_valid_type(self, value) -> bool:
        return self.validator._is_valid_type(value)

    def _get_valid_type_description(self) -> str:
        return self.validator._get_valid_type_description()

    def _is_valid_value(self, value) -> bool:
        return self.validator._is_valid_value(value)

    def _get_valid_value_description(self) -> str:
        return self.validator._get_valid_value_description()
