from pathlib import Path

from dtaianomaly.type_validation._BaseAttributeValidation import BaseAttributeValidation

__all__ = ["PathAttribute"]


class PathAttribute(BaseAttributeValidation):
    """
    Validate if a given value is a valid path.

    A valid path can be either a string or a pathlib.Path object, and the
    path must exist in the file system.

    Parameters
    ----------
    must_exist: bool, default=True
        Whether the path must exist.

    Examples
    --------
    >>> from dtaianomaly.type_validation import PathAttribute
    >>> a_path = PathAttribute()
    >>> a_path.raise_error_if_invalid(".", "my_attribute", "MyClass")  # No error
    >>> a_path.raise_error_if_invalid(Path("."), "my_attribute", "MyClass")  # No error
    >>> a_path.raise_error_if_invalid(123, "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'my_attribute' in class 'MyClass' must be of type str or pathlib.Path, but received '123' of type <class 'int'>!
    >>> a_path.raise_error_if_invalid("nonexistent_file.txt", "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    ValueError: Attribute 'my_attribute' in class 'MyClass' must be an existing path, but received 'nonexistent_file.txt'!
    """

    _must_exist: bool

    def __init__(self, must_exist: bool = True):
        if not isinstance(must_exist, bool):
            raise TypeError(
                f"Attribute 'must_exist' in class 'PathAttribute' must be of type bool, but received '{must_exist}' of type {type(must_exist)}!"
            )
        self._must_exist = must_exist

    @property
    def must_exist(self) -> bool:
        return self._must_exist

    def _is_valid_type(self, value) -> bool:
        return isinstance(value, (str, Path))

    def _get_valid_type_description(self) -> str:
        return "str or pathlib.Path"

    def _is_valid_value(self, value) -> bool:
        path = Path(value) if isinstance(value, str) else value
        return path.exists() or not self.must_exist

    def _get_valid_value_description(self) -> str:
        return "an existing path" if self.must_exist else "a path"
