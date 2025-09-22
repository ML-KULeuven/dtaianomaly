from dtaianomaly.type_validation._BaseAttributeValidation import BaseAttributeValidation

__all__ = ["ObjectAttribute"]


class ObjectAttribute(BaseAttributeValidation):
    """
    Validate if a given value has a specific type.

    Examples
    --------
    >>> from dtaianomaly.type_validation import ObjectAttribute
    >>> an_object = ObjectAttribute(int)
    >>> an_object.raise_error_if_invalid(5, "my_attribute", "MyClass")   # No error
    >>> an_object.raise_error_if_invalid(-1, "my_attribute", "MyClass")  # No error
    >>> an_object.raise_error_if_invalid("not-an-object", "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'my_attribute' in class 'MyClass' must be of type 'type <class 'int'>', but received 'not-an-object' of type <class 'str'>!
    """

    _object_type: type

    def __init__(self, object_type: type):
        if not isinstance(object_type, type):
            raise TypeError(
                f"Attribute 'object_type' in class 'ObjectAttribute' must be of type 'type', but received '{object_type}' of type {type(object_type)}!"
            )
        self._object_type = object_type

    @property
    def object_type(self) -> type:
        return self._object_type

    def _is_valid_type(self, value) -> bool:
        if self.object_type == int and isinstance(value, bool):
            return False
        return issubclass(value.__class__, self.object_type)

    def _get_valid_type_description(self) -> str:
        return f"'type {self.object_type}'"

    def _is_valid_value(self, value) -> bool:
        return self._is_valid_type(value)

    def _get_valid_value_description(self) -> str:
        return f"'type {self.object_type}'"
