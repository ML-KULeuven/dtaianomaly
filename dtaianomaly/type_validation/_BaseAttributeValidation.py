import abc

__all__ = ["BaseAttributeValidation", "UnionAttribute"]


class BaseAttributeValidation(abc.ABC):
    """
    Base class for Attribute validation.

    This abstract class offers the base functionality to check if a
    value is valid according to some rules. These rules are defined
    within the implementing classes. The :py:class:`~dtaianomaly.type_validation.BaseAttributeValidation`
    checks both the type and value for some attribute, and raises a
    human-readable error message if anything is wrong.
    """

    def raise_error_if_invalid(self, value, name: str, class_name: str) -> None:
        """
        Raise an error if the given value is invalid.

        Check if the type and the exact value are permitted, according to the
        rules of this attribute validation. If either the type or the value is
        invalid, an error is raised accordingly. Otherwise, nothing happens.

        Parameters
        ----------
        value : any
            The value to verify.
        name : str
            The name of the attribute that is being checked. Used for generating clear error messages, if needed.
        class_name : str
            The name of the class to which the attribute belongs. Used for generating clear error messages, if needed.

        Raises
        ------
        TypeError
            If the type of the given value is invalid according to the rules of this BaseAttributeValidation.
        ValueError
            If the value of the given value is invalid according to the rules of this BaseAttributeValidation.
        """
        if not self._is_valid_type(value):
            raise TypeError(
                f"Attribute '{name}' in class '{class_name}' must be of type {self._get_valid_type_description()}, but received '{value}' of type {type(value)}!"
            )
        if not self._is_valid_value(value):
            raise ValueError(
                f"Attribute '{name}' in class '{class_name}' must be {self._get_valid_value_description()}, but received '{value}'!"
            )

    def __or__(self, other: "BaseAttributeValidation") -> "UnionAttribute":
        if not isinstance(other, BaseAttributeValidation):
            raise TypeError(
                f"unsupported operand type(s) for |: 'BaseAttributeValidation' and '{other.__class__.__name__}'"
            )
        return UnionAttribute(self, other)

    @abc.abstractmethod
    def _is_valid_type(self, value) -> bool:
        """Check if the given value has a valid type."""

    @abc.abstractmethod
    def _get_valid_type_description(self) -> str:
        """Get a description of the valid type, used for the error message."""

    @abc.abstractmethod
    def _is_valid_value(self, value) -> bool:
        """Check if the given value is valid."""

    @abc.abstractmethod
    def _get_valid_value_description(self) -> str:
        """Get a description of the valid value, used for the error message."""


class UnionAttribute(BaseAttributeValidation):
    """
    Validate multiple :py:class:`~dtaianomaly.type_validation.BaseAttributeValidation` objects.

    Combine multiple :py:class:`~dtaianomaly.type_validation.BaseAttributeValidation`
    using an OR-operation. This validator will check if a given value
    satisfies the rules of the :py:class:`~dtaianomaly.type_validation.BaseAttributeValidation`.
    Multiple attribute validators can be either passed to the constructor,
    but it is also possible to combine them with ``|`` for a simpler and
    more pythonic syntax.

    Parameters
    ----------
    *attribute_validators : two or more :py:class:`~dtaianomaly.type_validation.BaseAttributeValidation` objects
        The attribute validators that are combined through an OR-operation for validation.
        The validators must be passes as separate parameters to the constructor. There should
        be at least two validators. If any of the given validators is a :py:class:`~dtaianomaly.type_validation.UnionAttribute`,
        then the validators will be flattened to contain only a single list of validators.

    Examples
    --------
    >>> from dtaianomaly.type_validation import IntegerAttribute, NoneAttribute
    >>> optional_int = IntegerAttribute(minimum=0) | NoneAttribute()
    >>> # Equivalently: optional_int = UnionAttribute(IntegerAttribute(minimum=0), NoneAttribute())
    >>> optional_int.raise_error_if_invalid(None, "my_attribute", "MyClass")  # No error
    >>> optional_int.raise_error_if_invalid(42, "my_attribute", "MyClass")    # No error
    >>> optional_int.raise_error_if_invalid("not-an-optional-int", "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'my_attribute' in class 'MyClass' must be of type int or None, but received 'not-an-optional-int' of type <class 'str'>!
    >>> optional_int.raise_error_if_invalid(-5, "my_attribute", "MyClass")
    Traceback (most recent call last):
        ...
    ValueError: Attribute 'my_attribute' in class 'MyClass' must be greater than or equal to 0 or None, but received '-5'!
    """

    _attribute_validators: list[BaseAttributeValidation]

    def __init__(self, *attribute_validators: BaseAttributeValidation):

        # Checks
        for attribute_validator in attribute_validators:
            if not isinstance(attribute_validator, BaseAttributeValidation):
                raise TypeError(
                    f"All elements of attribute 'attribute_validators' in class 'UnionAttribute' must be of type BaseAttributeValidation, but received '{attribute_validator}' of type {type(attribute_validator)}!"
                )
        if len(attribute_validators) <= 1:
            raise ValueError(
                "Attribute 'values' in class 'LiteralAttribute' must consist of at least two elements!"
            )

        # flatten nested unions
        self._attribute_validators = []
        for validator in attribute_validators:
            if isinstance(validator, UnionAttribute):
                self._attribute_validators.extend(validator.attribute_validators)
            else:
                self._attribute_validators.append(validator)

    @property
    def attribute_validators(self) -> list[BaseAttributeValidation]:
        return self._attribute_validators

    def _is_valid_type(self, value) -> bool:
        return any(
            validator._is_valid_type(value) for validator in self.attribute_validators
        )

    def _get_valid_type_description(self) -> str:
        types = [
            validator._get_valid_type_description()
            for validator in self.attribute_validators
        ]
        return ", ".join(types[:-1]) + " or " + types[-1]

    def _is_valid_value(self, value) -> bool:
        return any(
            validator._is_valid_type(value) and validator._is_valid_value(value)
            for validator in self.attribute_validators
        )

    def _get_valid_value_description(self) -> str:
        values = [
            validator._get_valid_value_description()
            for validator in self.attribute_validators
        ]
        return ", ".join(values[:-1]) + " or " + values[-1]
