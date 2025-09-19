import abc

from dtaianomaly.type_validation._BaseAttributeValidation import BaseAttributeValidation

__all__ = ["AttributeValidationMixin"]


class AttributeValidationMixin(abc.ABC):
    """
    Mixin object to automatically validate class-attributes.

    Mixin object to add functionality for validating the types of the attributes.
    By adding this class as parent to the new class, and defining the property
    :py:attr:`~dtaianomaly.type_validation.AttributeValidationMixin.attribute_validation`,
    the defined attributes will always be verified when their value changes (upon
    construction and upon setting).

    Attribute validation can also deal with using inheritance: only the parent class
    needs to add the :py:class:`~dtaianomaly.type_validation.AttributeValidationMixin`
    as a base, and then in each class you can define the valid types in
    :py:attr:`~dtaianomaly.type_validation.AttributeValidationMixin.attribute_validation`.

    Note that defining attributes in the :py:attr:`~dtaianomaly.type_validation.AttributeValidationMixin.attribute_validation`
    dictionary is completely optional. It is possible to exclude attributes from this
    dictionary. For such excluded attriutes, when there is a state change (e.g.,
    construction or through the setter), the change will simply happen without any
    checks.

    Attributes
    ----------
    attribute_validation: dict of (str, BaseAttributeValidation)
        The validation for each attribute. The key is the name of the attribute
        to check, and the value is the related BaseAttributeValidation that will
        verify the correct state of the property.

    Examples
    --------
    >>> from dtaianomaly.type_validation import AttributeValidationMixin, IntegerAttribute
    >>>
    >>> class MyObject(AttributeValidationMixin):
    ...     positive_int: int
    ...     attribute_validation = {"positive_int": IntegerAttribute(minimum=0)}
    ...
    ...     def __init__(self, positive_int: int):
    ...         self.positive_int = positive_int
    >>>
    >>> MyObject("not-an-int")  # Constructing with invalid type
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'positive_int' in class 'MyObject' must be of type int, but received 'not-an-int' of type <class 'str'>!
    >>> MyObject(-1)  # Constructing with invalid value
    Traceback (most recent call last):
        ...
    ValueError: Attribute 'positive_int' in class 'MyObject' must be greater than or equal to 0, but received '-1'!
    >>> my_object = MyObject(42)
    >>> my_object.positive_int
    42
    >>> my_object.positive_int = "another-one"  # Setting to invalid type
    Traceback (most recent call last):
        ...
    TypeError: Attribute 'positive_int' in class 'MyObject' must be of type int, but received 'another-one' of type <class 'str'>!
    >>> my_object.positive_int = -5  # Setting to invalid value
    Traceback (most recent call last):
        ...
    ValueError: Attribute 'positive_int' in class 'MyObject' must be greater than or equal to 0, but received '-5'!
    >>> my_object.positive_int = 1
    >>> my_object.positive_int
    1
    """

    attribute_validation: dict[str, BaseAttributeValidation]

    @property
    def _all_subclass_attribute_validation(
        self,
    ) -> dict[str, "BaseAttributeValidation"]:

        return {
            attribute_name: attribute_validation
            for cls in self.__class__.__mro__
            if issubclass(cls, AttributeValidationMixin)
            and "attribute_validation" in cls.__dict__
            for attribute_name, attribute_validation in cls.attribute_validation.items()
        }

    def __setattr__(self, name, value):
        # The attribute validation is immutable
        if name == "attribute_validation":
            raise AttributeError(
                "AttributeError: can't set attribute 'attribute_validation'"
            )

        # Check if the value is
        if name in self._all_subclass_attribute_validation:
            self._all_subclass_attribute_validation[name].raise_error_if_invalid(
                value, name, self.__class__.__name__
            )

        # Set the value
        super().__setattr__(name, value)
