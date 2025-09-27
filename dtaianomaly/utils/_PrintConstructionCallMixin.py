import abc
import inspect

__all__ = ["PrintConstructionCallMixin"]


class PrintConstructionCallMixin(abc.ABC):
    """
    Mixin object to print the construction call of an object.

    Mixin object to add functionality for automatically print the contruction
    call of an object, using the currently set paramters. By default, this
    mixin object will try to include a hyperparaemter as <name>=<value> in the
    construction-call, but if there are \\*arg values, then this is not possible
    and the first parameters are included as parameter. Note that this only
    works if the names of the parameters are identical to the names of the
    attributes.

    Examples
    --------
    >>> from dtaianomaly.utils import PrintConstructionCallMixin
    >>>
    >>> class MyObject(PrintConstructionCallMixin):
    ...
    ...     def __init__(self, obligated: int, optional: float = 3.14, *args, **kwargs):
    ...         self.obligated = obligated
    ...         self.optional = optional
    ...         self.args = args
    ...         self.kwargs = kwargs
    >>>
    >>> print(MyObject(42))
    MyObject(obligated=42)
    >>> print(MyObject(42, 1.41))
    MyObject(obligated=42,optional=1.41)
    >>> print(MyObject(42, 1.41, 1, 2, 3))
    MyObject(42,1.41,1,2,3)
    >>> print(MyObject(42, other_param="other-value"))
    MyObject(obligated=42,other_param='other-value')
    >>> print(MyObject(42, 1.41, 1, 2, 3, other_param="other-value"))
    MyObject(42,1.41,1,2,3,other_param='other-value')
    """

    def __str__(self) -> str:

        # Extract the signature
        signature = inspect.signature(self.__init__)

        # Check for empty constructor
        if "__init__" not in vars(self.__class__) and len(signature.parameters) == 2:

            first_param = list(signature.parameters.values())[0]
            first_is_args = (
                first_param.kind == inspect.Parameter.VAR_POSITIONAL
                and first_param.name == "args"
            )

            second_param = list(signature.parameters.values())[1]
            second_is_kwargs = (
                second_param.kind == inspect.Parameter.VAR_KEYWORD
                and second_param.name == "kwargs"
            )

            if first_is_args and second_is_kwargs:
                return f"{self.__class__.__name__}()"

        # Check if there are args
        has_args = False
        for parameter, value in signature.parameters.items():
            if (
                value.kind == inspect.Parameter.VAR_POSITIONAL
                and len(getattr(self, parameter)) > 0
            ):
                has_args = True

        # Variables to maintain
        args_to_print = []
        kwargs_to_print = {}
        args_are_passed = not has_args

        # Iterate over every parameter and add it to the correct list
        for parameter, value in signature.parameters.items():
            if not args_are_passed:
                if value.kind == inspect.Parameter.VAR_POSITIONAL:
                    args_are_passed = True
                    if len(getattr(self, parameter)) > 0:
                        args_to_print.extend(getattr(self, parameter))
                else:
                    args_to_print.append(getattr(self, parameter))

            else:
                if value.kind == inspect.Parameter.VAR_POSITIONAL:
                    pass  # Can be skipped

                elif value.kind == inspect.Parameter.VAR_KEYWORD:
                    if len(getattr(self, parameter)) > 0:
                        for kwarg, v in getattr(self, parameter).items():
                            kwargs_to_print[kwarg] = v
                else:
                    if value.default != getattr(self, parameter):
                        kwargs_to_print[parameter] = getattr(self, parameter)

        # Format the output string
        return (
            f"{self.__class__.__name__}"
            f"("
            f"{','.join(map(self._string_with_apostrophe, args_to_print))}"
            f"{',' if len(args_to_print) > 0 and len(kwargs_to_print) > 0 else ''}"
            f"{','.join(map(lambda k: f'{k}={self._string_with_apostrophe(kwargs_to_print[k])}', kwargs_to_print))}"
            f")"
        )

    def _string_with_apostrophe(self, obj):
        cls = obj.__class__
        if cls.__module__ != "builtins" and not issubclass(
            cls, PrintConstructionCallMixin
        ):
            raise AttributeError(
                f"All attributes of object '{self.__class__.__name__}' should either be built-in or inherit from 'PrintConstructionCallMixin', but has an attribute of type '{cls}'!"
            )

        return f"'{obj}'" if isinstance(obj, str) else f"{obj}"
