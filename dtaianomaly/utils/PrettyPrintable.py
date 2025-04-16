import abc
import inspect


class PrettyPrintable(abc.ABC):

    def __str__(self) -> str:
        return initialization_call_string(self)


def initialization_call_string(o: object) -> str:
    parameters = {
        parameter: getattr(o, parameter)
        for parameter, value in inspect.signature(o.__init__).parameters.items()
        if parameter not in ["args", "kwargs"]
        and value.default != getattr(o, parameter)
    }
    if hasattr(o, "kwargs"):
        parameters.update(o.kwargs)
    return (
        o.__class__.__name__
        + "("
        + ",".join(
            [
                f"{parameter}={string_with_apostrophe(value)}"
                for parameter, value in parameters.items()
            ]
        )
        + ")"
    )


def string_with_apostrophe(s):
    return f"'{s}'" if isinstance(s, str) else s
