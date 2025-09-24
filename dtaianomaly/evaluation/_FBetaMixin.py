from dtaianomaly.type_validation import (
    AttributeValidationMixin,
    FloatAttribute,
    IntegerAttribute,
)

__all__ = ["FBetaMixin"]


class FBetaMixin(AttributeValidationMixin):
    """
    Base class for all F-Beta based metrics. Takes a beta value, checks if it
    is correct, and offers a method to compute the F-score for a given precision
    and recall.

    Parameters
    ----------
    beta: int, float, default=1
        Desired beta parameter.
    """

    beta: float | int
    attribute_validation = {
        "beta": FloatAttribute(minimum=0.0, inclusive_minimum=False)
        | IntegerAttribute(minimum=1)
    }

    def __init__(self, beta: float | int = 1) -> None:
        self.beta = beta

    def _f_score(self, precision: float, recall: float) -> float:
        numerator = (1 + self.beta**2) * precision * recall
        denominator = self.beta**2 * precision + recall
        return 0.0 if denominator == 0 else numerator / denominator
