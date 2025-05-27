import abc


class FBetaBase(abc.ABC):
    """
    Base class for all F-Beta based metrics. Takes a beta value, checks if it
    is correct, and offers a method to compute the F-score for a given precision
    and recall.
    """

    beta: float

    def __init__(self, beta: (float, int)) -> None:
        if not isinstance(beta, (int, float)) or isinstance(beta, bool):
            raise TypeError("`beta` should be numeric")
        if beta <= 0.0:
            raise ValueError("`beta` should be strictly positive")
        self.beta = beta

    def _f_score(self, precision: float, recall: float) -> float:
        numerator = (1 + self.beta**2) * precision * recall
        denominator = self.beta**2 * precision + recall
        return 0.0 if denominator == 0 else numerator / denominator
