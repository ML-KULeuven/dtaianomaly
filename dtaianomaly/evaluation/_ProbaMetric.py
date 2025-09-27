import abc

from dtaianomaly.evaluation._Metric import Metric

__all__ = ["ProbaMetric"]


class ProbaMetric(Metric, abc.ABC):
    """
    Base class for probabilistic metrics.

    A class to indicate that a metric is probabilistic, i.e., that it computes
    a performance based on the continuous anomaly scores. This is mainly
    useful to differentiate with binary metrics.
    """
