import abc

from dtaianomaly.evaluation._Metric import Metric

__all__ = ["ProbaMetric"]


class ProbaMetric(Metric, abc.ABC):
    """A metric that takes as input continuous anomaly scores."""
