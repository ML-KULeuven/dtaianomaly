Event-wise metrics
==================

Implementations of the event wise metrics proposed by :cite:`el2024multivariate`. These
metrics will not look at the labeling of the individual points, but rather look at
the event-level capabilities of the models.

.. autoclass:: dtaianomaly.evaluation.EventWisePrecision
.. autoclass:: dtaianomaly.evaluation.EventWiseRecall
.. autoclass:: dtaianomaly.evaluation.EventWiseFBeta
