Event-wise metrics
==================

Implementations of the event wise metrics proposed by [el2024multivariate]_. These
metrics will not look at the labeling of the individual points, but rather look at
the event-level capabilities of the models.

.. [el2024multivariate] El Amine Sehili, Mohamed, and Zonghua Zhang. "Multivariate time series anomaly detection:
   Fancy algorithms and flawed evaluation methodology." Technology Conference on Performance
   Evaluation and Benchmarking (2024), vol 14247, doi: `10.1007/978-3-031-68031-1_1 <https://doi.org/10.1007/978-3-031-68031-1_1>`_.

.. autoclass:: dtaianomaly.evaluation.EventWisePrecision
.. autoclass:: dtaianomaly.evaluation.EventWiseRecall
.. autoclass:: dtaianomaly.evaluation.EventWiseFBeta
