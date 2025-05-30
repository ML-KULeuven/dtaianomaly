Range-based metrics
===================

Implementations of the range-based metrics proposed by :cite:`tatbul2018precision`.
In contrast to traditional metrics such as precision and recall, these metrics take
the temporal nature of time series into account. Another advantage of these metrics
is that they can be tuned to better evaluate the performance of an anomaly detector
for a specific domain. Specifically, you can tune the following criteria:

1. *Existence:* The importance of catching an true anomaly, even if it is only using
   a single predicted point, might by itself be useful for certain applications.
2. *Position:* In some cases, not only the size, but also the relative position of the
   predicted anomaly in the true anomaly may be important (e.g., early detection).
3. *Cardinality:* Detecting an true anomaly with only one predicted anomalous interval
   may be more valuable than doing so with multiple different ranges in a fragmented manner.

:cite:author:`tatbul2018precision` :cite:yearpar:`tatbul2018precision` also proposed to allow for
flexible inclusion of the size of the detected anomaly, by replacing the :math:`\omega()`-function.
This option is not included in our implementation.

.. warning::

    Note that, while tuning a metric to some domain is beneficial in practical applications,
    this flexibility makes it difficult for a large-scale, general-purpose evaluation of
    multiple anomaly detectors, as you can optimize the metric for a specific application.

.. autoclass:: dtaianomaly.evaluation.RangeBasedPrecision
.. autoclass:: dtaianomaly.evaluation.RangeBasedRecall
.. autoclass:: dtaianomaly.evaluation.RangeBasedFBeta
