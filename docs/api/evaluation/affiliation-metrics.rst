Event-wise metrics
==================

Implementations of the affiliation-based metrics proposed by :cite:`huet2022local`.
These metrics will consider local affiliations around the ground truth anomalous
events, and compute a distance within these affiliations to derive precision, recall
and :math:`F_\\beta`-score.

.. autoclass:: dtaianomaly.evaluation.AffiliationPrecision
.. autoclass:: dtaianomaly.evaluation.AffiliationRecall
.. autoclass:: dtaianomaly.evaluation.AffiliationFBeta
