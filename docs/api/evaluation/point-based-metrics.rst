Point-based metrics
===================

.. warning::
   It is known that the point-adjusted metrics heavily overestimate
   the performance of anomaly detectors. It is therefore not recommended
   to solely rely on those metrics to evaluate a model. These metrics
   were implemented for reproducibility of existing works.

.. autoclass:: dtaianomaly.evaluation.Precision
.. autoclass:: dtaianomaly.evaluation.Recall
.. autoclass:: dtaianomaly.evaluation.FBeta
.. autoclass:: dtaianomaly.evaluation.PointAdjustedPrecision
.. autoclass:: dtaianomaly.evaluation.PointAdjustedRecall
.. autoclass:: dtaianomaly.evaluation.PointAdjustedFBeta
