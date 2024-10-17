
Anomaly detection module
========================

.. automodule:: dtaianomaly.anomaly_detection

API cheatsheet
--------------

Below there is a quick overview of the most essential methods to
detect anomalies:

#. :py:meth:`dtaianomaly.anomaly_detection.BaseDetector.fit`. Fit the anomaly
   detector. This method requires both an ``X`` (the time series) and ``y``
   (anomaly labels) parameter. However, in practice, most anomaly detectors
   will not use the ground truth labels. The parameter ``y`` is present for
   API consistency and is not required.

#. :py:meth:`dtaianomaly.anomaly_detection.BaseDetector.decision_function`.
   Compute the decision scores of an observation being an anomaly for a given
   time series ``X``. Returns an array with an entry for each observation in
   the time series. Note that this score is not normalized, and depends on
   the specific anomaly detector. However, for all detectors, a higher score
   means `more anomalous`.

#. :py:meth:`dtaianomaly.anomaly_detection.BaseDetector.predict_proba`. Compute
   the probability of an anomaly being an anomaly. This is similar to the
   :py:meth:`~dtaianomaly.anomaly_detection.BaseDetector.decision_function`
   method, but the computed scores are normalized to the interval :math:`[0, 1]`,
   which enables the interpretation as a probability.

Implemented anomaly detectors
-----------------------------

.. toctree::
   :maxdepth: 1

   anomaly_detection_algorithms/baselines
   anomaly_detection_algorithms/isolation_forest
   anomaly_detection_algorithms/local_outlier_factor
   anomaly_detection_algorithms/matrix_profile_detector


BaseDetector
------------

.. autoclass:: dtaianomaly.anomaly_detection.BaseDetector
   :members:


Utilities
---------

.. autofunction:: dtaianomaly.anomaly_detection.load_detector
.. autofunction:: dtaianomaly.anomaly_detection.sliding_window
.. autofunction:: dtaianomaly.anomaly_detection.reverse_sliding_window
