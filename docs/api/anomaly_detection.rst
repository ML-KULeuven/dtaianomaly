
Anomaly detection module
========================

.. automodule:: dtaianomaly.anomaly_detection

API cheatsheet
--------------

Below there is a quick overview of the most essential methods to
detect anomalies:

#. :py:meth:`~dtaianomaly.anomaly_detection.BaseDetector.fit`. Fit the anomaly
   detector. This method requires both an ``X`` (the time series) and ``y``
   (anomaly labels) parameter. However, in practice, most anomaly detectors
   will not use the ground truth labels. The parameter ``y`` is present for
   API consistency and is not required.

#. :py:meth:`~dtaianomaly.anomaly_detection.BaseDetector.decision_function`.
   Compute the decision scores of an observation being an anomaly for a given
   time series ``X``. Returns an array with an entry for each observation in
   the time series. Note that this score is not normalized, and depends on
   the specific anomaly detector. However, for all detectors, a higher score
   means `more anomalous`.

#. :py:meth:`~dtaianomaly.anomaly_detection.BaseDetector.predict_proba`. Compute
   the probability of an anomaly being an anomaly. This is similar to the
   :py:meth:`~dtaianomaly.anomaly_detection.BaseDetector.decision_function`
   method, but the computed scores are normalized to the interval :math:`[0, 1]`,
   which enables the interpretation as a probability.

   .. note::
      The output of a ``predict_proba`` is often a matrix of size ``(n_samples, n_classes)``.
      For anomaly detection, this would lead to a matrix with two columns, one columns
      for the normal probabilities and one column for the anomalous probabilities.
      However, in ``dtaianomaly``, the :py:meth:`~dtaianomaly.anomaly_detection.BaseDetector.predict_proba`
      only returns the probability of a sample being anomalous, because this is
      the probability of interest in many anomaly detection applications.


Implemented anomaly detectors
-----------------------------

.. toctree::
   :maxdepth: 1
   :glob:

   anomaly_detection_algorithms/*


BaseDetector
------------

.. autoclass:: dtaianomaly.anomaly_detection.BaseDetector
   :members:


Utilities
---------

.. autofunction:: dtaianomaly.anomaly_detection.load_detector
.. autofunction:: dtaianomaly.anomaly_detection.sliding_window
.. autofunction:: dtaianomaly.anomaly_detection.reverse_sliding_window
.. autofunction:: dtaianomaly.anomaly_detection.check_is_valid_window_size
.. autofunction:: dtaianomaly.anomaly_detection.compute_window_size
