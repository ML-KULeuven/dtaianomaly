Examples
========

The main functionality of ``dtaianomaly`` is to detect anomalies in
time series data. Within ``dtaianomaly``, a time series is represented
as a Numpy-array of shape ``(n_samples, n_attributes)``, in which ``n_samples``
equals the number of measurements or observations in the time series, and
``n_attributes`` equals the number of variables that are being measured.

.. admonition:: Example

   Assume you have an accelerometer which measures the acceleration in the
   X, Y and Z direction with a sampling frequency of 100Hz (100 samples per
   second), measured over 15 seconds. The corresponding time series is a
   numpy array of shape (100*15, 3), which equals (1500, 3).

Specifically, ``dtaianomaly`` has the following three key features, which
are described in more detail below:

#. State-of-the-art time series anomaly detection via a simple API.
#. Develop custom models for anomaly detection.
#. Quantitative evaluation of time series anomaly detection.

.. toctree::
   :hidden:
   :glob:

   examples/*

.. rubric:: Anomaly detection
   :heading-level: 2

The most important functionality of ``dtaianomaly`` is to detect anomalies in
time series. Multiple state-of-the-art time series anomaly detectors are
implemented in ``dtaianomaly``, which can all be applied through a simple API,
similar to `scikit-learn <https://scikit-learn.org/stable/>`_. This means it
only takes a few lines of code to detect anomalies in your time series! For more
information regarding the anomaly detection API, we refer to the
:doc:`anomaly detection module<../api/anomaly_detection>`.

.. seealso::
    Check out the :doc:`anomaly detection example<examples/anomaly_detection>` for more information!


.. rubric:: Custom models
   :heading-level: 2

A key design philosophy of ``dtaianomaly`` is to easily implement custom components.
This can be a new anomaly detector being developed, some preprocessing step for your
time series data, or an evaluation metric which takes application specific KPIs into
account. For this, you only need to implement your component as a child of the type
of component you are implementing (e.g., :py:class:`~dtaianomaly.anomaly_detection.BaseDetector`
to implement an anomaly detector), after which the component can be used is if it
is natively a part of ``dtaianomaly``.

.. seealso::
    Check out the :doc:`custom model example<examples/custom_models>` for more information!


.. rubric:: Quantitative evaluation with a workflow
   :heading-level: 2

Oftentimes we need to select the best anomaly detector. For this, it is necessary
to perform a quantitative evaluation of the anomaly detectors, after which the
detector with highest performance can be chosen. To simply evaluate multiple
anomaly detectors with different preprocessing steps on multiple datasets, and
to measure different evaluation metrics, ``dtaianomaly`` offers the :py:class:`~dtaianomaly.workflow.Workflow`.
All you need to do is initialize the different components, pass them to the
:py:class:`~dtaianomaly.workflow.Workflow`, and call the :py:meth:`~dtaianomaly.workflow.Workflow.run`
method evaluate the anomaly detectors!

.. seealso::
    Check out the :doc:`workflow example <examples/quantitative_evaluation>` for more information!
