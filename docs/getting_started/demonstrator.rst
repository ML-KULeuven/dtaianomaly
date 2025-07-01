Demonstrator
============

One of the main goals of ``dtaianomaly`` is to offer a simple API for state-of-the-art
time series anomaly detection models. To further simplify the analysis of multiple
anomaly detectors, ``dtaianomaly`` offers this demonstrator, which enables you to
detect anomalies without having to write any code.

The instructions below describe how you can run the demonstrator locally.

Setting up your environment
---------------------------

It is relatively simple to run the demonstrator locally through ``dtaianomaly``.
First, make sure that the environment is setup correctly. For this you need to
install ``dtaianomaly``, along with the optional dependencies ``demonstrator``:

.. code-block:: bash

    pip install dtaianomaly[demonstrator]

You can go to the :doc:`installation page <installation>` for more information about how to install ``dtaianomaly``.

Starting the demonstrator
-------------------------

There are two ways to start the demonstrator. The first option is through the
command line using the following command:

.. code-block:: bash

    run-demonstrator

The second option is to start the demonstrator programmatically using ``Python``.
First, you should import the demonstrator module from ``dtaianomaly``. Then, you
can call the :py:meth:`~dtaianomaly.demonstrator.run` method to start the demonstrator.
Starting the demonstrator from code has the added benefit that you can include
custom components in the demonstrator, as will be discussed below.

.. code-block:: python

    from dtaianomaly import demonstrator
    demonstrator.run()

.. autofunction:: dtaianomaly.demonstrator.run

Custom components
-----------------

One of the key-strengths of the ``dtaianomaly`` demonstrator is that it is
possible to easily include custom components. Specifically, you can include
a custom
:py:meth:`~dtaianomaly.data.LazyDataLoader`,
:py:meth:`~dtaianomaly.anomaly_detection.BaseDetector`, or
:py:meth:`~dtaianomaly.evaluation.Metric`.
You first need to implement the classes you want to integrate in the demonstrator,
as described on :doc:`this page <examples/extensibility>`, and then you can
pass the class to the :py:meth:`~dtaianomaly.demonstrator.run` method.

Below code illustrates how this can be done, in which we assume that detector
``NbSigmaAnomalyDetector`` as implemented :ref:`here <custom-anomaly-detector>`
is available in the file ``NbSigmaAnomalyDetector.py``:

.. code-block:: python

    from dtaianomaly import demonstrator
    from NbSigmaAnomalyDetector import NbSigmaAnomalyDetector
    demonstrator.run(custom_anomaly_detectors=NbSigmaAnomalyDetector)

Custom visualizations
---------------------

For some detectors, it is possible to show additional information besides
the anomaly scores. This information is specific to the anomaly detector.
For example, :py:class:`~dtaianomaly.anomaly_detection.KMeansAnomalyDetector` and
:py:class:`~dtaianomaly.anomaly_detection.KShapeAnomalyDetector` use
cluster-centroids to detect anomalies. These cluster centers indicate
the normal behavior, and is useful to understand how the model
detects anomalies.

While some custom visualizations are already available, it is also possible
to add custom visualizations to the demonstrator. For this, you only need
to implement the :py:class:`~dtaianomaly.demonstrator.CustomDetectorVisualizer`.
This class has two methods: (1) :py:meth:`~dtaianomaly.demonstrator.CustomDetectorVisualizer.is_compatible`
which checks whether the visualization can be applied to the given anomaly
detector, and (2) :py:meth:`~dtaianomaly.demonstrator.CustomDetectorVisualizer.show_custom_visualization`
which effectively shows the visualization in a streamlit-application. Then,
similarly as above, you can pass this class to the :py:meth:`~dtaianomaly.demonstrator.run`
method, and your custom visualization will be included in the demonstrator.

.. autoclass:: dtaianomaly.demonstrator.CustomDetectorVisualizer
   :inherited-members:
   :members:

Configuration
-------------

A large part of the demonstrator is configured using a configuration file.
This file describes which models to show and which hyperparameters are
tunable.

The configuration file is in a ``json`` format with the following
keys: (1)``'data-loader'``, (2) ``'detector'``, and (3) ``'metric'``. The
corresponding values configure the data loaders, the anomaly detectors and
the evaluation metrics respectively.

Each of the three components has the following subitems:

- ``'default'``: the component to load upon starting the demonstrator.
  For the anomaly detectors and evaluation metrics, this can also be a
  list of multiple components to load.
- ``'exclude'``: the components to not show in the demonstrator.
- ``'parameters-required'``: A dictionary of required parameters.
  These parameters must be given upon initialization of the component
  (e.g., the window size). The keys in this dictionary are the names
  of the parameters and the values are their default value.
- ``'parameters-adjustable'``: the adjustable hyperparameters. A
  parameter is defined as an item in an dictionary, with as key the
  name of the parameter and as value a parameter-configuration. This is
  a dictionary with one special key: the ``'type'`` defines what input
  component (of streamlit) should be used to update the parameter. All
  other items in the parameter configuration are passed to that specific
  component.

To change the configuration, you can load the default configuration
file using the :py:meth:`~dtaianomaly.demonstrator.load_configuration`
method, but without providing any arguments. Then, you can adapt this
file to your needs, save it locally, and pass the path to the
:py:meth:`~dtaianomaly.demonstrator.run` method.

.. autofunction:: dtaianomaly.demonstrator.load_configuration
