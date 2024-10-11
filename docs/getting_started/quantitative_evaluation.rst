Quantitative evaluation with a workflow
=======================================

It is crucial to qualitatively evaluate the performance of anomaly detectors
to know their capabilities. For this, ``dtaianomaly`` offers the :py:class:`~dtaianomaly.workflow.Workflow`:
detect anomalies in a large set of time series using various detectors, and to measure
their performance using multiple evaluation criteria. The :py:class:`~dtaianomaly.workflow.Workflow`
facilitates the validation of the anomaly detectors, because you only need to define
the different components.

There are two ways to run a :py:class:`~dtaianomaly.workflow.Workflow`: from :ref:`Python <with-code>`
or from a :ref:`configuration file <with-config>`.

.. note::
    You can also evaluate `custom components <https://dtaianomaly.readthedocs.io/en/stable/getting_started/custom_models.html>`_
    in ``dtaianomaly`` via a :py:class:`~dtaianomaly.workflow.Workflow` in :ref:`Python <with-code>`. However,
    this is not possible via a :ref:`configuration file <with-config>` without extending the functionality of
    the :py:func:`~dtaianomaly.workflow.workflow_from_config` function!

.. _with-code:

Run a workflow from Python
--------------------------

We first need to initialize the different components of the :py:class:`~dtaianomaly.workflow.Workflow`.
We start by creating a list of :py:class:`~dtaianomaly.data.LazyDataLoader` objects. We manually selected
two time series to use for evaluation, but alternatively you can use all datasets in some directory using
the :py:func:`~dtaianomaly.data.from_directory` method in the data module.

.. code-block:: python

    dataloaders = [
        UCRLoader('../data/UCR-time-series-anomaly-archive/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt'),
        UCRLoader('../data/UCR-time-series-anomaly-archive/002_UCR_Anomaly_DISTORTED2sddb40_35000_56600_56900.txt')
    ]

Next, we initialize a number of :py:class:`~dtaianomaly.preprocessing.Preprocessor`s. Below, we create 4
preprocessors to analyze the effect of Z-normalization combined with smoothing. We also add the
:py:class:`~dtaianomaly.preprocessing.Identity` preprocessor, to analyze what happens if no preprocessing
is applied.

.. code-block:: python

    preprocessors = [
        Identity(),
        ZNormalizer(),
        ChainedPreprocessor([MovingAverage(10), ZNormalizer()]),
        ChainedPreprocessor([ExponentialMovingAverage(0.8), ZNormalizer()])
    ]

We will now initialize our anomaly detectors. Each anomaly detector will be combined with each
preprocessor, and applied to each time series.

.. code-block:: python

    detectors = [LocalOutlierFactor(50), IsolationForest(50)]


Finally, we need to define the :py:class:`~dtaianomaly.evaluation.Metric`s used to evaluate the models.
Both :py:class:`~dtaianomaly.evaluation.BinaryMetric` and :py:class:`~dtaianomaly.evaluation.ProbaMetric`
can be provided. However, the workflow evaluates the scores obtained by the :py:func:`~dtaianomaly.anomaly_detection.BaseDetector.predict_proba`.
To evaluate a :py:class:`~dtaianomaly.evaluation.BinaryMetric`, a number of thresholding strategies must
be provided to convert the continuous anomaly probabilities to discrete anomaly labels. Each thresholding
strategy is combined with each thresholding metric. The thresholds have no effect on the
:py:class:`~dtaianomaly.evaluation.ProbaMetric` objects.

.. note::
    To save on computational resources, the anomaly detector is used once to detect anomalies in a time
    series, and the predicted anomaly scores are used to evaluate all anomaly scores. This means that
    there is no computational overhead on providing more metrics, besides the resources required to
    compute the metric.

.. code-block:: python

    thresholds = [TopN(20), FixedCutoff(0.1)]
    metrics = [Precision(), AreaUnderPR(), AreaUnderROC()]


Once all components are defined, we initialize the :py:class:`~dtaianomaly.workflow.Workflow`. We also
define additional parameters, such ``n_jobs``, to allow for multiple anomaly detectors to detect anomalies
in parallel. Then, we can execute the workflow by calling the :py:func:`~dtaianomaly.workflow.Workflow.run`
method, which returns a dataframe with the results.

.. code-block:: python

    workflow = Workflow(
        dataloaders=dataloaders,
        metrics=metrics,
        thresholds=thresholds,
        preprocessors=preprocessors,
        detectors=detectors,
        n_jobs=4
    )
    results = workflow.run()



.. _with-config:

Run a workflow from a configuration file
----------------------------------------

Alternatively, you can define a workflow using JSON configuration files. The file
`Config.json`_ illustrates how the workflow defined above can be written as a
configuration file. More details regarding the syntax are provided below. Using the
:py:func:`~dtaianomaly.workflow.workflow_from_config` method, you can pass the path
to a configuration file to create the corresponding :py:class:`~dtaianomaly.workflow.Workflow`,
as shown in the example below. Then, you can run the :py:class:`~dtaianomaly.workflow.Workflow`
via the :py:func:`~dtaianomaly.workflow.Workflow.run` function.

.. code-block:: python

   from dtaianomaly.workflow import workflow_from_config
   workflow = workflow_from_config("Config.json")
   workflow.run()

A configuration file is build from different entries, with each entry representing a
component of the :py:class:`~dtaianomaly.workflow.Workflow`. These entries are build
as follows:
.. code-block:: json

    { 'type': <name-of-component>, 'optional-param': <value-optional-parameter>}

The ``'type'`` equals the name of the component, for example ``'LocalOutlierFactor'``
or ``'ZNormalizer'``. This string must exactly match the object name of the component
you want to add to the workflow. In addition, it is possible to define hyperparameters
of each component. For example for ``'LocalOutlierFactor'``, you must define a
``'window_size'``, but can optionally also define a ``'stride'``. An error will be
raised if the entry has missing obligated parameters or unknown parameters.

The configuration file itself is also a dictionary, in JSON format. The keys of this
dictionary correspond to the parameters of the :py:class:`~dtaianomaly.workflow.Workflow`.
The corresponding values can be either a single entry (if one component is requested)
or a list of entries (if multiple components are requested).

Below, we show a simplified version of the configuration in `Config.json`_.

.. literalinclude:: ../../notebooks/Config.json
   :language: json
   :tab-width: 4

.. _Config.json: https://github.com/ML-KULeuven/dtaianomaly/blob/main/notebooks/Config.json

