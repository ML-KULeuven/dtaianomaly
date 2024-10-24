Contributing to dtaianomaly
===========================

The goal of ``dtaianomaly`` is to be community-driven. All types of contributions
are welcome. This includes code, bug reports, improvements to the documentation,
additional tests. Below we give an overview of how to contribute to ``dtaianomaly``.

All contributions to ``dtaianomaly`` are welcome. We use `GitHub Issues <https://github.com/ML-KULeuven/dtaianomaly/issues>`_
to track bugs and feature requests. Feel free to open a new issue if you found a
bug or wish to see a new feature in ``dtaianomaly``. Please verify that the issue is
not already addressed by another issue or pull request before submitting a new issue.

You can follow below steps to make a contribution to ``dtaianomaly``:

#. Fork ``dtaianomaly``.

#. Clone your fork of ``dtaianomaly`` locally and create a new branch for your feature:

    .. code-block:: bash

     git checkout -b feature-or-bugfix-name

#. Create and activate a `virtual environment <https://docs.python.org/3/library/venv.html>`_
   and install all dependencies (including the development dependencies):

    .. code-block:: bash

     pip install -r requirements.txt
     pip install -r requirements-dev.txt

#. Implement your contribution.

#. Check if all tests run successfully:

    .. code-block:: bash

       pytest .\tests\ --cov=dtaianomaly --cov-report term-missing

#. Verify if the documentation builds correctly:

    .. code-block:: bash

       docs/make html
       docs/make doctest

#. Commit your changes and push to your fork.

#. Create a pull request for your fork.

#. If this is your first contribution, make sure you added your name to `CONTRIBUTORS <https://github.com/ML-KULeuven/dtaianomaly/blob/main/CONTRIBUTORS>`_.

It is highly recommended to follow below checklist if you are implementing a new
:py:class:`~dtaianomaly.anomaly_detection.BaseDetector`,
:py:class:`~dtaianomaly.data.LazyDataLoader`,
:py:class:`~dtaianomaly.preprocessing.Preprocessor`,
:py:class:`~dtaianomaly.thresholding.Thresholding`, or
:py:class:`~dtaianomaly.evaluation.Metric`.
This makes sure that the new component has been fully integrated in ``dtaianomaly``.

.. |check_box| raw:: html

    <input type="checkbox">

**Implement the component**
|   |check_box| Have you added a ``.py`` to the correct module, named identical to the component you implemented?
|   |check_box| Does the file contain a class named as the methodology, which inherits from the correct base class?
|   |check_box| Does the class documentation contain an explanation of the methodology?
|   |check_box| Are all hyperparameters and attributes discussed in the class documentation, including their meaning, type and hyperparameters?
|   |check_box| Has a reference been added to the class documentation?
|   |check_box| Does the documentation contain a code-example?
|   |check_box| Does the constructor call the parent constructor (``super().__init__(args)``)?
|   |check_box| Are all hyperparameters checked to be of the correct type and belong to the domain?
|   |check_box| Are all hyperparameters set as an attribute of the object (necessary for ``__str__()`` method)?
|   |check_box| Are all required methods correctly implemented?
|   |check_box| Did you add the component in ``__all__`` of the ``__init__.py`` file?
|   |check_box| Can you load the component via :py:func:`~dtaianomaly.workflow.interpret_config``?

**Test the component**
|   |check_box| Have you added a new file ``test_<class>.py`` in the correct directory under ``tests/``?
|   |check_box| Is a test coverage of at least 95% reached?
|   |check_box| *[Only applicable for anomaly detectors]* Has the method been included in the tests in ``tests/anomaly_detection/test_detectors.py``?
|   |check_box| *[Only applicable for preprocessors]* Has the method been included in the tests in ``tests/preprocessing/test_preprocessors.py``?
|   |check_box| *[Only applicable for evaluation metrics]* Has the method been included in the tests in ``tests/evaluation/test_metrics.py``?
|   |check_box| Have you tested loading the new object in ``tests/workflow/test_workflow_from_config.py``?
|   |check_box| Do all tests still succeed?

**Document the component**
|   |check_box| Has the documentation been updated?
|   |check_box| *[Only applicable for anomaly detectors]* Is a separate file for the anomaly detector created in ``docs/api/anomaly_detection_algorithms/`` with the same name as the anomaly detector, and has the file been included to the index in ``docs/api/anomaly_detection``?
|   |check_box| Does the documentation build correctly?

**Adding data**
|   |check_box| Did you update `data/README.rst <https://github.com/ML-KULeuven/dtaianomaly/blob/main/data/README.rst>`_?
