.. |check_box| raw:: html

    <input type="checkbox">

Contributing to dtaianomaly
===========================

The goal of ``dtaianomaly`` is to be community-driven. All types of contributions
are welcome. This includes code, but also bug reports, improvements to the documentation,
additional tests and more. Below we give an overview of how to contribute to ``dtaianomaly``.

Reporting issues
----------------

We use `GitHub Issues <https://github.com/ML-KULeuven/dtaianomaly/issues>`_
to track bugs and feature requests. Feel free to open a new issue if you found a
bug or wish to see a new feature in ``dtaianomaly``. Please verify that the issue is
not already addressed by another issue or pull request before submitting a new issue.

Resolving issues
----------------

Alternatively, you can also resolve an open issue. Below, we describe the required
steps for this.

Find your contribution
^^^^^^^^^^^^^^^^^^^^^^

The first step is to find an issue you wish to resolve on `GitHub Issues <https://github.com/ML-KULeuven/dtaianomaly/issues>`_.
These are feature requests, bug reports, or anything else that can improve
``dtaianomaly``. If you find an issue that seems interesting to you, you
can click on it to see its current status. You can then either join the
ongoing discussion on this issue, or start the discussion if you only see
the initial issue. In this case, you can leave a message with your proposed
solution. We will analyze your solution and review alternative solutions
together with you, to find the best way to fix the issue. Once we decided
on the best fix, the issue will be assigned to you, after which you will
be able to implement your solution.

.. rubric:: Checklist

| |check_box| Find an issue you wish to work on.
| |check_box| Join the discussion with your proposed solution.

Create a fork
^^^^^^^^^^^^^^

To start working on your issue, you need to `create a fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_
of ``dtaianomaly``, on which you can work to resolve the issue.

Next, you have to clone your fork to your own machine, as follows:

.. code-block:: bash

   git clone https://github.com/<link-to-your-fork>

Next, you should create a new branch in your fork. You can find more
information about git branches on `this webpage <https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging>`_.
In the end, you will merge this branch into ``dtaianomaly`` such that
everyone can benefit from your contribution. You can create a branch
with the following command:

.. code-block:: bash

     git checkout -b <branch-name>

Here you should replace <branch-name> by something descriptive to
make clear what you are working on. For example, if you are implementing
a new anomaly detector ``MyAmazingAnomalyDetector``, you can name your
branch ``implement_my_amazing_anomaly_detector``.

.. rubric:: Checklist

| |check_box| Create a fork of ``dtaianomaly`` on your own account.
| |check_box| Clone the fork to your local machine.
| |check_box| Create a branch for you to work on.

Setup the environment
^^^^^^^^^^^^^^^^^^^^^

Next, you can set up your environment to start working on the issue.
For this, we highly recommend to `virtual environment <https://docs.python.org/3/library/venv.html>`_
to isolate the dependencies. To install the core-dependencies (e.g.,
`Numpy <https://numpy.org/>`_) of``dtaianomaly``, run the following
command:

.. code-block:: bash

     pip install -r requirements.txt

In addition, you should also install the development dependencies
(e.g., `pytest <https://docs.pytest.org/en/stable/>`_) as follows:

.. code-block:: bash

     pip install -r requirements-dev.txt

To check if the environment is correct, you verify if all tests
succeed by running the following command (which also checks the
coverage of the unit tests):

.. code-block:: bash

   pytest .\tests\ --cov=dtaianomaly --cov-report term-missing

In addition, you should also check if the documentation generates
without any errors or warnings. This can be done as follows:

.. code-block:: bash

   docs/make html
   docs/make doctest

.. rubric:: Checklist

| |check_box| Install the dependencies in ``requirements.txt``.
| |check_box| Install the dependencies in ``requirements-dev.txt``.
| |check_box| Check if all tests run successfully.
| |check_box| Check if the documentation generates successfully.

Resolve the issue
^^^^^^^^^^^^^^^^^

Once everything has been set up, it is time to resolve the issue.
This can include writing code, fixing bugs, writing documentation, ...
depending on the issue you selected.

If this is your first contribution, also make sure you added your
name to `CONTRIBUTORS <https://github.com/ML-KULeuven/dtaianomaly/blob/main/CONTRIBUTORS>`_.


Be sure to go through the :ref:`checklist below <new_component_checklist>`
if your issue involves implementing a new
:py:class:`~dtaianomaly.anomaly_detection.BaseDetector`,
:py:class:`~dtaianomaly.data.LazyDataLoader`,
:py:class:`~dtaianomaly.preprocessing.Preprocessor`,
:py:class:`~dtaianomaly.thresholding.Thresholding`, or
:py:class:`~dtaianomaly.evaluation.Metric`.

Once you have resolved the issue, you commit the changes to your
remote fork via:

.. code-block:: bash

   git add .
   git commit -m <commit-message>
   git push

.. rubric:: Checklist

| |check_box| Add the resolution to the issue.
| |check_box| Add your update to the changelog
| |check_box| Check if all tests still run successfully.
| |check_box| Check if the documentation still generates successfully.
| |check_box| Commit your changes to your fork.

Synchronize your fork
^^^^^^^^^^^^^^^^^^^^^

Since you started working on the issue, it is likely that new changes
have been added to ``dtaianomaly``. Some of these changes might conflict
with your resolution of the issue. Therefore, it is necessary to first
sync your fork with ``dtaianomaly`` and pull all the changes in your
personal branch. If this does not lead to merge conflicts: great! Otherwise,
you will have to fix the conflicts and make sure the unit tests still
successfully run.

.. rubric:: Checklist

| |check_box| Synchronize your project with ``dtaianomaly``
| |check_box| Fix the merge conflicts, if there are any.
| |check_box| Check if the documentation still generates successfully.
| |check_box| Commit your changes to your fork.

Create a pull request
^^^^^^^^^^^^^^^^^^^^^

Now that you have resolved the issue and made sure your fork is
up to date with ``dtaianomaly``, you can create a pull request!
For this, you can go to the GitHub page of your fork, on which
there should be a button to automatically create a pull request.
Otherwise, you will have to manually create a pull request.

Make sure to add a descriptive title to your pull request. Also
add a description of the issue you tackled and how exactly you
solved it. Also add a reference to the issue you solved in the
body of the pull request.

Creating a pull request will automatically run various checks,
such as running the unit tests, doctests, and checking if
certain notebooks remain executable. These checks must succeed
before a pull request can be accepted.

.. rubric:: Checklist

| |check_box| Create a pull request.
| |check_box| Add an informative description to the pull request.
| |check_box| Add the issue number to the pull request.

Work on your pull request
^^^^^^^^^^^^^^^^^^^^^^^^^

We will likely have some questions, suggestions or comments on your solution.
This is our opportunity to collaborate and further improve the resolution.
If we see a further improvement to your solution, you can simply continue
working on the same branch you have been working on.

.. rubric:: Checklist

| |check_box| Engage in the discussion on your pull request.
| |check_box| Add the suggestions given in the documentation.

Merge!
^^^^^^

Once your contribution has been finalized and polished, we will
merge your pull request into ``dtaianomaly``! Thanks for your
contribution!

.. rubric:: Checklist

| |check_box| Celebrate your successful addition to ``dtaianomaly``!


.. _new_component_checklist:

Checklist for implementing new components
-----------------------------------------

It is highly recommended to follow below checklist if you are implementing a new
:py:class:`~dtaianomaly.anomaly_detection.BaseDetector`,
:py:class:`~dtaianomaly.data.LazyDataLoader`,
:py:class:`~dtaianomaly.preprocessing.Preprocessor`,
:py:class:`~dtaianomaly.thresholding.Thresholding`, or
:py:class:`~dtaianomaly.evaluation.Metric`.
This ensures a flawless integration of the new component into ``dtaianomaly``.

BaseDetector
^^^^^^^^^^^^

.. rubric:: Implement the anomaly detector

| |check_box| Have you added a ``.py`` in ``dtaianomaly/anomaly_detection`` named identical to the anomaly detector?
| |check_box| Does the file contain a class named as the anomaly detector, which inherits :py:class:`~dtaianomaly.anomaly_detection.BaseDetector`?
| |check_box| Does the constructor call ``super().__init__(Supervision)`` with the correct supervision type?
| |check_box| Are all hyperparameters checked to be of the correct type and belong to the domain?
| |check_box| Are all hyperparameters set as an attribute of the object (necessary for ``__str__()`` method)?
| |check_box| Have you implemented the :py:func:`~dtaianomaly.anomaly_detection.BaseDetector.fit()` method?
| |check_box| Have you implemented the :py:func:`~dtaianomaly.anomaly_detection.BaseDetector.decision_function()` method?
| |check_box| Did you add the anomaly detector in ``__all__`` of the ``dtaianomaly/anomaly_detection/__init__.py`` file?
| |check_box| Can you load the anomaly detector via :py:func:`~dtaianomaly.workflow.interpret_config`` (specifically, in the ``detector_entry()`` function)?

.. rubric:: Test the anomaly detector

| |check_box| Have you added a new file ``test_<class>.py`` in under ``tests/anomaly_detection``?
| |check_box| Is a test coverage of at least 95% reached?
| |check_box| Have you included the anomaly detector in the tests in ``tests/anomaly_detection/test_detectors.py``?
| |check_box| Have you tested loading the new anomaly detector in ``tests/workflow/test_workflow_from_config.py``?

.. rubric:: Document the anomaly detector

| |check_box| Have you added class documentation to your implementation?
| |check_box| Does the class documentation contain an explanation of the anomaly detector?
| |check_box| Are all hyperparameters and attributes discussed in the class documentation, including their meaning, type and default values?
| |check_box| Does the class documentation contain a code-example?
| |check_box| Has a reference to the relevant paper(s) been added in the class documentation?
| |check_box| Is a separate file for the anomaly detector created in ``docs/api/anomaly_detection_algorithms/`` with the same name as the anomaly detector?

LazyDataLoader
^^^^^^^^^^^^^^

.. rubric:: Implement the data loader

| |check_box| Have you added a ``.py`` in ``dtaianomaly/data`` named identical to the data loader?
| |check_box| Does the file contain a class named as the data loader, which inherits :py:class:`~dtaianomaly.data.LazyDataLoader`?
| |check_box| Are all hyperparameters checked to be of the correct type and belong to the domain?
| |check_box| Are all hyperparameters set as an attribute of the object (necessary for ``__str__()`` method)?
| |check_box| Have you implemented the :py:func:`~dtaianomaly.data.LazyDataLoader._load()` method?
| |check_box| Did you add the data loader in ``__all__`` of the ``dtaianomaly/data/__init__.py`` file?
| |check_box| Can you load the data loader via :py:func:`~dtaianomaly.workflow.interpret_config`` (specifically, in the ``data_entry()`` function)?

.. rubric:: Test the data loader

| |check_box| Have you added a new file ``test_<class>.py`` in under ``tests/data``?
| |check_box| Is a test coverage of at least 95% reached?
| |check_box| Have you tested loading the new data loader in ``tests/workflow/test_workflow_from_config.py``?

.. rubric:: Document the data loader

| |check_box| Have you added class documentation to your implementation?
| |check_box| Does the class documentation contain an explanation of expected format of the data?
| |check_box| Are all hyperparameters and attributes discussed in the class documentation, including their meaning, type and default values?
| |check_box| Has a reference to the relevant paper(s) been added in the class documentation?
| |check_box| Have you added the data loader to ``docs/api/data.rst``?
| |check_box| Did you update `data/README.rst <https://github.com/ML-KULeuven/dtaianomaly/blob/main/data/README.rst>`_?

Preprocessor
^^^^^^^^^^^^

.. rubric:: Implement the preprocessor

| |check_box| Have you added a ``.py`` in ``dtaianomaly/preprocessing`` named identical to the preprocessor?
| |check_box| Does the file contain a class named as the preprocessor, which inherits :py:class:`~dtaianomaly.preprocessing.Preprocessor`?
| |check_box| Are all hyperparameters checked to be of the correct type and belong to the domain?
| |check_box| Are all hyperparameters set as an attribute of the object (necessary for ``__str__()`` method)?
| |check_box| Have you implemented the :py:func:`~dtaianomaly.preprocessing.Preprocessor._fit()` method?
| |check_box| Have you implemented the :py:func:`~dtaianomaly.preprocessing.Preprocessor._transform()` method?
| |check_box| Did you add the preprocessor in ``__all__`` of the ``dtaianomaly/preprocessing/__init__.py`` file?
| |check_box| Can you load the preprocessor via :py:func:`~dtaianomaly.workflow.interpret_config`` (specifically, in the ``preprocessor_entry()`` function)?

.. rubric:: Test the preprocessor

| |check_box| Have you added a new file ``test_<class>.py`` in under ``tests/preprocessing``?
| |check_box| Is a test coverage of at least 95% reached?
| |check_box| Has the preprocessor been included in the tests in ``tests/preprocessing/test_preprocessors.py``?
| |check_box| Have you tested loading the new preprocessor in ``tests/workflow/test_workflow_from_config.py``?

.. rubric:: Document the preprocessor

| |check_box| Have you added class documentation to your implementation?
| |check_box| Does the class documentation contain an explanation of the preprocessor?
| |check_box| Are all hyperparameters and attributes discussed in the class documentation, including their meaning, type and default values?
| |check_box| Has a reference to the relevant paper(s) been added in the class documentation?
| |check_box| Have you added the preprocessor to ``docs/api/preprocessing.rst``?

Thresholding
^^^^^^^^^^^^

.. rubric:: Implement the thresholder

| |check_box| Have you added a ``.py`` in ``dtaianomaly/thresholding`` named identical to the thresholder?
| |check_box| Does the file contain a class named as the thresholder, which inherits :py:class:`~dtaianomaly.thresholding.Thresholder`?
| |check_box| Are all hyperparameters checked to be of the correct type and belong to the domain?
| |check_box| Are all hyperparameters set as an attribute of the object (necessary for ``__str__()`` method)?
| |check_box| Have you implemented the :py:func:`~dtaianomaly.thresholding.Thresholder.threshold()` method?
| |check_box| Did you add the thresholder in ``__all__`` of the ``dtaianomaly/thresholding/__init__.py`` file?
| |check_box| Can you load the thresholder via :py:func:`~dtaianomaly.workflow.interpret_config`` (specifically, in the `threshold_entry()`` function)?

.. rubric:: Test the thresholder

| |check_box| Have you added a new file ``test_<class>.py`` in under ``tests/thresholding``?
| |check_box| Is a test coverage of at least 95% reached?
| |check_box| Have you tested loading the new thresholder in ``tests/workflow/test_workflow_from_config.py``?

.. rubric:: Document the thresholder

| |check_box| Have you added class documentation to your implementation?
| |check_box| Does the class documentation contain an explanation of the thresholder?
| |check_box| Are all hyperparameters and attributes discussed in the class documentation, including their meaning, type and default values?
| |check_box| Has a reference to the relevant paper(s) been added in the class documentation?
| |check_box| Have you added the thresholder to ``docs/api/thresholding.rst``?

Evaluation Metric
^^^^^^^^^^^^^^^^^

.. rubric:: Implement the evaluation metric

| |check_box| Have you added a ``.py`` in ``dtaianomaly/evaluation`` named identical to the evaluation metric?
| |check_box| Does the file contain a class named as the evaluation metric, which inherits :py:class:`~dtaianomaly.evaluation.BinaryMetric` or :py:class:`~dtaianomaly.evaluation.ProbaMetric`, depending on if the metric accepts binary anomaly labels or continuous anomaly probabilities?
| |check_box| Are all hyperparameters checked to be of the correct type and belong to the domain?
| |check_box| Are all hyperparameters set as an attribute of the object (necessary for ``__str__()`` method)?
| |check_box| Have you implemented the :py:func:`~dtaianomaly.evaluation.Metric._compute()` method?
| |check_box| Did you add the metric in ``__all__`` of the ``dtaianomaly/evaluation/__init__.py`` file?
| |check_box| Can you load the metric via :py:func:`~dtaianomaly.workflow.interpret_config` (specifically, in the `metric_entry()`` function)?

.. rubric:: Test the evaluation metric

| |check_box| Have you added a new file ``test_<class>.py`` in under ``tests/evaluation``?
| |check_box| Is a test coverage of at least 95% reached?
| |check_box| Has the evaluation metric been included in the tests in ``tests/evaluation/test_metrics.py``?
| |check_box| Have you tested loading the new evaluation metric in ``tests/workflow/test_workflow_from_config.py``?

.. rubric:: Document the evaluation metric

| |check_box| Have you added class documentation to your implementation?
| |check_box| Does the class documentation contain an explanation of the evaluation metric?
| |check_box| Are all hyperparameters and attributes discussed in the class documentation, including their meaning, type and default values?
| |check_box| Has a reference to the relevant paper(s) been added in the class documentation?
| |check_box| Have you added the evaluation metric to ``docs/api/evaluation.rst``?
