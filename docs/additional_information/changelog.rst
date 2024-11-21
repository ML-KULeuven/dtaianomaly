Changelog
=========

All notable changes to this project will be documented in this file.

Latest
------

Added
^^^^^
- Added option to automatically compute the window size for various anomaly detectors
  using Fourier Transform, Autocorrelation, Multi-Window Finder, or Summary Statistics Subsequence.
- Implemented ``KNearestNeighbor`` anomaly detector.
- Implemented point-adjusted precision, recall and FBeta.
- Implemented ``BestThresholdMetric``, a ``ProbaMetric`` which computes the best value for
  a ``BinaryMetric`` over all thresholds.
- The property ``supervision`` to ``BaseDetector``, which indicates what type of supervision
  the anomaly detector requires. Possible options are:

  - ``Supervision.SUPERVISED``: the anomaly detector requires training data and labels
  - ``Supervision.SEMI_SUPERVISED``: the anomaly detector requires training data, but no
    training labels because the training data is assumed to be normal.
  - ``Supervision.UNSUPERVISED``: the anomaly detector does not require any training data
    or labels
- Added the property ``__version__`` to ``dtaianomaly``, which can be accessed from code.
- Included the used version of ``dtaianomaly`` when logging errors.

Changed
^^^^^^^
- Updated documentation to contain changelog and contributing information
- Rely on PyOD for non-time series anomaly detectors (instead of scikit-learn before)
- Separated training data and test data in ``DataSet``. This has also been integrated
  within the ``Workflow`` to use the correct data. To this end, a new flag has been
  added to the ``Workflow``, which decides if the training data or the test data
  should be used for training *unsupervised* anomaly detectors.

Fixed
^^^^^

[0.2.2] - 2024-10-30
--------------------

Added
^^^^^
- When executing a workflow, and an error occurs. The errors will be written to
  an error file. This file contains in which phase the error occurred and the
  entire traceback of the error. Additionally, the error file contains the code
  to reproduce the specific error. In fact, the error file can be run as any
  python script.
- Added baseline anomaly detectors: ``AlwaysNormal``, ``AlwaysAnomaly``, and
  ``RandomDetector``
- Added option ``novelty`` to ``MatrixProfileDetector``, which will compute the
  matrix profile in regard to the train data, if ``novelty=True``. By default,
  the matrix profile is computed based on a self-join of the test data.
- Implemented ``MedianMethod`` anomaly detector.
- Implemented ``Differencing`` preprocessor.
- Implemented ``PiecewiseAggregateApproximation`` preprocessor.

Changed
^^^^^^^
- Added the option to cache data in ``LazyDataLoader`` via parameter ``do_caching``.
  The ``load`` function in ``LazyDataLoader`` is adjusted to either load the data and
  potentially cache the data, or return a cached version of the data. As a consequence,
  the children of ``LazyDataLoader`` must implement the ``_load`` method (instead of
  the ``load()`` method), which will effectively load the data, independent of any cache.

Fixed
^^^^^
- ``utils.is_valid_array_like()`` could not handle multivariate lists. This functionality
  has now been added, and the tests are extended accordingly.
- Preprocessors can now take lists as input, which are automatically converted to a numpy
  array by the ``fit()`` and ``transform()`` method in ``Preprocessor``.

[0.2.1] - 2024-10-08
--------------------

In this release, all files were transferred from GitLab to GitHub. Therefore,
older links may no longer function as desired.

Added
^^^^^

Changed
^^^^^^^
- The ``__str__`` method of the different estimators are automatically done
  and now equal the name of the class and between parantheses the parameters
  that must be provided, i.e., the parameters that differntiate from the
  default parameters.

Fixed
^^^^^


[0.2.0] - 2024-10-01
--------------------

This release basically revamped the ``dtaianomaly``. In many ways, the package
has been simplified, while also ensuring its robustness. In general, the core
functionality remained similar, while the API might have slightly changed.
Below we mention the most notable changes.

Added
^^^^^
- A ``LazyDataLoader`` has been added, which can be used to read datasets from
  disk. This offers a simpler interface compared to the old ``DataManager``.
- A ``Pipeline`` has been added to easily combine time series anomaly detection
  with preprocessing the time series
- The Preprocessing module has been added, which includes a wide range of
  methods to preprocess a time series before detecting anomalies. Specifically,
  the implemented preprocessors are ``ExponentialMovingAverage``, ``MinMaxScaler``,
  ``MovingAverage``, ``SamplingRateUnderSampler``, ``NbSamplesUnderSampler``, and
  ``Znormalizer``. In addition, there is also a ``ChainedPreprocessor`` which
  combines multiple preprocessors.
- A ``Workflow`` object, which takes a set of dataloader, anomaly detectors,
  preprocessors and evaluation metrics and evaluates in a grid-like manner
  each anomaly detector in combination with each preprocessor on each dataset.
  As before, a workflow can be instantiated from a configuration file, but now
  it is also possible to start a workflow from Python itself, meaning that the
  Python scripts can serve as configuration files.
- More tests have been added to ensure ``dtaianomaly`` functions well and to
  guarantee that changes do not alter the existing functionality.

Changed
^^^^^^^
- The base anomaly detector has been renamed from ``TimeSeriesAnomalyDetector``
  to ``BaseDetector``. Additionally, the wrapper approaches to existing libraries
  for anomaly detection have been removed, as these rely on the active maintenance
  of said packages (specifically, ``PyODAnomalyDetector`` and ``TSBUADAnomalyDetecotor``
  have been removed).
- The evaluation module has been cleaned up to only contain well-established
  evaluation metrics. In the future, we plan on adding more performance metrics.
- The thresholding methods have been moved from the evaluation module into their
  own module: thresholding.
- The notebooks have been cleaned up to only show the core functionality to the
  users, making it easier to understand ``dtaianomaly``.

Fixed
^^^^^
- The visualization methods were relatively buggy. Most of the methods have been
  removed from this module, because it is simple to plot time series and the anomaly
  scores using ``plt.plot(X)`` and ``plt.plot(y)``. Only one method remained to
  plot a time series and color it according to the anomaly scores, as this is not
  trivial to do with just ``matplotlib``.
- The documentation has improved, including the API descriptions, but especially the
  getting started pages.

[0.1.4] - 2024-02-28
--------------------

This release mostly increased the amount of functionality, but also offers some
more quality-of-life features.

Added
^^^^^
- TSB-UAD has been integrated, thus increasing the amount of available algorithms.
- Options to read the results from a workflow and visualize them.
- An option has been added to log errors in the workflow, without letting
  the entire workflow crash and stop.
- Anomaly detector `STOMP` (based on the Matrix Profile) has been added.
- An option to include a specific stride when windowing the time series using the
  `Windowing` class has been added.

Changed
^^^^^^^
- Due to the dependency on TSB-UAD, this version (and likely also future versions)
  won't be available on PyPi anymore, because TSB-UAD is installed from source, which
  means is not supported through PyPi
- Changed how the algorithm configuration works.

  - you can provide multiple algorithms in one configuration to facilitate large
    scale experiments in which multiple algorithms are compared.
  - An option was implemented to provide template configurations, and then
    fill in the templates given a number of possible values in a grid-like
    fashion. this allows to more easily tune various parameters of anomaly
    detectors.
- The number of features in the `DataManager` are reduced such that only a limitted
  set of important features remain.

Fixed
^^^^^
- Some bugs related to visualizing the data have been fixed.
- There was a problem with using custom algorithms in the workflow, due to an
  unknown path.
- Added the opportunity to perform anomaly detection in parallel over multiple
  time series, thus reducing the total required running time.

[0.1.3] - 2023-11-07
--------------------

There was another, similar bug.

Added
^^^^^

Changed
^^^^^^^

Fixed
^^^^^
- Also added a `__init__.py` file in the utility directory for the affiliation metrics.

[0.1.2] - 2023-11-07
--------------------

This update is to fix a crucial but small bug.

Added
^^^^^
- The documentation has been extended (though far from finalized).

Changed
^^^^^^^

Fixed
^^^^^
- The `__init__.py` files in the `anomaly_detection` module were updated
  to properly import classes that are not directly in the `anomaly_detection`,
  but rather in a sub folder.

[0.1.1] - 2023-10-26
--------------------

This update doesn't include a lot of changes. It only slightly modified the
readme.

Added
^^^^^
- Added an official release to the repository, and a badge to indicate
  the latest release.

Changed
^^^^^^^

Fixed
^^^^^
- Fixed the link to the image showcasing the anomaly scores of an
  IForest on a Demo time series.

[0.1.0] - 2023-10-26
--------------------

First release of `dtaianomaly`! While our toolbox is still a work in progress,
we believe it is already in a usable stage. Additionally, by publicly releasing
`dtaianomaly`, we hope to receive feedback from the community! Be sure to check
out the [documentation](https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/)
for additional information!

Added
^^^^^
- `anomaly_detection`: a module for time series anomaly detection algorithms.
   Currently, basic algorithms using[PyOD](https://github.com/yzhao062/pyod)
   are included, but we plan to extend on this in the future!
- `data_management`: a module to easily handle datasets. You can filter the datasets on
   certain properties and add new datasets through a few simple function calls! More
   information can be found in the [Documentation](https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/getting_started/data_management.html).
- `evaluation`: It is crucial to evaluate an anomaly detector in order to quantify its
   performance. This module offers several metrics to this end. `dtaianomaly` offers
   traditional metrics such as precision, recall, and F1-score, but also more recent
   metrics that were tailored for time series anomaly detection such as the
   [Affiliation Score](https://dl.acm.org/doi/10.1145/3534678.3539339)
   [notebooks](notebooks) and [Volume under the surface (VUS)](https://dl.acm.org/doi/10.14778/3551793.3551830)
- `visualization`: This module allows to easily visualize the data and anomalies, as
   time series and anomalies inherently are great for visual inspection.
- `workflow`: This module allows to benchmark an algorithm on a larger set of datasets,
   through configuration files. This methodology ensures reproducibility by simply providing
   the configuration files!

Changed
^^^^^^^

Fixed
^^^^^
