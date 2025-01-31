:orphan:

Extensibility
=============

Even though ``dtaianomaly`` already offers a lot of functionality, there is always room
for more enhancements. ``dtaianomaly`` is designed with flexibility in mind: it is
extremely easy to integrate a new component in ``dtaianomaly``. These new components
can be either existing methods that haven't been implemented yet, or new state-of-the-art
time series anomaly detection methods. By implementing your new component in ``dtaianomaly``, y
ou can seamlessly use the existing tools - such as the :py:class:`~dtaianomaly.pipeline.Pipeline`
and :py:class:`~dtaianomaly.workflow.Workflow` - as if it were a native part of ``dtaianomaly``.

Below, we illustrate how you can implement your own
(1) :ref:`anomaly detector <custom-anomaly-detector>`,
(2) :ref:`dataloader <custom-dataloader>`,
(3) :ref:`preprocessor <custom-preprocessor>`,
(4) :ref:`thresholding <custom-thresholding>`, and
(5) :ref:`evaluation <custom-evaluation>`.

.. _custom-anomaly-detector:

Custom anomaly detector
-----------------------

The core functionality of ``dtaianomaly`` - time series anomaly detection - is extended
by implementing the :py:class:`~dtaianomaly.anomaly_detection.BaseDetector`. To achieve
this, you need to implement the :py:func:`~dtaianomaly.anomaly_detection.BaseDetector.fit()`,
and :py:func:`~dtaianomaly.anomaly_detection.BaseDetector.decision_function()`
methods. Below, we implement an anomaly detector that detects anomalies when the distance
between an observation and the mean value exceeds a specified number of standard deviations
(also known as the `3-sigma rule <https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule>`_.
The methods have the following functionality:

1. :py:func:`~dtaianomaly.anomaly_detection.BaseDetector.fit()`: learn the mean and standard
   deviation of the training data. These values are stored in the attributes ``mean_`` and ``std_``.
2. :py:func:`~dtaianomaly.anomaly_detection.BaseDetector.decision_function()`: compute the values
   that have distance larger than ``nb_sigmas`` times the learned standard deviation from the learned
   mean. These values are considered anomalies.

.. code-block:: python

    from dtaianomaly.anomaly_detection import BaseDetector

    class NbSigmaAnomalyDetector(BaseDetector):
        nb_sigmas: float
        mean_: float
        std_: float

        def __init__(self, nb_sigmas: float = 3.0):
            self.nb_sigmas = nb_sigmas

        def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'NbSigmaAnomalyDetector':
            """ Compute the mean and standard deviation of the given time series. """
            self.mean_ = X.mean()
            self.std_ = X.std()
            return self

        def decision_function(self, X: np.ndarray) -> np.ndarray:
            """ Compute which values are too far from the mean. """
            return np.abs(X - self.mean_) > self.nb_sigmas * self.std_

.. _custom-dataloader:

Custom data loader
------------------

Some dataloaders are provided within ``dtaianomaly``, but often we want to detect anomalies
in our own data. Typically, for such custom data, there is no dataloader available within
``dtaianomaly``. To address this, you can implement a new dataloader by extending the
:py:class:`~dtaianomaly.data.LazyDataLoader`, along with the :py:func:`~dtaianomaly.data.LazyDataLoader._load`
method. Upon initialization of the custom data loader, a ``path`` parameter is required,
which points to the location of the data. Optionally, you can pass a ``do_caching`` parameter
to prevent reading big files multiple times. The :py:func:`~dtaianomaly.data.LazyDataLoader._load`
function will effectively load this dataset and return a :py:class:`~dtaianomaly.data.DataSet`
object, which combines the data ``X`` and ground truth labels ``y``. The :py:func:`~dtaianomaly.data.LazyDataLoader.load`
function will either load the data or return a cached version of the data, depending on the
``do_caching`` property.

Implementing a custom dataloader is especially useful for quantitatively evaluating the anomaly
detectors on your own data, as you can pass the loader to a :py:class:`~dtaianomaly.workflow.Workflow`
and easily analyze multiple detectors simultaneously.

.. code-block:: python

    from dtaianomaly.data import LazyDataLoader, DataSet

    class SimpleDataLoader(LazyDataLoader):

        def _load(self)-> DataSet:
            """ Read a data frame with the data in column 'X' and the labels in column 'y'. """
            df = pd.read_clipboard(self.path)
            return DataSet(df['X'].values, df['y'].values)

.. _custom-preprocessor:

Custom preprocessor
-------------------

The preprocessors will perform some processing on the time series, after which the transformed
time series can be used for anomaly detection. Below, we implement a custom preprocessor by
extending the :py:class:`~dtaianomaly.preprocessing.Preprocessor` class. Our preprocessor
replaces all missing values (i.e., the NaN values) with the mean of the training data.
Specifically, we need to implement following methods:

1. :py:func:`~dtaianomaly.preprocessing.Preprocessor._fit`: learns the mean value of the given
   time series and stores it as the ``fill_value_`` attribute.
2. :py:func:`~dtaianomaly.preprocessing.Preprocessor._transform`: fills in all missing values
   with the given time series by the learned mean value. This method returns both a transformed
   ``X`` and ``y``, because some preprocessors also change the labels ``y`` (for example, the
   :py:class:`~dtaianomaly.preprocessing.SamplingRateUnderSampler`).

Notice that we implement the :py:func:`~dtaianomaly.preprocessing.Preprocessor._fit` and
:py:func:`~dtaianomaly.preprocessing.Preprocessor._transform` methods (with a starting underscore),
while we can call the :py:func:`~dtaianomaly.preprocessing.Preprocessor.fit` and
:py:func:`~dtaianomaly.preprocessing.Preprocessor.transform` methods (without the underscore) on
an instance of our ``Imputer``. This is because the public methods will first check if the input
is valid using the :py:func:`~dtaianomaly.preprocessing.check_preprocessing_inputs` method, and
only then call the protected methods with starting underscores, ensuring that valid data is passed
to these methods.

.. code-block:: python

    from dtaianomaly.preprocessing import Preprocessor

    class Imputer(Preprocessor):
        fill_value_: float

        def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'Preprocessor':
            self.fill_value_ = np.nanmean(X, axis=0)
            return self

        def _transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            X[np.isnan(X)] = self.fill_value_
            return X, y

.. _custom-thresholding:

Custom thresholding
-------------------

Many anomaly detectors compute continuous anomaly scores ("how *anomalous* is the sample?), while
many practical applications prefer binary labels ("is the sample *an anomaly*?"). Converting the
continuous scores to binary labels can be done via thresholding. The most common thresholding
strategies have already been implemented in ``dtaianomaly``, but is possible to add a new
thresholding technique, as we do below. For this, we extend the :py:class:`~dtaianomaly.thresholding.Thresholding`
object and implement the ``threshold`` method. Our custom thresholding technique sets a dynamic
threshold, such that observations with an anomaly score larger than a specified number of standard
deviations above the mean anomaly score are considered anomalous.

.. code-block:: python

    from dtaianomaly.thresholding import Thresholding

    class DynamicThreshold(Thresholding):
        factor: float

        def __init__(self, factor: float):
            self.factor = factor

        def threshold(self, scores: np.ndarray) -> np.ndarray:
            threshold = scores.mean() + self.factor * scores.std()
            return scores > threshold

.. _custom-evaluation:

Custom evaluation
-----------------

Various performance metrics exist to evaluate an anomaly detector. There are two types
of metrics in ``dtaianomaly``:

1. :py:class:`~dtaianomaly.evaluation.BinaryMetric`: the provided anomaly scores must be binary
   anomaly labels. An example of such metric is the precision.
2. :py:class:`~dtaianomaly.evaluation.ProbaMetric`:: the provided anomaly scores are expected to
   be continuous scores. An example of such metric is the area under the ROC curve (AUC-ROC).

Custom evaluation metrics can be implemented in ``dtaianomaly``. Below, we implement accuracy
by extending the :py:class:`~dtaianomaly.evaluation.BinaryMetric` class (since accuracy requires
binary labels) and implementing the :py:func:`~dtaianomaly.evaluation.Metric._compute` method.
Similar to the custom preprocessor above,we implement the :py:func:`~dtaianomaly.evaluation.Metric._compute`
method with starting underscore, while we call the :py:func:`~dtaianomaly.evaluation.Metric.compute`
method to measure the metric. This is because the public :py:func:`~dtaianomaly.evaluation.Metric.compute`
method performs checks on the input, ensuring that valid data is passed to the :py:func:`~dtaianomaly.evaluation.Metric._compute`
method.

.. warning::
    Anomaly detection is typically a highly unbalanced problem: anomalies are, by definition,
    rare. Therefore, it is not recommended to use accuracy for evaluation (time series) anomaly
    detection!

.. code-block:: python

    from dtaianomaly.evaluation import BinaryMetric

    class Accuracy(BinaryMetric):

        def _compute(self, y_true: np.ndarray, y_pred: np.ndarray):
            """ Compute the accuracy. """
            return np.nanmean(y_true == y_pred)
