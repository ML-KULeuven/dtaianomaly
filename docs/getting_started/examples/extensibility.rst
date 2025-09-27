:orphan:

.. testsetup::

   import numpy as np
   from typing import Optional, Tuple

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
(2) :ref:`neural anomaly detector <custom-neural-anomaly-detector>`,
(3) :ref:`dataloader <custom-dataloader>`,
(4) :ref:`preprocessor <custom-preprocessor>`,
(5) :ref:`thresholding <custom-thresholding>`, and
(6) :ref:`evaluation <custom-evaluation>`.

.. _custom-anomaly-detector:

Custom anomaly detector
-----------------------

The core functionality of ``dtaianomaly`` - time series anomaly detection - is extended
by implementing the :py:class:`~dtaianomaly.anomaly_detection.BaseDetector`. To achieve
this, you need to implement the :py:func:`~dtaianomaly.anomaly_detection.BaseDetector._fit()`,
and :py:func:`~dtaianomaly.anomaly_detection.BaseDetector._decision_function()`
methods. Below, we implement an anomaly detector that detects anomalies when the distance
between an observation and the mean value exceeds a specified number of standard deviations
(also known as the `3-sigma rule <https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule>`_).
The methods have the following functionality:

1. :py:func:`~dtaianomaly.anomaly_detection.BaseDetector._fit()`: learn the mean and standard
   deviation of the training data. These values are stored in the attributes ``mean_`` and ``std_``.
2. :py:func:`~dtaianomaly.anomaly_detection.BaseDetector._decision_function()`: compute the values
   that have distance larger than ``nb_sigmas`` times the learned standard deviation from the learned
   mean. These values are considered anomalies.

.. doctest::

    >>> from dtaianomaly.anomaly_detection import BaseDetector, Supervision
    >>>
    >>> class NbSigmaAnomalyDetector(BaseDetector):
    ...     nb_sigmas: float
    ...     mean_: float
    ...     std_: float
    ...
    ...     def __init__(self, nb_sigmas: float = 3.0):
    ...         super().__init__(Supervision.UNSUPERVISED)
    ...         self.nb_sigmas = nb_sigmas
    ...
    ...     def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'NbSigmaAnomalyDetector':
    ...         """ Compute the mean and standard deviation of the given time series. """
    ...         self.mean_ = X.mean()
    ...         self.std_ = X.std()
    ...         return self
    ...
    ...     def _decision_function(self, X: np.ndarray) -> np.ndarray:
    ...         """ Compute which values are too far from the mean. """
    ...         return np.abs(X - self.mean_) > self.nb_sigmas * self.std_
    >>>
    >>> detector = NbSigmaAnomalyDetector()

.. _custom-neural-anomaly-detector:

Custom neural anomaly detector
------------------------------

While above API also allows to implement neural methods, ``dtaianomaly`` offers
several approaches to simplify this process. Specifically, you can implement one
of the following classes, depending on how you want your neural net to detect
anomalies:

1. :py:class:`~dtaianomaly.anomaly_detection.BaseNeuralForecastingDetector`: detect
   anomalies by forecasting the data, and measuring the difference between the predicted
   values and the actual observations. An example is :py:class:`~dtaianomaly.anomaly_detection.MultilayerPerceptron`.
2. :py:class:`~dtaianomaly.anomaly_detection.BaseNeuralReconstructionDetector`: reconstruct
   windows of the data, and the instances that are more difficult to reconstruct were not
   seen in the data, and thus anomalies. An example is :py:class:`~dtaianomaly.anomaly_detection.AutoEncoder`.

Whatever strategy you choose, you only need to implement the :py:class:`~dtaianomaly.anomaly_detection.BaseNeuralDetector._build_architecture()`
function. This function receives the input dimension of the time series, and returns
the architecture of your neural network as a ``torch.nn.Module``.

Below code shows a very simple example of this: detect anomalies using a perceptron. We will
train a perceptron to forecast the data and then measure the deviation, hence we will extend
the :py:class:`~dtaianomaly.anomaly_detection.BaseNeuralForecastingDetector` class. Specifically,
given a time series with :math:`D` attributes and a window size of :math:`w`, the input is a
flattened :math:`(D \cdot w)`-array. If we want to forecast :math:`h` values in the future
(i.e., the parameter ``forecast_length``), then the output of the perceptron is a
:math:`(D \cdot h)`-array. The implementation is given below:

.. doctest::

    >>> import torch
    >>> from dtaianomaly.anomaly_detection import BaseNeuralForecastingDetector
    >>>
    >>> class Perceptron(BaseNeuralForecastingDetector):
    ...
    ...    def _build_architecture(self, n_attributes: int) -> torch.nn.Module:
    ...        return torch.nn.Linear(
    ...            in_features=n_attributes * self.window_size_,
    ...           out_features=n_attributes * self.forecast_length
    ...        )
    >>>
    >>> perceptron = Perceptron(window_size=16, forecast_length=1)

If you want more flexibility over your network, you can also directly implement
:py:class:`~dtaianomaly.anomaly_detection.BaseNeuralDetector`, in which you must
also implement the creation of a ``torch.utils.data.DataSet`` and the evaluation
and training on a single batch. It is also possible to further customize the training
process by overwriting some of the already implemented methods, or by extending the
:py:class:`~dtaianomaly.anomaly_detection.BaseDetector` and implement your network
from scratch!


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

.. doctest::

    >>> from dtaianomaly.data import LazyDataLoader, DataSet
    >>> import numpy as np
    >>>
    >>> class SimpleDataLoader(LazyDataLoader):
    ...     def _load(self) -> DataSet:
    ...         return DataSet(np.random.uniform(size=1000), np.random.choice([0, 1], p=(0.9, 0.1), size=1000))
    >>>
    >>> data_loader = SimpleDataLoader()

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

.. doctest::

    >>> from dtaianomaly.preprocessing import Preprocessor
    >>>
    >>> class Imputer(Preprocessor):
    ...     fill_value_: float
    ...
    ...     def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'Preprocessor':
    ...         self.fill_value_ = np.nanmean(X, axis=0)
    ...         return self
    ...
    ...     def _transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ...         X[np.isnan(X)] = self.fill_value_
    ...         return X, y
    >>>
    >>> imputer = Imputer()

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

.. doctest::

    >>> from dtaianomaly.thresholding import Thresholding
    >>>
    >>> class DynamicThreshold(Thresholding):
    ...     factor: float
    ...
    ...     def __init__(self, factor: float):
    ...         self.factor = factor
    ...
    ...     def _threshold(self, scores: np.ndarray) -> np.ndarray:
    ...         threshold = scores.mean() + self.factor * scores.std()
    ...         return scores > threshold
    >>>
    >>> dynamic_threshold = DynamicThreshold(1.0)

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

.. doctest::

    >>> from dtaianomaly.evaluation import BinaryMetric
    >>>
    >>> class Accuracy(BinaryMetric):
    ...     def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    ...         """ Compute the accuracy. """
    ...         return np.nanmean(y_true == y_pred)
    >>>
    >>> accuracy = Accuracy()
