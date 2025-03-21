from typing import Dict, List, Optional, Union

import numpy as np

from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector
from dtaianomaly.evaluation.metrics import ProbaMetric
from dtaianomaly.pipeline.Pipeline import Pipeline
from dtaianomaly.preprocessing.Preprocessor import Preprocessor
from dtaianomaly.utils import is_valid_list


class EvaluationPipeline:
    """
    Pipeline to combine a base pipeline, and a set of metrics. Used
    in the workflow. The given :py:class:`~dtaianomaly.preprocessing.Preprocessor`
    and :py:class:`~dtaianomaly.anomaly_detection.BaseDetector` are
    combined into a :py:class:`~dtaianomaly.pipeline.Pipeline` object.

    Parameters
    ----------
    preprocessor: Preprocessor or list of Preprocessors
        The preprocessors to include in this evaluation pipeline.
    detector: BaseDetector
        The anomaly detector to include in this evaluation pipeline.
    metrics: list of ProbaMetric objects
        The evaluation metrics to compute in this evaluation pipeline.
    """

    pipeline: Pipeline
    metrics: List[ProbaMetric]

    def __init__(
        self,
        preprocessor: Union[Preprocessor, List[Preprocessor]],
        detector: BaseDetector,
        metrics: Union[ProbaMetric, List[ProbaMetric]],
    ):
        if not (
            isinstance(metrics, ProbaMetric) or is_valid_list(metrics, ProbaMetric)
        ):
            raise TypeError("metrics should be a list of ProbaMetric objects")
        self.pipeline = Pipeline(preprocessor=preprocessor, detector=detector)
        self.metrics = metrics if isinstance(metrics, list) else [metrics]

    def fit(self, X_train: np.ndarray, y_train: Optional[np.array], **kwargs) -> None:
        """
        Apply the fit stage of this evaluation pipeline.

        Parameters
        ----------
        X_train: array-like of shape (n_samples, n_attributes)
            The train time series data.
        y_train: array-like of shape (n_samples) or ``None``.
            The ground truth anomaly labels of the train data. Note that, even
            though ``y_train`` can be ``None``, it must be provided (i.e., there
            is no default value).
        """
        self.pipeline.fit(X_train, y_train, **kwargs)

    def predict(self, X_test: np.ndarray):
        """
        Apply the predict stage of the pipeline.

        Parameters
        ----------
        X_test: array-like of shape (n_samples, n_attributes)
            The test time series data.

        Returns
        -------
        y_pred: array-like of shape (n_samples)
            The predicted anomaly scores.
        """
        return self.pipeline.predict_proba(X=X_test)

    def format_y_test(self, X_test: np.ndarray, y_test: np.array) -> np.array:
        """
        Format the test labels using the preprocessor in this pipeline. This is
        necessary if some preprocessors are used that undersample the data.

        Parameters
        ----------
        X_test: array-like of shape (n_samples, n_attributes)
            The test time series data.
        y_test: array-like of shape (n_samples)
            The ground truth anomaly labels of the test data.

        Returns
        -------
        y_test_: array-like of shape (n_samples\\_)
            The formatted ground truth labels.
        """
        _, y_test_ = self.pipeline.preprocessor.transform(X_test, y_test)
        return y_test_

    def evaluate(self, y_test_: np.array, y_pred: np.array) -> Dict[str, float]:
        """
        Evaluate this pipeline by computing the evaluation scores.

        Parameters
        ----------
        y_test_: array-like of shape (n_samples)
            The formatted ground truth anomaly labels.
        y_pred: array-like of shape (n_samples)
            The predicted anomaly scores.

        Returns
        -------
        performances: Dict[str, float]
            The evaluation of the performance metrics. The keys are
            string descriptors of the performance metrics, with values
            the corresponding performance score.
        """
        return {
            str(metric): metric.compute(y_true=y_test_, y_pred=y_pred)
            for metric in self.metrics
        }
