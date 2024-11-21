
import numpy as np
from typing import List, Dict, Union, Optional

from dtaianomaly.utils import is_valid_list, is_valid_array_like
from dtaianomaly.preprocessing.Preprocessor import Preprocessor
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector
from dtaianomaly.evaluation.metrics import ProbaMetric
from dtaianomaly.pipeline.Pipeline import Pipeline


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
    metrics: list of Probametric objects
        The evaluation metrics to compute in this evaluation pipeline.
    """
    pipeline: Pipeline
    metrics: List[ProbaMetric]

    def __init__(self,
                 preprocessor: Union[Preprocessor, List[Preprocessor]],
                 detector: BaseDetector,
                 metrics: Union[ProbaMetric, List[ProbaMetric]]):
        if not (isinstance(metrics, ProbaMetric) or is_valid_list(metrics, ProbaMetric)):
            raise TypeError("metrics should be a list of ProbaMetric objects")
        self.pipeline = Pipeline(preprocessor=preprocessor, detector=detector)
        self.metrics = metrics if isinstance(metrics, list) else [metrics]

    def run(self,
            X_test: np.ndarray,
            y_test: np.ndarray,
            X_train: np.ndarray,
            y_train: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Run the pipeline and evaluate performance. The pipeline will
        be trained on the given train data (potentially without labels)
        and performance will be estimated on the test data.

        Parameters
        ----------
        X_test: array-like of shape (n_samples_test, n_attributes)
            The test time series data.
        y_test: array-like of shape (n_samples_test)
            The ground truth anomaly labels of the test data.
        X_train: array-like of shape (n_samples_train, n_attributes)
            The train time series data.
        y_train: array-like of shape (n_samples_train) or ``None``.
            The ground truth anomaly labels of the train data. Note that, even
            though ``y_train`` can be ``None``, it must be provided.

        Returns
        -------
        performances: Dict[str, float]
            The evaluation of the performance metrics. The keys are
            string descriptors of the performance metrics, with values
            the corresponding performance score.
        """
        # Validate the input
        if not is_valid_array_like(X_test):
            raise ValueError("X_test is not a valid array-like!")
        if not is_valid_array_like(y_test):
            raise ValueError("y_test is not a valid array-like!")
        if not is_valid_array_like(X_train):
            raise ValueError("X_train is not a valid array-like!")
        if not (y_train is  None or is_valid_array_like(y_train)):
            raise ValueError("y_train is not a valid array-like!")

        # Fit on the train data
        self.pipeline.fit(X_train, y_train)

        # Predict on the test data
        y_pred = self.pipeline.predict_proba(X=X_test)

        # Transform the test labels
        _, y_test_ = self.pipeline.preprocessor.transform(X_test, y_test)

        # Compute the performances
        return {
            str(metric): metric.compute(y_true=y_test_, y_pred=y_pred)
            for metric in self.metrics
        }
