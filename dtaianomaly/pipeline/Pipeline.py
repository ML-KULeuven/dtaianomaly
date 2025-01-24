import numpy as np
from typing import List, Optional, Union

from dtaianomaly.utils import is_valid_list
from dtaianomaly.preprocessing import Preprocessor, ChainedPreprocessor
from dtaianomaly.anomaly_detection import BaseDetector


class Pipeline(BaseDetector):
    """
    Pipeline to combine preprocessing and anomaly detection

    The pipeline works with a single :py:class:`~dtaianomaly.preprocessing.Preprocessor` object or a
    list of :py:class:`~dtaianomaly.preprocessing.Preprocessor` objects. This list is converted into a
    :py:class:`~dtaianomaly.preprocessing.ChainedPreprocessor`. At the moment the `Pipeline` always
    requires a `Preprocessor` object passed at construction. If
    no preprocessing is desired, you need to explicitly pass an
    :py:class:`~dtaianomaly.preprocessing.Identity` preprocessor.

    Parameters
    ----------
    preprocessor: Preprocessor or list of Preprocessors
        The preprocessors to include in this pipeline.
    detector: BaseDetector
        The anomaly detector to include in this pipeline.
    """

    preprocessor: Preprocessor
    detector: BaseDetector

    def __init__(
        self,
        preprocessor: Union[Preprocessor, List[Preprocessor]],
        detector: BaseDetector,
    ):
        if not (
            isinstance(preprocessor, Preprocessor)
            or is_valid_list(preprocessor, Preprocessor)
        ):
            raise TypeError(
                "preprocessor expects a Preprocessor object or list of Preprocessors"
            )
        if not isinstance(detector, BaseDetector):
            raise TypeError("detector expects a BaseDetector object")
        super().__init__(detector.supervision)

        if isinstance(preprocessor, list):
            self.preprocessor = ChainedPreprocessor(preprocessor)
        else:
            self.preprocessor = preprocessor
        self.detector = detector

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Pipeline":
        """
        Fit this pipeline to the given data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.
        y: array-like of shape (n_samples)
            The ground truth labels, passed to the preprocessor and detector.

        Returns
        -------
        self: Pipeline
            Returns the instance itself
        """
        X, y = self.preprocessor.fit_transform(X=X, y=y)
        self.detector.fit(X=X, y=y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw anomaly scores.

        Note that depending on the preprocessor this output might
        not have the same shape as the input.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Raw time series

        Returns
        -------
        anomaly_scores: array-like of shape (n_samples)
            The predicted anomaly scores
        """
        X, _ = self.preprocessor.transform(X=X, y=None)
        return self.detector.decision_function(X)

    def __str__(self) -> str:
        return f"{self.preprocessor}->{self.detector}"
