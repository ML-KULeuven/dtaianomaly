import numpy as np

from dtaianomaly.anomaly_detection import BaseDetector
from dtaianomaly.preprocessing import ChainedPreprocessor, Preprocessor
from dtaianomaly.type_validation import ObjectAttribute

__all__ = ["Pipeline"]


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

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import IsolationForest
    >>> from dtaianomaly.data import demonstration_time_series
    >>> from dtaianomaly.pipeline import Pipeline
    >>> from dtaianomaly.preprocessing import StandardScaler
    >>> X, y = demonstration_time_series()
    >>> pipeline = Pipeline(StandardScaler(), IsolationForest(16))
    >>> pipeline.fit(X).decision_function(X)
    array([-0.01080726, -0.01053199, -0.00883758, ..., -0.05298726,
           -0.05898066, -0.05713733])
    """

    preprocessor: Preprocessor
    detector: BaseDetector

    attribute_validation = {
        "preprocessor": ObjectAttribute(Preprocessor),
        "detector": ObjectAttribute(BaseDetector),
    }

    def __init__(
        self,
        preprocessor: Preprocessor | list[Preprocessor],
        detector: BaseDetector,
    ):
        self.preprocessor = (
            ChainedPreprocessor(preprocessor)
            if isinstance(preprocessor, list)
            else preprocessor
        )
        self.detector = detector
        super().__init__(detector.supervision)

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        X, y = self.preprocessor.fit_transform(X=X, y=y)
        self.detector.fit(X=X, y=y, **kwargs)

    def _decision_function(self, X: np.ndarray) -> np.array:
        X, _ = self.preprocessor.transform(X=X, y=None)
        return self.detector.decision_function(X)

    def piped_str(self) -> str:
        return f"{self.preprocessor.piped_str()}->{self.detector}"
