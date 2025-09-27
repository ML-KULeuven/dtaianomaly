from dtaianomaly.anomaly_detection import BaseDetector
from dtaianomaly.data import LazyDataLoader
from dtaianomaly.pipeline import Pipeline
from dtaianomaly.preprocessing import Identity, Preprocessor
from dtaianomaly.type_validation import (
    AttributeValidationMixin,
    BoolAttribute,
    NoneAttribute,
    ObjectAttribute,
)

__all__ = ["Job"]


class Job(AttributeValidationMixin):
    """
    A job to execute within the JobBasedWorkflow.

    A single job to be executed within a workflow. A job contains (1) a data loader
    to indicate which data to use, (2) an optional preprocessor to process the data
    in some way, and (3) a detector to find the anomalies.

    Parameters
    ----------
    dataloader : LazyDataLoader
        The data loader that should be used for loading the time series data.
    detector : BaseDetector
        The anomaly detector to use for detecting anomalies, after the time series
        has been preprocessed.
    preprocessor : Preprocessor or None
        The preprocessor to use for processing the time series, before the anomalies
        are detected. If no preprocessor is given (``preprocessor=None``), then the
        time series does not need to be processed.

    Attributes
    ----------
    pipeline : Pipeline
        The pipeline which combines the preprocessor and the anomaly detector, such
        that the anomalies can be detected with a single call. If no preprocessor was
        given, the preprocessor of the pipeline equals ``Identity()``.
    has_preprocessor : bool
        Whether this job has a preprocessor, i.e., whether ``preprocessor == None``.
    """

    dataloader: LazyDataLoader
    preprocessor: Preprocessor | None
    detector: BaseDetector
    pipeline: Pipeline
    has_preprocessor: bool

    attribute_validation = {
        "dataloader": ObjectAttribute(LazyDataLoader),
        "preprocessor": ObjectAttribute(Preprocessor) | NoneAttribute(),
        "detector": ObjectAttribute(BaseDetector),
        "pipeline": ObjectAttribute(Pipeline),
        "has_preprocessor": BoolAttribute(),
    }

    def __init__(
        self,
        dataloader: LazyDataLoader,
        detector: BaseDetector,
        preprocessor: Preprocessor = None,
    ):
        self.dataloader = dataloader
        self.preprocessor = preprocessor
        self.detector = detector
        self.pipeline = Pipeline(preprocessor or Identity(), detector)
        self.has_preprocessor = preprocessor is not None
