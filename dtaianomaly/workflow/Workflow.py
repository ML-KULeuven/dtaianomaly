from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector
from dtaianomaly.data.LazyDataLoader import LazyDataLoader
from dtaianomaly.evaluation.metrics import Metric
from dtaianomaly.preprocessing.Preprocessor import Preprocessor
from dtaianomaly.thresholding.thresholding import Thresholding
from dtaianomaly.workflow.JobBasedWorkflow import Job, JobBasedWorkflow
from dtaianomaly.workflow.utils import convert_to_list


class Workflow(JobBasedWorkflow):
    """
    Run anomaly detection experiments

    Run all combinations of ``dataloaders``, ``preprocessors``, ``detectors``,
    and ``metrics``. The metrics requiring a thresholding operation are
    combined with every element of ``thresholds``. If an error occurs in any
    execution of an anomaly detector or loading of data, then the error will
    be written to an error file, which is an executable Python file to reproduce
    the error.

    Parameters
    ----------
    dataloaders: LazyDataLoader or list of LazyDataLoader
        The dataloaders that will be used to load data, and consequently
        this data is used for evaluation within this workflow.

    metrics: Metric or list of Metric
        The metrics to evaluate within this workflow.

    detectors: BaseDetector or list of BaseDetector
        The anomaly detectors to evaluate.

    thresholds: Thresholding or list of Thresholding, default=None
        The thresholds used for converting continuous anomaly scores to
        binary anomaly predictions. Each threshold will be combined with
        each :py:class:`~dtaianomaly.evaluation.BinaryMetric` given via
        the ``metrics`` parameter. The thresholds do not apply on a
        :py:class:`~dtaianomaly.evaluation.ProbaMetric`. If equals None
        or an empty list, then all the given metrics via the ``metrics``
        argument must be of type :py:class:`~dtaianomaly.evaluation.ProbaMetric`.
        Otherwise, a ValueError will be raised.

    preprocessors: Preprocessor or list of Preprocessor, default=None
        The preprocessors to apply before evaluating the model. If equals
        None or an empty list, then no preprocessing will be done, aka.
        using :py:class:`dtaianomaly.preprocessing.Preprocessor` as the
        preprocessor for each pipeline.

    n_jobs: int, default=1
        Number of processes to run in parallel while evaluating all
        combinations.

    trace_memory: bool, default=False
        Whether or not memory usage of each run is reported. While this
        might give additional insights into the models, their runtime
        will be higher due to additional internal bookkeeping.

    anomaly_scores_path: str, default=None
        The path where the anomaly scores should be saved. If ``None``, the
        anomaly scores will not be saved.

    error_log_path: str, default='./error_logs'
        The path in which the error logs should be saved.

    fit_unsupervised_on_test_data: bool, default=False
        Whether to fit the unsupervised anomaly detectors on the test data.
        If True, then the test data will be used to fit the detector and
        to evaluate the detector. This is no issue, since unsupervised
        detectors do not use labels and can deal with anomalies in the
        training data.

    fit_semi_supervised_on_test_data: bool, default=False
        Whether to fit the semi-supervised anomaly detectors on the test data.
        If True, then the test data will be used to fit the detector and
        to evaluate the detector. This is not really an issue, because it only
        breaks the assumption of semi-supervised methods of normal training data.
        However, these methods do not use the training labels themselves.

    show_progress: bool, default=False
        Whether to show the progress using a TQDM progress bar or not.

        .. note::

           Ensure ``tqdm`` installed for this (which is not part of the core
           dependencies of ``dtaianomaly``). Otherwise, no progress bar will
           be shown.

    Attributes
    ----------
    jobs: list of Job
        The jobs to execute within this workflow.
    """

    dataloaders: list[LazyDataLoader]
    preprocessors: list[Preprocessor]
    detectors: list[BaseDetector]

    def __init__(
        self,
        dataloaders: LazyDataLoader | list[LazyDataLoader],
        metrics: Metric | list[Metric],
        detectors: BaseDetector | list[BaseDetector],
        preprocessors: Preprocessor | list[Preprocessor] = None,
        thresholds: Thresholding | list[Thresholding] = None,
        n_jobs: int = 1,
        trace_memory: bool = False,
        anomaly_scores_path: str = None,
        error_log_path: str = "./error_logs",
        fit_unsupervised_on_test_data: bool = False,
        fit_semi_supervised_on_test_data: bool = False,
        show_progress: bool = False,
    ):

        # Make sure the inputs are lists.
        dataloaders = convert_to_list(dataloaders)
        preprocessors = convert_to_list(preprocessors or [])
        detectors = convert_to_list(detectors)

        # Initialize the jobs
        if len(preprocessors) == 0:
            jobs = [
                Job(dataloader, None, detector)
                for dataloader in dataloaders
                for detector in detectors
            ]
        else:
            jobs = [
                Job(dataloader, preprocessor, detector)
                for dataloader in dataloaders
                for preprocessor in preprocessors
                for detector in detectors
            ]

        super().__init__(
            jobs=jobs,
            metrics=metrics,
            thresholds=thresholds,
            n_jobs=n_jobs,
            trace_memory=trace_memory,
            anomaly_scores_path=anomaly_scores_path,
            error_log_path=error_log_path,
            fit_unsupervised_on_test_data=fit_unsupervised_on_test_data,
            fit_semi_supervised_on_test_data=fit_semi_supervised_on_test_data,
            show_progress=show_progress,
        )

        # Set the properties of this workflow
        self.dataloaders = dataloaders
        self.preprocessors = preprocessors
        self.detectors = detectors
