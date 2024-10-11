
import multiprocessing
import time
import tracemalloc
import pandas as pd
from typing import Dict, List, Union
from functools import partial

from dtaianomaly.data import LazyDataLoader
from dtaianomaly.evaluation import Metric, BinaryMetric
from dtaianomaly.thresholding import Thresholding
from dtaianomaly.preprocessing import Preprocessor, Identity
from dtaianomaly.anomaly_detection import BaseDetector
from dtaianomaly.pipeline import EvaluationPipeline

from dtaianomaly.workflow.utils import build_pipelines, convert_to_proba_metrics, convert_to_list
from dtaianomaly.workflow.error_logging import log_error


class Workflow:
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
        None or an empty list, then no preprocssing will be done, aka.
        using :py:class:`dtaianomaly.preprocessing.Preprocessor` as the
        preprocessor for each pipeline.

    n_jobs: int, default=1
        Number of processes to run in parallel while evaluating all
        combinations.

    trace_memory: bool, default=False
        Whether or not memory usage of each run is reported. While this
        might give additional insights into the models, their runtime
        will be higher due to additional internal bookkeeping.

    error_log_path: str, default='./error_logs'
        The path in which the error logs should be saved.
    """
    dataloaders: List[LazyDataLoader]
    pipelines: List[EvaluationPipeline]
    provided_preprocessors: bool
    n_jobs: int
    trace_memory: bool
    error_log_path: str
    
    def __init__(self,
                 dataloaders: Union[LazyDataLoader, List[LazyDataLoader]],
                 metrics: Union[Metric, List[Metric]],
                 detectors: Union[BaseDetector, List[BaseDetector]],
                 preprocessors: Union[Preprocessor, List[Preprocessor]] = None,
                 thresholds: Union[Thresholding, List[Thresholding]] = None,
                 n_jobs: int = 1,
                 trace_memory: bool = False,
                 error_log_path: str = './error_logs'):

        # Make sure the inputs are lists.
        dataloaders = convert_to_list(dataloaders)
        metrics = convert_to_list(metrics)
        thresholds = convert_to_list(thresholds or [])
        preprocessors = convert_to_list(preprocessors or [])
        self.provided_preprocessors = len(preprocessors) > 0
        if not self.provided_preprocessors:
            preprocessors = [Identity()]
        detectors = convert_to_list(detectors)

        # Add thresholding to the binary metrics
        if len(thresholds) == 0 and any(isinstance(metric, BinaryMetric) for metric in metrics):
            raise ValueError('There should be at least one thresholding option if a binary metric is passed!')
        proba_metrics = convert_to_proba_metrics(
            metrics=metrics,
            thresholds=thresholds
        )

        # Perform checks on input
        if len(dataloaders) == 0:
            raise ValueError('At least one data loader should be given to the workflow!')
        if len(metrics) == 0:
            raise ValueError('At least one metrics should be given to the workflow!')
        if len(detectors) == 0:
            raise ValueError('At least one detectors should be given to the workflow!')
        if n_jobs < 1:
            raise ValueError('There should be at least one job within a workflow!')

        # Set the properties of this workflow
        self.pipelines = build_pipelines(
            preprocessors=preprocessors,
            detectors=detectors,
            metrics=proba_metrics
        )
        self.dataloaders = dataloaders
        self.n_jobs = n_jobs
        self.trace_memory = trace_memory
        self.error_log_path = error_log_path

    def run(self) -> pd.DataFrame:
        """
        Run the experimental workflow. Evaluate each pipeline within this
        workflow on each dataset within this workflow in a grid-like manner.

        Returns
        -------
        results: pd.DataFrame
            A pandas dataframe with the results of this workflow. Each row
            represents an execution of an anomaly detector on a given dataset
            with some preprocessing steps. The columns correspond to the
            different evaluation metrics, running time and potentially also
            the memory usage.
        """
        # Create all the jobs
        unit_jobs = [
            (dataloader, pipeline)
            for dataloader in self.dataloaders
            for pipeline in self.pipelines
        ]

        # Execute the jobs
        if self.n_jobs == 1:
            result = [_single_job(*job, trace_memory=self.trace_memory, error_log_path=self.error_log_path) for job in unit_jobs]
        else:
            single_run_function = partial(_single_job, trace_memory=self.trace_memory, error_log_path=self.error_log_path)
            with multiprocessing.Pool(processes=self.n_jobs) as pool:
                result = pool.starmap(single_run_function, unit_jobs)

        # Create a dataframe of the results
        results_df = pd.DataFrame(result)

        # Reorder the columns
        columns = ['Dataset', 'Detector', 'Preprocessor', 'Runtime [s]']
        if self.trace_memory:
            columns.append('Peak Memory [MB]')
        results_df = results_df[columns + [x for x in results_df.columns if x not in columns]]

        # Drop the processors column, if none were provided.
        if not self.provided_preprocessors:
            results_df.drop(columns='Preprocessor', inplace=True)

        # Return the results
        return results_df


def _single_job(dataloader: LazyDataLoader, pipeline: EvaluationPipeline, trace_memory: bool, error_log_path: str) -> Dict[str, Union[str, float]]:

    # Initialize the results, and by default everything went wrong ('Error')
    results = {'Dataset': str(dataloader)}
    for key in pipeline.metrics + ['Detector', 'Preprocessor', 'Runtime [s]']:
        results[str(key)] = 'Error'
    if trace_memory:
        results['Peak Memory [MB]'] = 'Error'

    # Try to load the data set, if this fails, return the results
    try:
        dataset = dataloader.load()
    except Exception as exception:
        results['Error file'] = log_error(error_log_path, exception, dataloader)
        return results

    # We can already save the used preprocessor and detector
    results['Preprocessor'] = str(pipeline.pipeline.preprocessor)
    results['Detector'] = str(pipeline.pipeline.detector)

    # Start tracing the memory, if requested
    if trace_memory:
        tracemalloc.start()

    # Evaluate the pipeline, and measure the time
    start = time.time()
    try:
        results.update(pipeline.run(X=dataset.x, y=dataset.y))
    except Exception as exception:
        results['Error file'] = log_error(error_log_path, exception, dataloader, pipeline.pipeline)
    stop = time.time()

    # Save the runtime
    results['Runtime [s]'] = stop - start

    # Save the memory if requested, and stop tracing
    if trace_memory:
        _, peak = tracemalloc.get_traced_memory()
        results['Peak Memory [MB]'] = peak / 10**6
        tracemalloc.stop()

    # Return the results
    return results
