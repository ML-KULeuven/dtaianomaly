import multiprocessing
import time
import tracemalloc
import warnings
from functools import partial
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.data.DataSet import DataSet
from dtaianomaly.data.LazyDataLoader import LazyDataLoader
from dtaianomaly.evaluation.metrics import BinaryMetric, Metric
from dtaianomaly.pipeline.EvaluationPipeline import EvaluationPipeline
from dtaianomaly.preprocessing.Preprocessor import Identity, Preprocessor
from dtaianomaly.thresholding.thresholding import Thresholding
from dtaianomaly.workflow.error_logging import log_error
from dtaianomaly.workflow.utils import (
    build_pipelines,
    convert_to_list,
    convert_to_proba_metrics,
)


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
    """

    dataloaders: List[LazyDataLoader]
    pipelines: List[EvaluationPipeline]
    provided_preprocessors: bool
    n_jobs: int
    trace_memory: bool
    error_log_path: str
    fit_unsupervised_on_test_data: bool
    fit_semi_supervised_on_test_data: bool
    show_progress: bool

    def __init__(
        self,
        dataloaders: Union[LazyDataLoader, List[LazyDataLoader]],
        metrics: Union[Metric, List[Metric]],
        detectors: Union[BaseDetector, List[BaseDetector]],
        preprocessors: Union[Preprocessor, List[Preprocessor]] = None,
        thresholds: Union[Thresholding, List[Thresholding]] = None,
        n_jobs: int = 1,
        trace_memory: bool = False,
        error_log_path: str = "./error_logs",
        fit_unsupervised_on_test_data: bool = False,
        fit_semi_supervised_on_test_data: bool = False,
        show_progress: bool = False,
    ):

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
        if len(thresholds) == 0 and any(
            isinstance(metric, BinaryMetric) for metric in metrics
        ):
            raise ValueError(
                "There should be at least one thresholding option if a binary metric is passed!"
            )
        proba_metrics = convert_to_proba_metrics(metrics=metrics, thresholds=thresholds)

        # Perform checks on input
        if len(dataloaders) == 0:
            raise ValueError(
                "At least one data loader should be given to the workflow!"
            )
        if len(metrics) == 0:
            raise ValueError("At least one metrics should be given to the workflow!")
        if len(detectors) == 0:
            raise ValueError("At least one detectors should be given to the workflow!")
        if n_jobs < 1:
            raise ValueError("There should be at least one job within a workflow!")

        # Set the properties of this workflow
        self.pipelines = build_pipelines(
            preprocessors=preprocessors, detectors=detectors, metrics=proba_metrics
        )
        self.dataloaders = dataloaders
        self.n_jobs = n_jobs
        self.trace_memory = trace_memory
        self.error_log_path = error_log_path
        self.fit_unsupervised_on_test_data = fit_unsupervised_on_test_data
        self.fit_semi_supervised_on_test_data = fit_semi_supervised_on_test_data
        self.show_progress = show_progress

    def run(self, **kwargs) -> pd.DataFrame:
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

        if self.show_progress:
            try:
                import tqdm
            except ModuleNotFoundError:
                warnings.warn(
                    "Flag 'tqdm_progress' was set to True in the workflow, but tqdm is not installed!\n"
                    "No progress will be shown using tqdm. To do so, run 'pip install tqdm'!"
                )
                self.show_progress = False

        # Execute the jobs
        if self.n_jobs == 1:
            if self.show_progress:
                import tqdm

                unit_jobs = tqdm.tqdm(unit_jobs)

            result = [
                _single_job(
                    *job,
                    trace_memory=self.trace_memory,
                    error_log_path=self.error_log_path,
                    fit_unsupervised_on_test_data=self.fit_unsupervised_on_test_data,
                    fit_semi_supervised_on_test_data=self.fit_semi_supervised_on_test_data,
                    **kwargs,
                )
                for job in unit_jobs
            ]

        else:
            single_run_function = partial(
                _single_job,
                trace_memory=self.trace_memory,
                error_log_path=self.error_log_path,
                fit_unsupervised_on_test_data=self.fit_unsupervised_on_test_data,
                fit_semi_supervised_on_test_data=self.fit_semi_supervised_on_test_data,
                **kwargs,
            )
            if self.show_progress:
                import tqdm

                # Run jobs with tqdm progress bar
                with multiprocessing.Pool(processes=self.n_jobs) as pool:
                    with tqdm.tqdm(total=len(unit_jobs)) as pbar:
                        result = [
                            pool.apply_async(
                                single_run_function,
                                args=job,
                                callback=lambda _: pbar.update(1),
                            )
                            for job in unit_jobs
                        ]
                        pool.close()
                        pool.join()  # Wait for all processes to complete

                result = [r.get() for r in result]

            else:
                with multiprocessing.Pool(processes=self.n_jobs) as pool:
                    result = pool.starmap(single_run_function, unit_jobs)

        # Create a dataframe of the results
        results_df = pd.DataFrame(result)

        # Reorder the columns
        columns = [
            "Dataset",
            "Detector",
            "Preprocessor",
            "Runtime Fit [s]",
            "Runtime Predict [s]",
            "Runtime [s]",
        ]
        if self.trace_memory:
            columns.extend(
                ["Peak Memory Fit [MB]", "Peak Memory Predict [MB]", "Peak Memory [MB]"]
            )
        results_df = results_df[
            columns + [x for x in results_df.columns if x not in columns]
        ]

        # Drop the processors column, if none were provided.
        if not self.provided_preprocessors:
            results_df.drop(columns="Preprocessor", inplace=True)

        # Return the results
        return results_df


def _single_job(
    dataloader: LazyDataLoader,
    pipeline: EvaluationPipeline,
    trace_memory: bool,
    error_log_path: str,
    fit_unsupervised_on_test_data: bool,
    fit_semi_supervised_on_test_data: bool,
    **kwargs,
) -> Dict[str, Union[str, float]]:
    # Initialize the results, and by default everything went wrong ('Error')
    results = {"Dataset": str(dataloader)}
    for key in pipeline.metrics + [
        "Detector",
        "Preprocessor",
        "Runtime Fit [s]",
        "Runtime Predict [s]",
        "Runtime [s]",
    ]:
        results[str(key)] = "Error"
    if trace_memory:
        for key in [
            "Peak Memory Fit [MB]",
            "Peak Memory Predict [MB]",
            "Peak Memory [MB]",
        ]:
            results[key] = "Error"

    # Try to load the data set, if this fails, return the results
    try:
        data_set = dataloader.load()
    except Exception as exception:
        results["Error file"] = log_error(error_log_path, exception, dataloader)
        return results

    # We can already save the used preprocessor and detector
    results["Preprocessor"] = str(pipeline.pipeline.preprocessor)
    results["Detector"] = str(pipeline.pipeline.detector)

    # Check if the dataset and the anomaly detector are compatible
    if not data_set.is_compatible(pipeline.pipeline):
        error_message = (
            f"Not compatible: detector with supervision {pipeline.pipeline.supervision} "
            f"for data set with compatible supervision ["
        )
        error_message += ", ".join([str(s) for s in data_set.compatible_supervision()])
        error_message += "]"
        for key, value in results.items():
            if value == "Error":
                results[key] = error_message
        return results

    # Format X_train, y_train, X_test and y_test
    X_test, y_test, X_train, y_train, fit_on_X_train = _get_train_test_data(
        data_set,
        pipeline.pipeline,
        fit_unsupervised_on_test_data,
        fit_semi_supervised_on_test_data,
    )

    # Run the anomaly detector, and catch any exceptions
    try:
        # Fitting
        _start_tracing_memory(trace_memory)
        start = _start_tracing_runtime()
        pipeline.fit(X_train, y_train, **kwargs)
        results["Runtime Fit [s]"] = _end_tracing_runtime(start)
        _end_tracing_memory(trace_memory, results, "Peak Memory Fit [MB]")

        # Predicting
        _start_tracing_memory(trace_memory)
        start = _start_tracing_runtime()
        y_pred = pipeline.predict(X_test)
        results["Runtime Predict [s]"] = _end_tracing_runtime(start)
        _end_tracing_memory(trace_memory, results, "Peak Memory Predict [MB]")

        # Scoring
        y_test_ = pipeline.format_y_test(X_test, y_test)
        results.update(pipeline.evaluate(y_test_, y_pred))

        # Aggregate the used resources
        results["Runtime [s]"] = (
            results["Runtime Fit [s]"] + results["Runtime Predict [s]"]
        )
        if trace_memory:
            results["Peak Memory [MB]"] = max(
                results["Peak Memory Fit [MB]"], results["Peak Memory Predict [MB]"]
            )

    except Exception as exception:
        # Log the errors
        results["Error file"] = log_error(
            error_log_path,
            exception,
            dataloader,
            pipeline.pipeline,
            fit_on_X_train,
            **kwargs,
        )

    # Return the results
    return results


def _start_tracing_runtime() -> float:
    return time.time()


def _end_tracing_runtime(start_time: float) -> float:
    return time.time() - start_time


def _start_tracing_memory(trace_memory: bool) -> None:
    if trace_memory:
        tracemalloc.start()


def _end_tracing_memory(trace_memory: bool, results, key) -> None:
    if trace_memory:
        _, peak = tracemalloc.get_traced_memory()
        results[key] = peak / 10**6
        tracemalloc.stop()


def _get_train_test_data(
    data_set: DataSet,
    detector: BaseDetector,
    fit_unsupervised_on_test_data: bool,
    fit_semi_supervised_on_test_data: bool,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool):
    """
    Separates the train and test data depending on the type of the anomaly
    detector and whether the test data should be used for fitting in an
    unsupervised detector.

    Also returns a bool indicating if the train data is actually used for
    fitting or not.
    """
    X_test = data_set.X_test
    y_test = data_set.y_test
    X_train = data_set.X_train
    y_train = data_set.y_train

    fit_on_X_train = True

    # If no train data is given but the detector is unsupervised, then use the test data for training
    # This is only ok if the detector is unsupervised, because no labels are used
    # If this happens, the train labels will be None anyway (otherwise data_set would be invalid)
    if detector.supervision == Supervision.UNSUPERVISED and X_train is None:
        X_train = X_test
        fit_on_X_train = False

    # If unsupervised detectors should fit on the test data.
    if (
        fit_unsupervised_on_test_data
        and detector.supervision == Supervision.UNSUPERVISED
    ):
        X_train = X_test
        fit_on_X_train = False

    # If semi-supervised detectors should fit on the test data.
    if (
        fit_semi_supervised_on_test_data
        and detector.supervision == Supervision.SEMI_SUPERVISED
    ):
        X_train = X_test
        fit_on_X_train = False

    return X_test, y_test, X_train, y_train, fit_on_X_train
