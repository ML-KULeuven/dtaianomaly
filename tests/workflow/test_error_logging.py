import os
import pathlib
import py_compile
import subprocess
import sys

import numpy as np

import dtaianomaly
from dtaianomaly.anomaly_detection import BaseDetector, IsolationForest, Supervision
from dtaianomaly.data import DataSet, DemonstrationTimeSeriesLoader, LazyDataLoader
from dtaianomaly.evaluation import AreaUnderROC
from dtaianomaly.pipeline import Pipeline
from dtaianomaly.preprocessing import ChainedPreprocessor, Identity, Preprocessor
from dtaianomaly.workflow import Workflow
from dtaianomaly.workflow._error_logging import log_error


class DemonstrationDataLoaderWithTrainData(LazyDataLoader):

    def _load(self) -> DataSet:
        dataset = DemonstrationTimeSeriesLoader().load()
        return DataSet(dataset.X_test, dataset.y_test, X_train=np.zeros(shape=1000))


class ErrorDataLoader(LazyDataLoader):

    def _load(self):
        raise Exception("An error occurred when loading data!")


class ErrorPreprocessor(Preprocessor):

    def _fit(self, X, y=None):
        return self

    def _transform(self, X, y=None):
        raise Exception("An error occurred preprocessing data!")


class ErrorAnomalyDetector(BaseDetector):

    def __init__(self):
        super().__init__(Supervision.UNSUPERVISED)

    def _fit(self, X, y=None, **kwargs):
        pass

    def _decision_function(self, X):
        raise Exception("An error occurred when detecting anomalies!")


class TestErrorLogging:

    def test_error_loading(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=ErrorDataLoader(),
            metrics=AreaUnderROC(),
            preprocessors=ChainedPreprocessor(Identity(), ErrorPreprocessor()),
            detectors=IsolationForest(15),
            error_log_path=str(tmp_path_factory.mktemp("error-log")),
        )
        results = workflow.run()

        assert results.shape == (1, 8)
        assert "Error file" in results.columns

        error_file = results.loc[0, "Error file"]
        error = Exception("An error occurred when loading data!")
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_dtaianomaly_version(
            error_file, dtaianomaly.__version__
        )
        assert error_file_contains_error(error_file, error)
        assert error_file_results_in_error(error_file, error)

    def test_error_preprocessing(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=DemonstrationTimeSeriesLoader(),
            metrics=AreaUnderROC(),
            preprocessors=ErrorPreprocessor(),
            detectors=IsolationForest(15),
            error_log_path=str(tmp_path_factory.mktemp("error-log")),
        )
        results = workflow.run()

        assert results.shape == (1, 8)
        assert "Error file" in results.columns

        error_file = results.loc[0, "Error file"]
        error = Exception("An error occurred preprocessing data!")
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_dtaianomaly_version(
            error_file, dtaianomaly.__version__
        )
        assert error_file_contains_error(error_file, error)
        assert error_file_results_in_error(error_file, error)

    def test_error_chained_preprocessing(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=DemonstrationTimeSeriesLoader(),
            metrics=AreaUnderROC(),
            preprocessors=ErrorPreprocessor(),
            detectors=IsolationForest(15),
            error_log_path=str(tmp_path_factory.mktemp("error-log")),
        )
        results = workflow.run()

        assert results.shape == (1, 8)
        assert "Error file" in results.columns

        error_file = results.loc[0, "Error file"]
        error = Exception("An error occurred preprocessing data!")
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_dtaianomaly_version(
            error_file, dtaianomaly.__version__
        )
        assert error_file_contains_error(error_file, error)
        assert error_file_results_in_error(error_file, error)

    def test_error_detecting_anomalies(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=DemonstrationTimeSeriesLoader(),
            metrics=AreaUnderROC(),
            preprocessors=Identity(),
            detectors=ErrorAnomalyDetector(),
            error_log_path=str(tmp_path_factory.mktemp("error-log")),
        )
        results = workflow.run()

        assert results.shape == (1, 8)
        assert "Error file" in results.columns

        error_file = results.loc[0, "Error file"]
        error = Exception("An error occurred when detecting anomalies!")
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_dtaianomaly_version(
            error_file, dtaianomaly.__version__
        )
        assert error_file_contains_error(error_file, error)
        assert error_file_results_in_error(error_file, error)

    def test_error_train_on_fit_data(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=DemonstrationDataLoaderWithTrainData(),
            metrics=AreaUnderROC(),
            preprocessors=Identity(),
            detectors=ErrorAnomalyDetector(),
            error_log_path=str(tmp_path_factory.mktemp("error-log")),
        )
        results = workflow.run()

        assert results.shape == (1, 8)
        assert "Error file" in results.columns

        error_file = results.loc[0, "Error file"]
        error = Exception("An error occurred when detecting anomalies!")
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_dtaianomaly_version(
            error_file, dtaianomaly.__version__
        )
        assert error_file_contains_error(error_file, error)
        assert error_file_results_in_error(error_file, error)

    def test_log_no_exception(self, tmp_path_factory):
        error = Exception("Dummy")
        error_file = log_error(
            error_log_path=str(tmp_path_factory.mktemp("error-log")),
            exception=Exception("Dummy"),
            data_loader=DemonstrationDataLoaderWithTrainData(),
            pipeline=Pipeline(preprocessor=Identity(), detector=IsolationForest(15)),
        )
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_dtaianomaly_version(
            error_file, dtaianomaly.__version__
        )
        assert error_file_contains_error(error_file, error)
        assert error_file_runs_successfully(error_file)

    def test_log_error_with_kwargs(self, tmp_path_factory):
        kwargs = {"lower_bound": 32, "upper_bound": 16}
        X = DemonstrationDataLoaderWithTrainData().load().X_train
        error = ValueError(
            f"Something went wrong when computing the window size using 'fft', "
            f"with lower bound 32 and upper bound 16 on a time series "
            f"with shape {X.shape}!"
        )
        error_file = log_error(
            error_log_path=str(tmp_path_factory.mktemp("error-log")),
            exception=error,
            data_loader=DemonstrationDataLoaderWithTrainData(),
            pipeline=Pipeline(preprocessor=Identity(), detector=IsolationForest("fft")),
            **kwargs,
        )
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_dtaianomaly_version(
            error_file, dtaianomaly.__version__
        )
        assert error_file_contains_error(error_file, error)
        assert error_file_results_in_error(error_file, error)


def error_file_has_correct_syntax(error_file):
    try:
        py_compile.compile(error_file, doraise=True)
        return True
    except py_compile.PyCompileError:
        return False


def error_file_contains_dtaianomaly_version(error_file, version):
    with open(error_file, "r") as file:
        for line in file:
            if f"dtaianomaly.__version__ == '{version}'" in line:
                return True
    return False


def error_file_contains_error(error_file, error):
    with open(error_file, "r") as file:
        for line in file:
            if line.startswith("#") and str(error) in line:
                return True
    return False


def error_file_results_in_error(error_file, error):
    output = _run_error_file(error_file)
    return output.returncode == 1 and str(error) in output.stderr


def error_file_runs_successfully(error_file):
    output = _run_error_file(error_file)
    return output.returncode == 0


def _run_error_file(error_file):
    # Include this file to the python path to find the classes
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = current_dir + os.pathsep + env.get("PYTHONPATH", "")

    # Add this file as import
    with open(error_file, "r+") as file:
        content = file.read()
        file.seek(0, 0)
        file.write(f"from {pathlib.Path(__file__).stem} import *\n" + content)

    return subprocess.run(
        [sys.executable, error_file], capture_output=True, text=True, env=env
    )
