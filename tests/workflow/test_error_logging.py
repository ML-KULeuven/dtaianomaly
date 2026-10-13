import os
import pathlib
import py_compile
import re
import subprocess
import sys

import numpy as np
import pytest

import dtaianomaly
from dtaianomaly.anomaly_detection import BaseDetector, IsolationForest, Supervision
from dtaianomaly.data import DataSet, DemonstrationTimeSeriesLoader, LazyDataLoader
from dtaianomaly.evaluation import AreaUnderROC, ProbaMetric
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


class ErrorMetric(ProbaMetric):

    def _compute(self, y_true, y_pred, **kwargs):
        raise Exception("An error occurred when computing the metric!")


@pytest.mark.slow
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

    def test_error_computing_metric(self, tmp_path_factory):
        results = Workflow(
            dataloaders=DemonstrationTimeSeriesLoader(),
            metrics=ErrorMetric(),
            detectors=IsolationForest(15),
            error_log_path=str(tmp_path_factory.mktemp("error-log")),
        ).run()

        error_file = results.loc[0, "Error file"]
        error = Exception("An error occurred when computing the metric!")
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


def _log_dummy_error(error_log_path, **kwargs):
    return log_error(
        error_log_path=error_log_path,
        exception=Exception("Dummy"),
        data_loader=DemonstrationDataLoaderWithTrainData(),
        pipeline=Pipeline(preprocessor=Identity(), detector=IsolationForest(15)),
        **kwargs,
    )


def _read(error_file):
    with open(error_file, "r") as file:
        return file.read()


class TestLogErrorFileName:

    def test_matches_pattern(self, tmp_path_factory):
        error_file = _log_dummy_error(str(tmp_path_factory.mktemp("error-log")))
        assert re.fullmatch(
            r"DemonstrationDataLoaderWithTrainData-IsolationForest-"
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-[0-9a-f]{8}\.err",
            os.path.basename(error_file),
        )

    def test_quick_succession_gives_distinct_files(self, tmp_path_factory):
        error_log_path = str(tmp_path_factory.mktemp("error-log"))
        first = _log_dummy_error(error_log_path)
        second = _log_dummy_error(error_log_path)
        assert first != second
        assert os.path.isfile(first) and os.path.isfile(second)


class TestLogErrorContent:

    def test_metric_error_header(self, tmp_path_factory):
        error_file = _log_dummy_error(
            str(tmp_path_factory.mktemp("error-log")),
            metrics=[ErrorMetric()],
            error_in_metric=True,
        )
        assert "# An error occurred while computing a metric!" in _read(error_file)

    def test_detection_error_header(self, tmp_path_factory):
        error_file = _log_dummy_error(str(tmp_path_factory.mktemp("error-log")))
        assert "# An error occurred while detecting anomalies!" in _read(error_file)

    def test_fit_on_X_train(self, tmp_path_factory):
        error_file = _log_dummy_error(
            str(tmp_path_factory.mktemp("error-log")), fit_on_X_train=True
        )
        assert "pipeline.fit(data.X_train, data.y_train" in _read(error_file)

    def test_fit_on_X_test(self, tmp_path_factory):
        error_file = _log_dummy_error(
            str(tmp_path_factory.mktemp("error-log")), fit_on_X_train=False
        )
        assert "pipeline.fit(data.X_test, None" in _read(error_file)

    def test_fit_on_X_test_with_kwargs(self, tmp_path_factory):
        error_file = _log_dummy_error(
            str(tmp_path_factory.mktemp("error-log")),
            fit_on_X_train=False,
            lower_bound=32,
        )
        assert "pipeline.fit(data.X_test, None, **kwargs)" in _read(error_file)


@pytest.mark.slow
class TestErrorLoggingFailedMetric:

    def test_error_file(self, tmp_path_factory):
        results = Workflow(
            dataloaders=DemonstrationTimeSeriesLoader(),
            metrics=[ErrorMetric(), AreaUnderROC()],
            detectors=IsolationForest(15),
            error_log_path=str(tmp_path_factory.mktemp("error-log")),
        ).run()

        error_file = results.loc[0, "Error file"]
        error = Exception("An error occurred when computing the metric!")
        assert error_file_has_correct_syntax(error_file)
        assert "# An error occurred while computing a metric!" in _read(error_file)
        assert "metrics = [ErrorMetric()]\n" in _read(error_file)
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
