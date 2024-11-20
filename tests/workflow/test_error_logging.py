
import os
import sys
import py_compile
import subprocess
import pathlib

from dtaianomaly.data import LazyDataLoader, DataSet, demonstration_time_series
from dtaianomaly.anomaly_detection import BaseDetector, IsolationForest, Supervision
from dtaianomaly.preprocessing import Preprocessor, Identity, ChainedPreprocessor
from dtaianomaly.pipeline import Pipeline
from dtaianomaly.evaluation import AreaUnderROC
from dtaianomaly.workflow import Workflow
from dtaianomaly.workflow.error_logging import log_error


class DemonstrationDataLoader(LazyDataLoader):

    def __init__(self):
        super().__init__('.')

    def _load(self) -> DataSet:
        X, y = demonstration_time_series()
        return DataSet(X, y)


class ErrorDataLoader(LazyDataLoader):

    def _load(self):
        raise Exception('An error occurred when loading data!')


class ErrorPreprocessor(Preprocessor):

    def _fit(self, X, y=None):
        return self

    def _transform(self, X, y=None):
        raise Exception('An error occurred preprocessing data!')


class ErrorAnomalyDetector(BaseDetector):

    def __init__(self):
        super().__init__(Supervision.UNSUPERVISED)

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        raise Exception('An error occurred when detecting anomalies!')


class TestErrorLogging:

    def test_error_loading(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=ErrorDataLoader('.'),
            metrics=AreaUnderROC(),
            preprocessors=ChainedPreprocessor(Identity(), ErrorPreprocessor()),
            detectors=IsolationForest(15),
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()

        assert results.shape == (1, 6)
        assert 'Error file' in results.columns

        error_file = results.loc[0, 'Error file']
        error = Exception('An error occurred when loading data!')
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_error(error_file, error)
        assert error_file_results_in_error(error_file, error)

    def test_error_preprocessing(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=DemonstrationDataLoader(),
            metrics=AreaUnderROC(),
            preprocessors=ErrorPreprocessor(),
            detectors=IsolationForest(15),
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()

        assert results.shape == (1, 6)
        assert 'Error file' in results.columns

        error_file = results.loc[0, 'Error file']
        error = Exception('An error occurred preprocessing data!')
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_error(error_file, error)
        assert error_file_results_in_error(error_file, error)

    def test_error_chained_preprocessing(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=DemonstrationDataLoader(),
            metrics=AreaUnderROC(),
            preprocessors=ErrorPreprocessor(),
            detectors=IsolationForest(15),
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()

        assert results.shape == (1, 6)
        assert 'Error file' in results.columns

        error_file = results.loc[0, 'Error file']
        error = Exception('An error occurred preprocessing data!')
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_error(error_file, error)
        assert error_file_results_in_error(error_file, error)

    def test_error_detecting_anomalies(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=DemonstrationDataLoader(),
            metrics=AreaUnderROC(),
            preprocessors=Identity(),
            detectors=ErrorAnomalyDetector(),
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()

        assert results.shape == (1, 6)
        assert 'Error file' in results.columns

        error_file = results.loc[0, 'Error file']
        error = Exception('An error occurred when detecting anomalies!')
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_error(error_file, error)
        assert error_file_results_in_error(error_file, error)

    def test_log_no_exception(self, tmp_path_factory):
        error = Exception('Dummy')
        error_file = log_error(
            error_log_path=str(tmp_path_factory.mktemp('error-log')),
            exception=Exception('Dummy'),
            data_loader=DemonstrationDataLoader(),
            pipeline=Pipeline(
                preprocessor=Identity(),
                detector=IsolationForest(15)
            )
        )
        assert error_file_has_correct_syntax(error_file)
        assert error_file_contains_error(error_file, error)
        assert error_file_runs_successfully(error_file)


def error_file_has_correct_syntax(error_file):
    try:
        py_compile.compile(error_file, doraise=True)
        return True
    except py_compile.PyCompileError:
        return False


def error_file_contains_error(error_file, error):
    with open(error_file, 'r') as file:
        for line in file:
            if line.startswith('#') and str(error) in line:
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
    env['PYTHONPATH'] = current_dir + os.pathsep + env.get('PYTHONPATH', '')

    # Add this file as import
    with open(error_file, 'r+') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(f'from {pathlib.Path(__file__).stem} import *\n' + content)

    return subprocess.run([sys.executable, error_file], capture_output=True, text=True, env=env)
