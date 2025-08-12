
import sys
from unittest.mock import MagicMock
import pytest
import warnings
import numpy as np
import os

from dtaianomaly.workflow import Workflow, JobBasedWorkflow
from dtaianomaly.data import UCRLoader, LazyDataLoader, DataSet, DemonstrationTimeSeriesLoader
from dtaianomaly.evaluation import Precision, Recall, AreaUnderROC
from dtaianomaly.thresholding import TopN, FixedCutoff
from dtaianomaly.preprocessing import Identity, StandardScaler, Preprocessor
from dtaianomaly.anomaly_detection import MatrixProfileDetector, IsolationForest, LocalOutlierFactor, BaseDetector, Supervision, RandomDetector
from dtaianomaly.workflow.JobBasedWorkflow import _get_train_test_data


class TestWorkflowInitialization:

    def test(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=[
                UCRLoader(path=str(tmp_path_factory.mktemp('some-path-1'))),
                UCRLoader(path=str(tmp_path_factory.mktemp('some-path-2')))
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), StandardScaler()],
            detectors=[MatrixProfileDetector(window_size=100), IsolationForest(15)],
            n_jobs=4,
            trace_memory=True
        )
        assert len(workflow.jobs) == 8
        assert workflow.provided_preprocessors()

    def test_invalid_jobs(self):
        with pytest.raises(TypeError):
            JobBasedWorkflow(
                jobs=[1, 2, 3],
                metrics=[Precision(), Recall(), AreaUnderROC()],
                thresholds=[TopN(10), FixedCutoff(0.5)],
            )

    def test_no_dataloaders(self):
        with pytest.raises(ValueError):
            Workflow(
                dataloaders=[],
                metrics=[Precision(), Recall(), AreaUnderROC()],
                thresholds=[TopN(10), FixedCutoff(0.5)],
                preprocessors=[Identity(), StandardScaler()],
                detectors=[MatrixProfileDetector(window_size=100), IsolationForest(15)],
                n_jobs=4,
                trace_memory=True
            )

    def test_no_metrics(self, tmp_path_factory):
        with pytest.raises(ValueError):
            Workflow(
                dataloaders=[
                    UCRLoader(path=str(tmp_path_factory.mktemp('some-path-1'))),
                    UCRLoader(path=str(tmp_path_factory.mktemp('some-path-2')))
                ],
                metrics=[],
                thresholds=[TopN(10), FixedCutoff(0.5)],
                preprocessors=[Identity(), StandardScaler()],
                detectors=[MatrixProfileDetector(window_size=100), IsolationForest(15)],
                n_jobs=4,
                trace_memory=True
            )

    def test_no_detectors(self, tmp_path_factory):
        with pytest.raises(ValueError):
            Workflow(
                dataloaders=[
                    UCRLoader(path=str(tmp_path_factory.mktemp('some-path-1'))),
                    UCRLoader(path=str(tmp_path_factory.mktemp('some-path-2')))
                ],
                metrics=[Precision(), Recall(), AreaUnderROC()],
                thresholds=[TopN(10), FixedCutoff(0.5)],
                preprocessors=[Identity(), StandardScaler()],
                detectors=[],
                n_jobs=4,
                trace_memory=True
            )

    def test_no_preprocessors(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=[
                UCRLoader(path=str(tmp_path_factory.mktemp('some-path-1'))),
                UCRLoader(path=str(tmp_path_factory.mktemp('some-path-2')))
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            detectors=[MatrixProfileDetector(window_size=100), IsolationForest(15)],
            n_jobs=4,
            trace_memory=True
        )
        assert all(not job.has_preprocessor for job in workflow.jobs)
        assert not workflow.provided_preprocessors()

    def test_empty_preprocessors(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=[
                UCRLoader(path=str(tmp_path_factory.mktemp('some-path-1'))),
                UCRLoader(path=str(tmp_path_factory.mktemp('some-path-2')))
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[],
            detectors=[MatrixProfileDetector(window_size=100), IsolationForest(15)],
            n_jobs=4,
            trace_memory=True
        )
        assert all(not job.has_preprocessor for job in workflow.jobs)
        assert not workflow.provided_preprocessors()

    def test_no_thresholds_binary_metrics(self, tmp_path_factory):
        with pytest.raises(ValueError):
            Workflow(
                dataloaders=[
                    UCRLoader(path=str(tmp_path_factory.mktemp('some-path-1'))),
                    UCRLoader(path=str(tmp_path_factory.mktemp('some-path-2')))
                ],
                metrics=[Precision(), Recall(), AreaUnderROC()],
                preprocessors=[Identity(), StandardScaler()],
                detectors=[MatrixProfileDetector(window_size=100), IsolationForest(15)],
                n_jobs=4,
                trace_memory=True
            )

    def test_no_thresholds_no_binary_metrics(self, tmp_path_factory):
        Workflow(
            dataloaders=[
                UCRLoader(path=str(tmp_path_factory.mktemp('some-path-1'))),
                UCRLoader(path=str(tmp_path_factory.mktemp('some-path-2')))
            ],
            metrics=[AreaUnderROC()],
            preprocessors=[Identity(), StandardScaler()],
            detectors=[MatrixProfileDetector(window_size=100), IsolationForest(15)],
            n_jobs=4,
            trace_memory=True
        )

    def test_invalid_nb_jobs(self, tmp_path_factory):
        with pytest.raises(ValueError):
            Workflow(
                dataloaders=[
                    UCRLoader(path=str(tmp_path_factory.mktemp('some-path-1'))),
                    UCRLoader(path=str(tmp_path_factory.mktemp('some-path-2')))
                ],                metrics=[Precision(), Recall(), AreaUnderROC()],
                thresholds=[TopN(10), FixedCutoff(0.5)],
                preprocessors=[Identity(), StandardScaler()],
                detectors=[MatrixProfileDetector(window_size=100), IsolationForest(15)],
                n_jobs=0,
                trace_memory=True
            )


@pytest.mark.parametrize("show_progress", [True, False])
class TestWorkflowSuccess:
    pytest.importorskip("tqdm")

    def test(self, tmp_path_factory, show_progress):
        workflow = Workflow(
            dataloaders=[DemonstrationTimeSeriesLoader()],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), StandardScaler()],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=1,
            trace_memory=False,
            show_progress=show_progress
        )
        results = workflow.run()
        assert results.shape == (4, 11)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['StandardScaler()'] == 2
        assert results['Detector'].value_counts()['LocalOutlierFactor(window_size=15)'] == 2
        assert results['Detector'].value_counts()['IsolationForest(window_size=15)'] == 2
        assert 'Peak Memory Fit [MB]' not in results.columns
        assert 'Peak Memory Predict [MB]' not in results.columns
        assert 'Peak Memory [MB]' not in results.columns
        assert not np.any(results == 'Error')
        assert not results.isna().any().any()
        # Check the order
        assert results.columns[0] == 'Dataset'
        assert results.columns[1] == 'Detector'
        assert results.columns[2] == 'Preprocessor'
        assert results.columns[3] == 'Runtime Fit [s]'
        assert results.columns[4] == 'Runtime Predict [s]'
        assert results.columns[5] == 'Runtime [s]'
        # Check that no anomaly scores are saved
        assert 'Anomaly scores file' not in results.columns

    def test_with_kwargs(self, tmp_path_factory, show_progress):
        workflow = Workflow(
            dataloaders=[DemonstrationTimeSeriesLoader()],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), StandardScaler()],
            detectors=[LocalOutlierFactor('fft'), IsolationForest('fft')],
            n_jobs=1,
            trace_memory=False,
            show_progress=show_progress
        )
        results = workflow.run(
            # Force default window size by passing invalid values
            lower_bound=32,
            upper_bound=16,
            default_window_size=24,
        )
        assert results.shape == (4, 11)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['StandardScaler()'] == 2
        assert results['Detector'].value_counts()["LocalOutlierFactor(window_size='fft')"] == 2
        assert results['Detector'].value_counts()["IsolationForest(window_size='fft')"] == 2
        assert 'Peak Memory Fit [MB]' not in results.columns
        assert 'Peak Memory Predict [MB]' not in results.columns
        assert 'Peak Memory [MB]' not in results.columns
        assert not np.any(results == 'Error')
        assert not results.isna().any().any()
        # Check the order
        assert results.columns[0] == 'Dataset'
        assert results.columns[1] == 'Detector'
        assert results.columns[2] == 'Preprocessor'
        assert results.columns[3] == 'Runtime Fit [s]'
        assert results.columns[4] == 'Runtime Predict [s]'
        assert results.columns[5] == 'Runtime [s]'

    def test_parallel(self, tmp_path_factory, show_progress):
        workflow = Workflow(
            dataloaders=[DemonstrationTimeSeriesLoader()],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), StandardScaler()],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=4,
            trace_memory=False,
            show_progress=show_progress
        )
        results = workflow.run()
        assert results.shape == (4, 11)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['StandardScaler()'] == 2
        assert results['Detector'].value_counts()['LocalOutlierFactor(window_size=15)'] == 2
        assert results['Detector'].value_counts()['IsolationForest(window_size=15)'] == 2
        assert 'Peak Memory Fit [MB]' not in results.columns
        assert 'Peak Memory Predict [MB]' not in results.columns
        assert 'Peak Memory [MB]' not in results.columns
        assert not np.any(results == 'Error')
        assert not results.isna().any().any()
        # Check the order
        assert results.columns[0] == 'Dataset'
        assert results.columns[1] == 'Detector'
        assert results.columns[2] == 'Preprocessor'
        assert results.columns[3] == 'Runtime Fit [s]'
        assert results.columns[4] == 'Runtime Predict [s]'
        assert results.columns[5] == 'Runtime [s]'

    def test_trace_memory(self, tmp_path_factory, show_progress):
        workflow = Workflow(
            dataloaders=[DemonstrationTimeSeriesLoader()],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), StandardScaler()],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=4,
            trace_memory=True,
            show_progress=show_progress
        )
        results = workflow.run()
        assert results.shape == (4, 14)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['StandardScaler()'] == 2
        assert results['Detector'].value_counts()['LocalOutlierFactor(window_size=15)'] == 2
        assert results['Detector'].value_counts()['IsolationForest(window_size=15)'] == 2
        assert 'Peak Memory Fit [MB]' in results.columns
        assert 'Peak Memory Predict [MB]' in results.columns
        assert 'Peak Memory [MB]' in results.columns
        assert not np.any(results == 'Error')
        assert not results.isna().any().any()
        # Check the order
        assert results.columns[0] == 'Dataset'
        assert results.columns[1] == 'Detector'
        assert results.columns[2] == 'Preprocessor'
        assert results.columns[3] == 'Runtime Fit [s]'
        assert results.columns[4] == 'Runtime Predict [s]'
        assert results.columns[5] == 'Runtime [s]'
        assert results.columns[6] == 'Peak Memory Fit [MB]'
        assert results.columns[7] == 'Peak Memory Predict [MB]'
        assert results.columns[8] == 'Peak Memory [MB]'

    def test_no_preprocessors(self, tmp_path_factory, univariate_time_series, show_progress):
        workflow = Workflow(
            dataloaders=[DemonstrationTimeSeriesLoader()],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=4,
            trace_memory=True,
            show_progress=show_progress
        )
        results = workflow.run()
        assert results.shape == (2, 13)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 2
        assert results['Detector'].value_counts()['LocalOutlierFactor(window_size=15)'] == 1
        assert results['Detector'].value_counts()['IsolationForest(window_size=15)'] == 1
        assert 'Peak Memory Fit [MB]' in results.columns
        assert 'Peak Memory Predict [MB]' in results.columns
        assert 'Peak Memory [MB]' in results.columns
        assert not np.any(results == 'Error')
        assert not results.isna().any().any()
        # Check the order
        assert results.columns[0] == 'Dataset'
        assert results.columns[1] == 'Detector'
        assert results.columns[2] == 'Runtime Fit [s]'
        assert results.columns[3] == 'Runtime Predict [s]'
        assert results.columns[4] == 'Runtime [s]'
        assert results.columns[5] == 'Peak Memory Fit [MB]'
        assert results.columns[6] == 'Peak Memory Predict [MB]'
        assert results.columns[7] == 'Peak Memory [MB]'
        assert 'Preprocessor' not in results.columns


class DummyDataLoaderError(LazyDataLoader):

    def _load(self) -> DataSet:
        raise Exception('Dummy exception')


class PreprocessorError(Preprocessor):

    def _fit(self, X, y=None):
        return self

    def _transform(self, X, y=None):
        raise Exception('Dummy exception')


class DetectorError(BaseDetector):

    def __init__(self):
        super().__init__(Supervision.UNSUPERVISED)

    def _fit(self, X, y=None, **kwargs):
        pass

    def _decision_function(self, X):
        raise Exception('Dummy exception')


class SupervisedDetector(BaseDetector):

    def __init__(self):
        super().__init__(Supervision.SUPERVISED)

    def _fit(self, X, y=None, **kwargs):
        pass

    def _decision_function(self, X):
        return np.zeros(X.shape[0])


class TestWorkflowFail:

    def test_failed_to_read_data(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=[
                DemonstrationTimeSeriesLoader(),
                DummyDataLoaderError(),
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), StandardScaler()],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=1,
            trace_memory=True,
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()
        assert results.shape == (8, 15)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 4
        assert results['Dataset'].value_counts()["DummyDataLoaderError()"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['StandardScaler()'] == 2
        assert results['Detector'].value_counts()['LocalOutlierFactor(window_size=15)'] == 2
        assert results['Detector'].value_counts()['IsolationForest(window_size=15)'] == 2
        assert 'Peak Memory Fit [MB]' in results.columns
        assert 'Peak Memory Predict [MB]' in results.columns
        assert 'Peak Memory [MB]' in results.columns
        assert np.any(results == 'Error', axis=0).sum() == 13
        assert np.any(results == 'Error', axis=1).sum().sum() == 4
        assert 'Error file' in results.columns
        assert results['Error file'].isna().sum() == 4

    def test_failed_to_preprocess(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=[DemonstrationTimeSeriesLoader()],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[PreprocessorError(), StandardScaler()],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=1,
            trace_memory=True,
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()
        assert results.shape == (4, 15)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 4
        assert results['Preprocessor'].value_counts()['PreprocessorError()'] == 2
        assert results['Preprocessor'].value_counts()['StandardScaler()'] == 2
        assert results['Detector'].value_counts()['LocalOutlierFactor(window_size=15)'] == 2
        assert results['Detector'].value_counts()['IsolationForest(window_size=15)'] == 2
        assert 'Peak Memory Fit [MB]' in results.columns
        assert 'Peak Memory Predict [MB]' in results.columns
        assert 'Peak Memory [MB]' in results.columns
        assert np.any(results == 'Error', axis=0).sum() == 11
        assert np.any(results == 'Error', axis=1).sum().sum() == 2
        assert 'Error file' in results.columns
        assert results['Error file'].isna().sum() == 2

    def test_failed_to_fit_model(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=[DemonstrationTimeSeriesLoader()],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), StandardScaler()],
            detectors=[DetectorError(), IsolationForest(15)],
            n_jobs=1,
            trace_memory=True,
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()
        assert results.shape == (4, 15)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['StandardScaler()'] == 2
        assert results['Detector'].value_counts()['DetectorError()'] == 2
        assert results['Detector'].value_counts()['IsolationForest(window_size=15)'] == 2
        assert 'Peak Memory Fit [MB]' in results.columns
        assert 'Peak Memory Predict [MB]' in results.columns
        assert 'Peak Memory [MB]' in results.columns
        assert np.any(results == 'Error', axis=0).sum() == 9
        assert np.any(results == 'Error', axis=1).sum().sum() == 2
        assert 'Error file' in results.columns
        assert results['Error file'].isna().sum() == 2

    def test_failed_to_preprocess_and_to_fit_model(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=[DemonstrationTimeSeriesLoader()],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[PreprocessorError(), StandardScaler()],
            detectors=[DetectorError(), IsolationForest(15)],
            n_jobs=1,
            trace_memory=True,
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()
        assert results.shape == (4, 15)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 4
        assert results['Preprocessor'].value_counts()['PreprocessorError()'] == 2
        assert results['Preprocessor'].value_counts()['StandardScaler()'] == 2
        assert results['Detector'].value_counts()['DetectorError()'] == 2
        assert results['Detector'].value_counts()['IsolationForest(window_size=15)'] == 2
        assert 'Peak Memory Fit [MB]' in results.columns
        assert 'Peak Memory Predict [MB]' in results.columns
        assert 'Peak Memory [MB]' in results.columns
        assert np.any(results == 'Error', axis=0).sum() == 11
        assert np.any(results == 'Error', axis=1).sum().sum() == 3
        assert 'Error file' in results.columns
        assert results['Error file'].isna().sum() == 1

    def test_not_compatible(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=[DemonstrationTimeSeriesLoader()],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), StandardScaler()],
            detectors=[SupervisedDetector(), IsolationForest(15)],
            n_jobs=1,
            trace_memory=True,
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()
        assert results.shape == (4, 14)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['StandardScaler()'] == 2
        assert results['Detector'].value_counts()['SupervisedDetector()'] == 2
        assert results['Detector'].value_counts()['IsolationForest(window_size=15)'] == 2
        assert 'Peak Memory Fit [MB]' in results.columns
        assert 'Peak Memory Predict [MB]' in results.columns
        assert 'Peak Memory [MB]' in results.columns
        expected_error_message = f'Not compatible: detector with supervision Supervision.SUPERVISED ' \
                                 f'for data set with compatible supervision [Supervision.UNSUPERVISED]'
        assert np.any(results == expected_error_message, axis=0).sum() == 11
        assert np.any(results == expected_error_message, axis=1).sum().sum() == 2
        assert 'Error file' not in results.columns


class DummyDetector(BaseDetector):

    def __init__(self, supervision):
        super().__init__(supervision)

    def _fit(self, X, y=None, **kwargs):
        pass

    def _decision_function(self, X):
        return np.zeros(X.shape[0])


@pytest.mark.parametrize("fit_unsupervised_on_test_data", [True, False])
@pytest.mark.parametrize("fit_semi_supervised_on_test_data", [True, False])
class TestGetTrainTestData:

    def test_unsupervised(self, fit_unsupervised_on_test_data, fit_semi_supervised_on_test_data):
        data_set = DataSet(
            X_test=np.array([1, 2, 3, 4, 5, 6]),
            y_test=np.array([1, 0, 0, 0, 1, 0]),
            X_train=np.array([10, 20, 30, 40, 50])
        )
        detector = DummyDetector(Supervision.UNSUPERVISED)
        X_test, y_test, X_train, y_train, fit_on_X_train = _get_train_test_data(
            data_set,
            detector,
            fit_unsupervised_on_test_data=fit_unsupervised_on_test_data,
            fit_semi_supervised_on_test_data=fit_semi_supervised_on_test_data
        )
        assert np.array_equal(data_set.X_test, X_test)
        assert np.array_equal(data_set.y_test, y_test)

        if fit_unsupervised_on_test_data:
            assert np.array_equal(data_set.X_test, X_train)
            assert not fit_on_X_train
        else:
            assert np.array_equal(data_set.X_test, X_test)
            assert fit_on_X_train

    def test_semi_supervised(self, fit_unsupervised_on_test_data, fit_semi_supervised_on_test_data):
        data_set = DataSet(
            X_test=np.array([1, 2, 3, 4, 5, 6]),
            y_test=np.array([1, 0, 0, 0, 1, 0]),
            X_train=np.array([10, 20, 30, 40, 50])
        )
        detector = DummyDetector(Supervision.SEMI_SUPERVISED)
        X_test, y_test, X_train, y_train, fit_on_X_train = _get_train_test_data(
            data_set,
            detector,
            fit_unsupervised_on_test_data=fit_unsupervised_on_test_data,
            fit_semi_supervised_on_test_data=fit_semi_supervised_on_test_data
        )
        assert np.array_equal(data_set.X_test, X_test)
        assert np.array_equal(data_set.y_test, y_test)

        if fit_semi_supervised_on_test_data:
            assert np.array_equal(data_set.X_test, X_train)
            assert not fit_on_X_train
        else:
            assert np.array_equal(data_set.X_test, X_test)
            assert fit_on_X_train

    def test_supervised(self, fit_unsupervised_on_test_data, fit_semi_supervised_on_test_data):
        data_set = DataSet(
            X_test=np.array([1, 2, 3, 4, 5, 6]),
            y_test=np.array([1, 0, 0, 0, 1, 0]),
            X_train=np.array([10, 20, 30, 40, 50])
        )
        detector = DummyDetector(Supervision.SUPERVISED)
        X_test, y_test, X_train, y_train, fit_on_X_train = _get_train_test_data(
            data_set,
            detector,
            fit_unsupervised_on_test_data=fit_unsupervised_on_test_data,
            fit_semi_supervised_on_test_data=fit_semi_supervised_on_test_data
        )
        assert np.array_equal(data_set.X_test, X_test)
        assert np.array_equal(data_set.y_test, y_test)

        assert np.array_equal(data_set.X_test, X_test)
        assert fit_on_X_train


@pytest.mark.parametrize('n_jobs', [1, 2])
class TestShowProgress:

    def test_show_progress_tqdm_installed(self, monkeypatch, tmp_path_factory, n_jobs):

        # Skip if tqdm doesn't exist
        tqdm = pytest.importorskip("tqdm")

        mock = MagicMock(side_effect=tqdm.tqdm)
        monkeypatch.setitem(sys.modules, "tqdm", MagicMock(tqdm=mock))

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Treat warnings as errors
            Workflow(
                dataloaders=DemonstrationTimeSeriesLoader(),
                metrics=AreaUnderROC(),
                detectors=[RandomDetector(seed) for seed in range(5)],
                n_jobs=n_jobs,
                show_progress=True
            ).run()

        mock.assert_called_once()

    def test_show_progress_tqdm_not_installed(self, monkeypatch, tmp_path_factory, n_jobs):

        class BlockTqdmImport:
            """A custom import hook that prevents tqdm from being imported."""

            @staticmethod
            def find_spec(fullname, path, target=None):
                if fullname == "tqdm":
                    raise ModuleNotFoundError("No module named 'tqdm'")

        monkeypatch.syspath_prepend("")  # Ensure imports still work
        monkeypatch.setattr(sys, "meta_path", [BlockTqdmImport()] + sys.meta_path)
        monkeypatch.delitem(sys.modules, "tqdm", raising=False)

        with pytest.raises(ModuleNotFoundError, match="No module named 'tqdm'"):
            import tqdm  # First attempt should fail

        with pytest.raises(ModuleNotFoundError, match="No module named 'tqdm'"):
            import tqdm  # Second attempt should also fail => keeps failing

        with pytest.warns(Warning):
            workflow = Workflow(
                dataloaders=DemonstrationTimeSeriesLoader(),
                metrics=AreaUnderROC(),
                detectors=[RandomDetector(seed) for seed in range(5)],
                n_jobs=n_jobs,
                show_progress=True
            )
            workflow.run()

        assert not workflow.show_progress

    def test_do_not_show_progress_tqdm_installed(self, monkeypatch, tmp_path_factory, n_jobs):

        # Skip if tqdm doesn't exist
        pytest.importorskip("tqdm")

        mock = MagicMock(side_effect=Exception)
        monkeypatch.setitem(sys.modules, "tqdm", MagicMock(tqdm=mock))

        Workflow(
            dataloaders=DemonstrationTimeSeriesLoader(),
            metrics=AreaUnderROC(),
            detectors=[RandomDetector(seed) for seed in range(5)],
            n_jobs=n_jobs,
            show_progress=False
        ).run()

        mock.assert_not_called()

    def test_do_not_show_progress_tqdm_not_installed(self, monkeypatch, tmp_path_factory, n_jobs):
        mock = MagicMock(side_effect=Exception)
        monkeypatch.setitem(sys.modules, "tqdm", MagicMock(tqdm=mock))

        Workflow(
            dataloaders=DemonstrationTimeSeriesLoader(),
            metrics=AreaUnderROC(),
            detectors=[RandomDetector(seed) for seed in range(5)],
            n_jobs=n_jobs,
            show_progress=False
        ).run()

        mock.assert_not_called()


class TestSaveAnomalyScores:

    def test(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=[DemonstrationTimeSeriesLoader()],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=4,
            trace_memory=True,
            anomaly_scores_path=str(tmp_path_factory.mktemp('anomaly-scores')),
        )
        results = workflow.run()
        assert results.shape == (2, 14)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 2
        assert results['Detector'].value_counts()['LocalOutlierFactor(window_size=15)'] == 1
        assert results['Detector'].value_counts()['IsolationForest(window_size=15)'] == 1
        assert 'Peak Memory Fit [MB]' in results.columns
        assert 'Peak Memory Predict [MB]' in results.columns
        assert 'Peak Memory [MB]' in results.columns
        assert not np.any(results == 'Error')
        assert not results.isna().any().any()
        # Check the order
        assert results.columns[0] == 'Dataset'
        assert results.columns[1] == 'Detector'
        assert results.columns[2] == 'Runtime Fit [s]'
        assert results.columns[3] == 'Runtime Predict [s]'
        assert results.columns[4] == 'Runtime [s]'
        assert results.columns[5] == 'Peak Memory Fit [MB]'
        assert results.columns[6] == 'Peak Memory Predict [MB]'
        assert results.columns[7] == 'Peak Memory [MB]'
        assert results.columns[-1] == 'Anomaly scores file'
        assert 'Preprocessor' not in results.columns

        for path in results['Anomaly scores file']:
            os.path.exists(path)

    def test_with_error(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=[DemonstrationTimeSeriesLoader()],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            detectors=[DetectorError(), IsolationForest(15)],
            n_jobs=1,
            trace_memory=True,
            anomaly_scores_path=str(tmp_path_factory.mktemp('anomaly-scores')),
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()
        results.to_csv('tmp.csv')
        assert results.shape == (2, 15)
        assert results['Dataset'].value_counts()["DemonstrationTimeSeriesLoader()"] == 2
        assert results['Detector'].value_counts()['DetectorError()'] == 1
        assert results['Detector'].value_counts()['IsolationForest(window_size=15)'] == 1
        assert 'Peak Memory Fit [MB]' in results.columns
        assert 'Peak Memory Predict [MB]' in results.columns
        assert 'Peak Memory [MB]' in results.columns
        assert np.any(results == 'Error', axis=0).sum() == 9
        assert np.any(results == 'Error', axis=1).sum().sum() == 1
        assert 'Error file' in results.columns
        assert results['Error file'].isna().sum() == 1
        assert results['Anomaly scores file'].isna().sum() == 1

        assert results.columns[-1] == 'Anomaly scores file'
        for path in results['Anomaly scores file']:
            if isinstance(path, str):
                os.path.exists(path)
