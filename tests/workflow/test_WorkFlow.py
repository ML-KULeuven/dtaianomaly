
import pytest
import numpy as np

from dtaianomaly.workflow import Workflow
from dtaianomaly.data import UCRLoader, LazyDataLoader, DataSet, demonstration_time_series
from dtaianomaly.evaluation import Precision, Recall, AreaUnderROC
from dtaianomaly.thresholding import TopN, FixedCutoff
from dtaianomaly.preprocessing import Identity, ZNormalizer, Preprocessor
from dtaianomaly.anomaly_detection import MatrixProfileDetector, IsolationForest, LocalOutlierFactor, BaseDetector, Supervision
from dtaianomaly.workflow.Workflow import _get_train_test_data


class TestWorkflowInitialization:

    def test(self, tmp_path_factory):
        workflow = Workflow(
            dataloaders=[
                UCRLoader(path=str(tmp_path_factory.mktemp('some-path-1'))),
                UCRLoader(path=str(tmp_path_factory.mktemp('some-path-2')))
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), ZNormalizer()],
            detectors=[MatrixProfileDetector(window_size=100), IsolationForest(15)],
            n_jobs=4,
            trace_memory=True
        )
        assert len(workflow.pipelines) == 4
        assert workflow.provided_preprocessors

    def test_no_dataloaders(self):
        with pytest.raises(ValueError):
            Workflow(
                dataloaders=[],
                metrics=[Precision(), Recall(), AreaUnderROC()],
                thresholds=[TopN(10), FixedCutoff(0.5)],
                preprocessors=[Identity(), ZNormalizer()],
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
                preprocessors=[Identity(), ZNormalizer()],
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
                preprocessors=[Identity(), ZNormalizer()],
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
        assert all(isinstance(pipeline.pipeline.preprocessor, Identity) for pipeline in workflow.pipelines)
        assert not workflow.provided_preprocessors

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
        assert all(isinstance(pipeline.pipeline.preprocessor, Identity) for pipeline in workflow.pipelines)
        assert not workflow.provided_preprocessors

    def test_no_thresholds_binary_metrics(self, tmp_path_factory):
        with pytest.raises(ValueError):
            Workflow(
                dataloaders=[
                    UCRLoader(path=str(tmp_path_factory.mktemp('some-path-1'))),
                    UCRLoader(path=str(tmp_path_factory.mktemp('some-path-2')))
                ],
                metrics=[Precision(), Recall(), AreaUnderROC()],
                preprocessors=[Identity(), ZNormalizer()],
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
            preprocessors=[Identity(), ZNormalizer()],
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
                preprocessors=[Identity(), ZNormalizer()],
                detectors=[MatrixProfileDetector(window_size=100), IsolationForest(15)],
                n_jobs=0,
                trace_memory=True
            )


class DummyDataLoader(LazyDataLoader):

    def _load(self) -> DataSet:
        X, y = demonstration_time_series()
        return DataSet(X, y)


class TestWorkflowSuccess:

    def test(self, tmp_path_factory):
        path = str(tmp_path_factory.mktemp('some-path-1'))
        workflow = Workflow(
            dataloaders=[
                DummyDataLoader(path=path),
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), ZNormalizer()],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=1,
            trace_memory=False
        )
        results = workflow.run()
        assert results.shape == (4, 11)
        assert results['Dataset'].value_counts()[f"DummyDataLoader(path='{path}')"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['ZNormalizer()'] == 2
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

    def test_parallel(self, tmp_path_factory):
        path = str(tmp_path_factory.mktemp('some-path-1'))
        workflow = Workflow(
            dataloaders=[
                DummyDataLoader(path=path),
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), ZNormalizer()],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=4,
            trace_memory=False
        )
        results = workflow.run()
        assert results.shape == (4, 11)
        assert results['Dataset'].value_counts()[f"DummyDataLoader(path='{path}')"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['ZNormalizer()'] == 2
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

    def test_trace_memory(self, tmp_path_factory):
        path = str(tmp_path_factory.mktemp('some-path-1'))
        workflow = Workflow(
            dataloaders=[
                DummyDataLoader(path=path),
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), ZNormalizer()],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=4,
            trace_memory=True
        )
        results = workflow.run()
        assert results.shape == (4, 14)
        assert results['Dataset'].value_counts()[f"DummyDataLoader(path='{path}')"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['ZNormalizer()'] == 2
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

    def test_no_preprocessors(self, tmp_path_factory, univariate_time_series):
        path = str(tmp_path_factory.mktemp('some-path-1'))
        workflow = Workflow(
            dataloaders=[
                DummyDataLoader(path=path),
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=4,
            trace_memory=True
        )
        results = workflow.run()
        assert results.shape == (2, 13)
        assert results['Dataset'].value_counts()[f"DummyDataLoader(path='{path}')"] == 2
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

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        raise Exception('Dummy exception')


class SupervisedDetector(BaseDetector):

    def __init__(self):
        super().__init__(Supervision.SUPERVISED)

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        return np.zeros(X.shape[0])


class TestWorkflowFail:

    def test_failed_to_read_data(self, tmp_path_factory):
        path = str(tmp_path_factory.mktemp('some-path-1'))
        workflow = Workflow(
            dataloaders=[
                DummyDataLoader(path=path),
                DummyDataLoaderError(path=path),
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), ZNormalizer()],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=1,
            trace_memory=True,
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()
        assert results.shape == (8, 15)
        assert results['Dataset'].value_counts()[f"DummyDataLoader(path='{path}')"] == 4
        assert results['Dataset'].value_counts()[f"DummyDataLoaderError(path='{path}')"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['ZNormalizer()'] == 2
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
        path = str(tmp_path_factory.mktemp('some-path-1'))
        workflow = Workflow(
            dataloaders=[
                DummyDataLoader(path=path),
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[PreprocessorError(), ZNormalizer()],
            detectors=[LocalOutlierFactor(15), IsolationForest(15)],
            n_jobs=1,
            trace_memory=True,
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()
        assert results.shape == (4, 15)
        assert results['Dataset'].value_counts()[f"DummyDataLoader(path='{path}')"] == 4
        assert results['Preprocessor'].value_counts()['PreprocessorError()'] == 2
        assert results['Preprocessor'].value_counts()['ZNormalizer()'] == 2
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
        path = str(tmp_path_factory.mktemp('some-path-1'))
        workflow = Workflow(
            dataloaders=[
                DummyDataLoader(path=path),
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), ZNormalizer()],
            detectors=[DetectorError(), IsolationForest(15)],
            n_jobs=1,
            trace_memory=True,
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()
        assert results.shape == (4, 15)
        assert results['Dataset'].value_counts()[f"DummyDataLoader(path='{path}')"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['ZNormalizer()'] == 2
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
        path = str(tmp_path_factory.mktemp('some-path-1'))
        workflow = Workflow(
            dataloaders=[
                DummyDataLoader(path=path),
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[PreprocessorError(), ZNormalizer()],
            detectors=[DetectorError(), IsolationForest(15)],
            n_jobs=1,
            trace_memory=True,
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()
        assert results.shape == (4, 15)
        assert results['Dataset'].value_counts()[f"DummyDataLoader(path='{path}')"] == 4
        assert results['Preprocessor'].value_counts()['PreprocessorError()'] == 2
        assert results['Preprocessor'].value_counts()['ZNormalizer()'] == 2
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
        path = str(tmp_path_factory.mktemp('some-path-1'))
        workflow = Workflow(
            dataloaders=[
                DummyDataLoader(path=path),
            ],
            metrics=[Precision(), Recall(), AreaUnderROC()],
            thresholds=[TopN(10), FixedCutoff(0.5)],
            preprocessors=[Identity(), ZNormalizer()],
            detectors=[SupervisedDetector(), IsolationForest(15)],
            n_jobs=1,
            trace_memory=True,
            error_log_path=str(tmp_path_factory.mktemp('error-log'))
        )
        results = workflow.run()
        assert results.shape == (4, 14)
        assert results['Dataset'].value_counts()[f"DummyDataLoader(path='{path}')"] == 4
        assert results['Preprocessor'].value_counts()['Identity()'] == 2
        assert results['Preprocessor'].value_counts()['ZNormalizer()'] == 2
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

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        return np.zeros(X.shape[0])


class TestGetTrainTestData:

    def test_unsupervised_use_test_set_for_fit(self):
        data_set = DataSet(
            X_test=np.array([1, 2, 3, 4, 5, 6]),
            y_test=np.array([1, 0, 0, 0, 1, 0]),
            X_train=np.array([10, 20, 30, 40, 50])
        )
        detector = DummyDetector(Supervision.UNSUPERVISED)
        X_test, y_test, X_train, y_train, fit_on_X_train = _get_train_test_data(data_set, detector, fit_unsupervised_on_test_data=True)
        assert np.array_equal(data_set.X_test, X_test)
        assert np.array_equal(data_set.y_test, y_test)
        assert np.array_equal(data_set.X_test, X_train)
        assert y_train is None
        assert not fit_on_X_train

    def test_unsupervised_do_not_use_test_set_for_fit(self):
        data_set = DataSet(
            X_test=np.array([1, 2, 3, 4, 5, 6]),
            y_test=np.array([1, 0, 0, 0, 1, 0]),
            X_train=np.array([10, 20, 30, 40, 50])
        )
        detector = DummyDetector(Supervision.UNSUPERVISED)
        X_test, y_test, X_train, y_train, fit_on_X_train = _get_train_test_data(data_set, detector, fit_unsupervised_on_test_data=False)
        assert np.array_equal(data_set.X_test, X_test)
        assert np.array_equal(data_set.y_test, y_test)
        assert np.array_equal(data_set.X_train, X_train)
        assert y_train is None
        assert fit_on_X_train

    def test_semi_supervised_use_test_set_for_fit(self):
        data_set = DataSet(
            X_test=np.array([1, 2, 3, 4, 5, 6]),
            y_test=np.array([1, 0, 0, 0, 1, 0]),
            X_train=np.array([10, 20, 30, 40, 50])
        )
        detector = DummyDetector(Supervision.SEMI_SUPERVISED)
        X_test, y_test, X_train, y_train, fit_on_X_train = _get_train_test_data(data_set, detector, fit_unsupervised_on_test_data=True)
        assert np.array_equal(data_set.X_test, X_test)
        assert np.array_equal(data_set.y_test, y_test)
        assert np.array_equal(data_set.X_train, X_train)
        assert y_train is None
        assert fit_on_X_train

    def test_semi_supervised_do_not_use_test_set_for_fit(self):
        data_set = DataSet(
            X_test=np.array([1, 2, 3, 4, 5, 6]),
            y_test=np.array([1, 0, 0, 0, 1, 0]),
            X_train=np.array([10, 20, 30, 40, 50])
        )
        detector = DummyDetector(Supervision.SEMI_SUPERVISED)
        X_test, y_test, X_train, y_train, fit_on_X_train = _get_train_test_data(data_set, detector, fit_unsupervised_on_test_data=False)
        assert np.array_equal(data_set.X_test, X_test)
        assert np.array_equal(data_set.y_test, y_test)
        assert np.array_equal(data_set.X_train, X_train)
        assert y_train is None
        assert fit_on_X_train
