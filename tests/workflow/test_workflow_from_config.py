
import pytest
import json
import toml
import pathlib

from dtaianomaly import preprocessing, anomaly_detection, evaluation, thresholding, data
from dtaianomaly.workflow import workflow_from_config, interpret_config, Workflow
from dtaianomaly.workflow.workflow_from_config import interpret_dataloaders, data_entry
from dtaianomaly.workflow.workflow_from_config import interpret_detectors, detector_entry
from dtaianomaly.workflow.workflow_from_config import interpret_preprocessing, preprocessing_entry
from dtaianomaly.workflow.workflow_from_config import interpret_metrics, metric_entry
from dtaianomaly.workflow.workflow_from_config import interpret_thresholds, threshold_entry
from dtaianomaly.workflow.workflow_from_config import interpret_additional_information


DATA_PATH = f'{pathlib.Path(__file__).parent.parent.parent}/data'


@pytest.fixture
def valid_config():
    return {
        "dataloaders": [
            {"type": "UCRLoader", "path": f"{DATA_PATH}/UCR-time-series-anomaly-archive/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"},
            {"type": "directory", "path": f"{DATA_PATH}/UCR-time-series-anomaly-archive", "base_type": "UCRLoader"}
        ],
        "metrics": [
            {"type": "Precision"},
            {"type": "Recall"},
            {"type": "AreaUnderROC"}
        ],
        "thresholds": [
            {"type": "TopN", "n": 10},
            {"type": "FixedCutoff", "cutoff": 0.5}
        ],
        "preprocessors": [
            {"type": "MovingAverage", "window_size": 15},
            {"type": "Identity"}
        ],
        "detectors": [
            {"type": "IsolationForest", "window_size": 50},
            {"type": "MatrixProfileDetector", "window_size": 50}
        ],
        'n_jobs': 4,
        'trace_memory': True
    }


class TestWorkflowFromConfig:

    def test_non_str_path(self):
        with pytest.raises(TypeError):
            workflow_from_config({'detector': 'IsolationForest'})

    def test_non_existing_file(self):
        with pytest.raises(FileNotFoundError):
            workflow_from_config('path-to-non-existing-file.json')

    def test_non_json_file(self, tmp_path):
        open(tmp_path / 'config.txt', 'a').close()  # Make the file
        with pytest.raises(ValueError):
            workflow_from_config(str(tmp_path / 'config.txt'))

    def test_file_too_large(self, tmp_path, valid_config):
        with open(tmp_path / 'config.json', 'a') as file:
            valid_config['detectors'].extend([{"type": "IsolationForest", "window_size": w} for w in range(1, 10000)])  # Make the file big
            json.dump(valid_config, file)
        workflow_from_config(str(tmp_path / 'config.json'), 1000000)  # Does not throw an error
        with pytest.raises(ValueError):
            workflow_from_config(str(tmp_path / 'config.json'), 1)

    def test_success_json(self, tmp_path, valid_config):
        with open(tmp_path / 'config.json', 'a') as file:
            json.dump(valid_config, file)
        workflow = workflow_from_config(str(tmp_path / 'config.json'))
        assert isinstance(workflow, Workflow)

    def test_success_toml(self, tmp_path, valid_config):
        with open(tmp_path / 'config.toml', 'w') as file:
            toml.dump(valid_config, file)
        workflow = workflow_from_config(str(tmp_path / 'config.toml'))
        assert isinstance(workflow, Workflow)


class TestInterpretConfig:

    def test(self, valid_config):
        workflow = interpret_config(valid_config)
        assert len(workflow.dataloaders) > 0
        assert len(workflow.pipelines) == 4
        assert len(workflow.pipelines[0].metrics) == 5
        assert workflow.n_jobs == 4
        assert workflow.trace_memory

    def test_no_n_jobs(self, valid_config):
        del valid_config['n_jobs']
        workflow = interpret_config(valid_config)
        assert len(workflow.dataloaders) > 0
        assert len(workflow.pipelines) == 4
        assert len(workflow.pipelines[0].metrics) == 5
        assert workflow.n_jobs == 1
        assert workflow.trace_memory

    def test_no_trace_memory(self, valid_config):
        del valid_config['trace_memory']
        workflow = interpret_config(valid_config)
        assert len(workflow.dataloaders) > 0
        assert len(workflow.pipelines) == 4
        assert len(workflow.pipelines[0].metrics) == 5
        assert workflow.n_jobs == 4
        assert not workflow.trace_memory

    def test_invalid_config(self, tmp_path):
        open(tmp_path / 'config.json', 'a').close()  # Make the file
        with pytest.raises(TypeError):
            interpret_config(str(tmp_path / 'config.json'))


class TestInterpretThresholds:

    def test_empty(self, valid_config):
        del valid_config['thresholds']
        valid_config['metrics'] = [  # Make sure there are only proba metrics
            {"type": "AreaUnderROC"}
        ]
        interpret_config(valid_config)  # No error

    def test_single_entry(self):
        thresholds = interpret_thresholds({'thresholds': {"type": "TopN", "n": 10}})
        assert isinstance(thresholds, list)
        assert len(thresholds) == 1
        assert isinstance(thresholds[0], thresholding.TopN)
        assert thresholds[0].n == 10

    def test_multiple_entries(self):
        thresholds = interpret_thresholds({
            'thresholds': [
                {"type": "TopN", "n": 10},
                {"type": "FixedCutoff", "cutoff": 0.5}
            ]
        })
        assert isinstance(thresholds, list)
        assert len(thresholds) == 2
        assert isinstance(thresholds[0], thresholding.TopN)
        assert thresholds[0].n == 10
        assert isinstance(thresholds[1], thresholding.FixedCutoff)
        assert thresholds[1].cutoff == 0.5


class TestInterpretDataloaders:

    def test_empty(self, valid_config):
        del valid_config['dataloaders']
        with pytest.raises(ValueError):
            interpret_config(valid_config)

    def test_single_entry(self):
        path = f"{DATA_PATH}/UCR-time-series-anomaly-archive/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
        dataloaders = interpret_dataloaders({"dataloaders": {"type": "UCRLoader", "path": path}})
        assert isinstance(dataloaders, list)
        assert len(dataloaders) == 1
        assert isinstance(dataloaders[0], data.UCRLoader)
        assert dataloaders[0].path == path

    def test_multiple_entries(self):
        path = f"{DATA_PATH}/UCR-time-series-anomaly-archive/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
        dataloaders = interpret_dataloaders({
            'dataloaders': [
                {"type": "UCRLoader", "path": path},
                {"type": "directory", "path": f"{DATA_PATH}/UCR-time-series-anomaly-archive", "base_type": "UCRLoader"}
            ]
        })
        assert isinstance(dataloaders, list)
        assert len(dataloaders) >= 1
        assert all(isinstance(loader, data.UCRLoader) for loader in dataloaders)
        assert all(loader.path.startswith(f'{DATA_PATH}/UCR-time-series-anomaly-archive') for loader in dataloaders)
        assert dataloaders[0].path == path

    def test_from_directory(self):
        dataloaders = data_entry({"type": "directory", "path": f"{DATA_PATH}/UCR-time-series-anomaly-archive", "base_type": "UCRLoader"})
        assert all(isinstance(loader, data.UCRLoader) for loader in dataloaders)
        assert all(loader.path.startswith(f'{DATA_PATH}/UCR-time-series-anomaly-archive') for loader in dataloaders)

    def test_from_directory_too_many_entries(self):
        with pytest.raises(ValueError):
            data_entry({"type": "directory", "path": f"{DATA_PATH}/UCR-time-series-anomaly-archive", "base_type": "UCRLoader", 'something-else': 0})

    def test_from_directory_no_path(self):
        with pytest.raises(TypeError):
            data_entry({"type": "directory", "base_type": "UCRLoader", 'some-replacement': 'to-have-proper-number-items'})

    def test_from_directory_no_base_type(self):
        with pytest.raises(TypeError):
            data_entry({"type": "directory", "path": f"{DATA_PATH}/UCR-time-series-anomaly-archive", 'some-replacement': 'to-have-proper-number-items'})

    def test_from_directory_invalid_base_type(self):
        with pytest.raises(ValueError):
            data_entry({"type": "directory", "path": f"{DATA_PATH}/UCR-time-series-anomaly-archive", "base_type": "INVALID"})


class TestInterpretMetrics:

    def test_empty(self, valid_config):
        del valid_config['metrics']
        with pytest.raises(ValueError):
            interpret_config(valid_config)

    def test_single_entry(self):
        metrics = interpret_metrics({'metrics': {"type": "Precision"}})
        assert isinstance(metrics, list)
        assert len(metrics) == 1
        assert isinstance(metrics[0], evaluation.Precision)

    def test_multiple_entries(self):
        metrics = interpret_metrics({'metrics': [{"type": "Precision"}, {"type": "Recall"}]})
        assert isinstance(metrics, list)
        assert len(metrics) == 2
        assert isinstance(metrics[0], evaluation.Precision)
        assert isinstance(metrics[1], evaluation.Recall)

    def test_threshold_metric_no_thresholder(self):
        with pytest.raises(ValueError):
            metric_entry({
                'type': 'ThresholdMetric',
                'no_thresholder': {'type': 'FixedCutoff'},
                'metric': {"type": "Precision"}
            })

    def test_threshold_metric_no_metric(self):
        with pytest.raises(ValueError):
            metric_entry({
                'type': 'ThresholdMetric',
                'thresholder': {'type': 'FixedCutoff'},
                'no_metric': {"type": "Precision"}
            })

    def test_best_threshold_metric_no_metric(self):
        with pytest.raises(ValueError):
            metric_entry({
                'type': 'BestThresholdMetric',
                'no_metric': {"type": "Precision"}
            })


class TestInterpretDetectors:

    def test_empty(self, valid_config):
        del valid_config['detectors']
        with pytest.raises(ValueError):
            interpret_config(valid_config)

    def test_single_entry(self):
        detectors = interpret_detectors({'detectors': {"type": "IsolationForest", 'window_size': 15}})
        assert isinstance(detectors, list)
        assert len(detectors) == 1
        assert isinstance(detectors[0], anomaly_detection.IsolationForest)
        assert detectors[0].window_size == 15

    def test_multiple_entries(self):
        detectors = interpret_detectors({
            'detectors': [
                {"type": "IsolationForest", 'window_size': 15},
                {"type": "MatrixProfileDetector", 'window_size': 25}
            ]
        })
        assert isinstance(detectors, list)
        assert len(detectors) == 2
        assert isinstance(detectors[0], anomaly_detection.IsolationForest)
        assert detectors[0].window_size == 15
        assert isinstance(detectors[1], anomaly_detection.MatrixProfileDetector)
        assert detectors[1].window_size == 25


class TestInterpretPreprocessors:

    def test_empty(self, valid_config):
        del valid_config['preprocessors']
        interpret_config(valid_config)  # No error

    def test_single_entry(self):
        preprocessors = interpret_preprocessing({'preprocessors': {"type": "MinMaxScaler"}})
        assert isinstance(preprocessors, list)
        assert len(preprocessors) == 1
        assert isinstance(preprocessors[0], preprocessing.MinMaxScaler)

    def test_multiple_entries(self):
        preprocessors = interpret_preprocessing({'preprocessors': [{"type": "MinMaxScaler"}, {'type': 'MovingAverage', 'window_size': 40}]})
        assert isinstance(preprocessors, list)
        assert len(preprocessors) == 2
        assert isinstance(preprocessors[0], preprocessing.MinMaxScaler)
        assert isinstance(preprocessors[1], preprocessing.MovingAverage)
        assert preprocessors[1].window_size == 40

    def test_chained_preprocessor_non_list_processor(self):
        with pytest.raises(ValueError):
            preprocessing_entry({
                'type': 'ChainedPreprocessor',
                'base_preprocessors': {"type": "MinMaxScaler"}
            })

    def test_chained_preprocessor_no_base_preprocessors(self):
        with pytest.raises(ValueError):
            preprocessing_entry({
                'type': 'ChainedPreprocessor',
                'no_base_preprocessors': [{"type": "MinMaxScaler"}, {'type': 'MovingAverage', 'window_size': 40}]
            })

    def test_chained_preprocessor(self):
        preprocessor = preprocessing_entry({
            'type': 'ChainedPreprocessor',
            'base_preprocessors': [{"type": "MinMaxScaler"}, {'type': 'MovingAverage', 'window_size': 40}]
        })
        assert isinstance(preprocessor, preprocessing.ChainedPreprocessor)
        assert len(preprocessor.base_preprocessors) == 2
        assert isinstance(preprocessor.base_preprocessors[0], preprocessing.MinMaxScaler)
        assert isinstance(preprocessor.base_preprocessors[1], preprocessing.MovingAverage)
        assert preprocessor.base_preprocessors[1].window_size == 40


@pytest.mark.parametrize("entry_function,object_type,entry", [
    # Thresholds
    (threshold_entry, thresholding.FixedCutoff, {'cutoff': 0.9}),
    (threshold_entry, thresholding.ContaminationRate, {'contamination_rate': 0.05}),
    (threshold_entry, thresholding.TopN, {'n': 10}),
    # Dataloaders
    (data_entry, data.UCRLoader, {"path": f"{DATA_PATH}/UCR-time-series-anomaly-archive/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"}),
    # Metrics
    (metric_entry, evaluation.Precision, {}),
    (metric_entry, evaluation.Recall, {}),
    (metric_entry, evaluation.FBeta, {}),
    (metric_entry, evaluation.FBeta, {'beta': 2.0}),
    (metric_entry, evaluation.AreaUnderROC, {}),
    (metric_entry, evaluation.AreaUnderPR, {}),
    (metric_entry, evaluation.PointAdjustedPrecision, {}),
    (metric_entry, evaluation.PointAdjustedRecall, {}),
    (metric_entry, evaluation.PointAdjustedFBeta, {}),
    (metric_entry, evaluation.PointAdjustedFBeta, {'beta': 2.0}),
    (metric_entry, evaluation.ThresholdMetric, {
        'thresholder': {"type": "FixedCutoff", 'cutoff': 0.9},
        'metric': {"type": "Precision"}
    }),
    (metric_entry, evaluation.BestThresholdMetric, {
        'metric': {"type": "Precision"}
    }),
    # Detectors
    (detector_entry, anomaly_detection.baselines.AlwaysNormal, {}),
    (detector_entry, anomaly_detection.baselines.AlwaysAnomalous, {}),
    (detector_entry, anomaly_detection.baselines.RandomDetector, {}),
    (detector_entry, anomaly_detection.baselines.RandomDetector, {'seed': 42}),
    (detector_entry, anomaly_detection.ClusterBasedLocalOutlierFactor, {'window_size': 10}),
    (detector_entry, anomaly_detection.CopulaBasedOutlierDetector, {'window_size': 10}),
    (detector_entry, anomaly_detection.HistogramBasedOutlierScore, {'window_size': 1}),
    (detector_entry, anomaly_detection.IsolationForest, {'window_size': 15}),
    (detector_entry, anomaly_detection.IsolationForest, {'window_size': 25, 'stride': 5}),
    (detector_entry, anomaly_detection.IsolationForest, {'window_size': 35, 'n_estimators': 100}),
    (detector_entry, anomaly_detection.KNearestNeighbors, {'window_size': 15}),
    (detector_entry, anomaly_detection.KNearestNeighbors, {'window_size': 25, 'stride': 100}),
    (detector_entry, anomaly_detection.KNearestNeighbors, {'window_size': 35, 'n_neighbors': 8}),
    (detector_entry, anomaly_detection.KShapeAnomalyDetector, {'window_size': 35}),
    (detector_entry, anomaly_detection.KShapeAnomalyDetector, {'window_size': 35, 'sequence_length_multiplier': 5}),
    (detector_entry, anomaly_detection.KShapeAnomalyDetector, {'window_size': 35, 'n_clusters': 8}),
    (detector_entry, anomaly_detection.KMeansAnomalyDetector, {'window_size': 25}),
    (detector_entry, anomaly_detection.KMeansAnomalyDetector, {'window_size': 35, 'n_clusters': 20}),
    (detector_entry, anomaly_detection.LocalOutlierFactor, {'window_size': 15}),
    (detector_entry, anomaly_detection.LocalOutlierFactor, {'window_size': 25, 'stride': 5}),
    (detector_entry, anomaly_detection.LocalOutlierFactor, {'window_size': 35, 'n_neighbors': 4}),
    (detector_entry, anomaly_detection.MatrixProfileDetector, {'window_size': 15}),
    (detector_entry, anomaly_detection.MatrixProfileDetector, {'window_size': 15, 'novelty': True}),
    (detector_entry, anomaly_detection.MatrixProfileDetector, {'window_size': 25, 'normalize': True, 'p': 1.5, 'k': 5}),
    (detector_entry, anomaly_detection.MedianMethod, {'neighborhood_size_before': 15}),
    (detector_entry, anomaly_detection.MedianMethod, {'neighborhood_size_before': 25, 'neighborhood_size_after': 5}),
    (detector_entry, anomaly_detection.OneClassSupportVectorMachine, {'window_size': 15}),
    (detector_entry, anomaly_detection.OneClassSupportVectorMachine, {'window_size': 15, 'kernel': 'poly'}),
    (detector_entry, anomaly_detection.PrincipalComponentAnalysis, {'window_size': 15}),
    (detector_entry, anomaly_detection.PrincipalComponentAnalysis, {'window_size': 15, 'n_components': 0.5}),
    (detector_entry, anomaly_detection.KernelPrincipalComponentAnalysis, {'window_size': 15}),
    (detector_entry, anomaly_detection.KernelPrincipalComponentAnalysis, {'window_size': 15, 'n_components': 0.5}),
    (detector_entry, anomaly_detection.RobustPrincipalComponentAnalysis, {'window_size': 15}),
    (detector_entry, anomaly_detection.RobustPrincipalComponentAnalysis, {'window_size': 15, 'max_iter': 100}),
    # Preprocessors
    (preprocessing_entry, preprocessing.Identity, {}),
    (preprocessing_entry, preprocessing.ChainedPreprocessor, {
        'base_preprocessors': [
            {"type": "MovingAverage", "window_size": 15},
            {"type": "Identity"}
        ]
    }),
    (preprocessing_entry, preprocessing.MinMaxScaler, {}),
    (preprocessing_entry, preprocessing.StandardScaler, {}),
    (preprocessing_entry, preprocessing.RobustScaler, {}),
    (preprocessing_entry, preprocessing.MovingAverage, {'window_size': 40}),
    (preprocessing_entry, preprocessing.ExponentialMovingAverage, {"alpha": 0.8}),
    (preprocessing_entry, preprocessing.NbSamplesUnderSampler, {'nb_samples': 250}),
    (preprocessing_entry, preprocessing.SamplingRateUnderSampler, {'sampling_rate': 5}),
    (preprocessing_entry, preprocessing.Differencing, {'order': 1}),
    (preprocessing_entry, preprocessing.PiecewiseAggregateApproximation, {'n': 32}),
])
class TestInterpretEntries:

    def test(self, entry_function, object_type, entry):
        entry_copy = entry.copy()
        entry_copy['type'] = object_type.__name__
        read_object = entry_function(entry_copy)
        assert isinstance(read_object, object_type)
        for key, value in entry.items():
            if key in ['base_preprocessors', 'metric', 'thresholder']:
                pass  # Simply assume it is correct, because test would become so difficult that errors can sneak in
            elif hasattr(read_object, key):
                assert getattr(read_object, key) == value
            elif hasattr(read_object, 'kwargs'):
                assert getattr(read_object, 'kwargs')[key] == value
            else:
                pytest.fail(f"Object should either have '{key}' as attribute, or have 'kwargs' as attribute, which in turn has '{key}' as attribute!")

    def test_additional_parameter(self, entry_function, object_type, entry):
        entry_copy = entry.copy()
        entry_copy['type'] = object_type.__name__
        entry_copy['some-other-random-parameter'] = 0
        with pytest.raises(TypeError):
            entry_function(entry_copy)

    def test_invalid_type(self, entry_function, object_type, entry):
        entry_copy = entry.copy()
        entry_copy['type'] = 'INVALID-TYPE'
        with pytest.raises(ValueError):
            entry_function(entry_copy)

    def test_no_type(self, entry_function, object_type, entry):
        with pytest.raises(KeyError):
            entry_function(entry)
            

@pytest.mark.parametrize("entry_function,object_type", [
    (threshold_entry, thresholding.FixedCutoff),
    (threshold_entry, thresholding.ContaminationRate),
    (threshold_entry, thresholding.TopN),
    # Dataloaders
    (data_entry, data.UCRLoader),
    # Metrics
    (metric_entry, evaluation.ThresholdMetric),
    (metric_entry, evaluation.BestThresholdMetric),
    # Detectors
    (detector_entry, anomaly_detection.ClusterBasedLocalOutlierFactor),
    (detector_entry, anomaly_detection.CopulaBasedOutlierDetector),
    (detector_entry, anomaly_detection.HistogramBasedOutlierScore),
    (detector_entry, anomaly_detection.IsolationForest),
    (detector_entry, anomaly_detection.KernelPrincipalComponentAnalysis),
    (detector_entry, anomaly_detection.KNearestNeighbors),
    (detector_entry, anomaly_detection.KMeansAnomalyDetector),
    (detector_entry, anomaly_detection.KShapeAnomalyDetector),
    (detector_entry, anomaly_detection.LocalOutlierFactor),
    (detector_entry, anomaly_detection.MatrixProfileDetector),
    (detector_entry, anomaly_detection.MedianMethod),
    (detector_entry, anomaly_detection.OneClassSupportVectorMachine),
    (detector_entry, anomaly_detection.PrincipalComponentAnalysis),
    (detector_entry, anomaly_detection.RobustPrincipalComponentAnalysis),
    # Preprocessors
    (preprocessing_entry, preprocessing.ChainedPreprocessor),
    (preprocessing_entry, preprocessing.MovingAverage),
    (preprocessing_entry, preprocessing.ExponentialMovingAverage),
    (preprocessing_entry, preprocessing.NbSamplesUnderSampler),
    (preprocessing_entry, preprocessing.SamplingRateUnderSampler),
    (preprocessing_entry, preprocessing.Differencing),
])
class TestEntriesWithObligatedParameters:

    def test(self, entry_function, object_type):
        with pytest.raises(TypeError):
            entry_function({'type': object_type.__name__})


class TestAdditionalInformation:

    def test(self):
        additional_inforamtion = interpret_additional_information({'n_jobs': 3, 'error_log_path': 'test', 'something_else': 5})
        assert len(additional_inforamtion) == 2
        assert 'n_jobs' in additional_inforamtion
        assert additional_inforamtion['n_jobs'] == 3
        assert 'error_log_path' in additional_inforamtion
        assert additional_inforamtion['error_log_path'] == 'test'
