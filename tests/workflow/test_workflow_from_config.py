
import pytest
import inspect
import json
import toml
import pathlib

from dtaianomaly import preprocessing, anomaly_detection, evaluation, thresholding, data, utils
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

    def test_with_data_root(self, valid_config):
        config = valid_config.copy()
        config['data_root'] = DATA_PATH
        config['dataloaders'] = [
            {"type": "UCRLoader", "path": f"UCR-time-series-anomaly-archive/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"},
            {"type": "directory", "path": f"UCR-time-series-anomaly-archive", "base_type": "UCRLoader"}
        ]
        self.test(config)


class TestInterpretThresholds:

    def test_empty(self, valid_config):
        del valid_config['thresholds']
        valid_config['metrics'] = [  # Make sure there are only proba metrics
            {"type": "AreaUnderROC"}
        ]
        interpret_config(valid_config)  # No error

    def test_single_entry(self):
        thresholds = interpret_thresholds({"type": "TopN", "n": 10})
        assert isinstance(thresholds, list)
        assert len(thresholds) == 1
        assert isinstance(thresholds[0], thresholding.TopN)
        assert thresholds[0].n == 10

    def test_multiple_entries(self):
        thresholds = interpret_thresholds([
            {"type": "TopN", "n": 10},
            {"type": "FixedCutoff", "cutoff": 0.5}
        ])
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
        dataloaders = interpret_dataloaders({"type": "UCRLoader", "path": path})
        assert isinstance(dataloaders, list)
        assert len(dataloaders) == 1
        assert isinstance(dataloaders[0], data.UCRLoader)
        assert dataloaders[0].path == path

    def test_multiple_entries(self):
        path = f"{DATA_PATH}/UCR-time-series-anomaly-archive/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
        dataloaders = interpret_dataloaders([
            {"type": "UCRLoader", "path": path},
            {"type": "directory", "path": f"{DATA_PATH}/UCR-time-series-anomaly-archive", "base_type": "UCRLoader"}
        ])
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

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            data_entry({"type": "INVALID"})


class TestInterpretMetrics:

    def test_empty(self, valid_config):
        del valid_config['metrics']
        with pytest.raises(ValueError):
            interpret_config(valid_config)

    def test_single_entry(self):
        metrics = interpret_metrics({"type": "Precision"})
        assert isinstance(metrics, list)
        assert len(metrics) == 1
        assert isinstance(metrics[0], evaluation.Precision)

    def test_multiple_entries(self):
        metrics = interpret_metrics([{"type": "Precision"}, {"type": "Recall"}])
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
        with pytest.raises(TypeError):
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
        detectors = interpret_detectors({"type": "IsolationForest", 'window_size': 15})
        assert isinstance(detectors, list)
        assert len(detectors) == 1
        assert isinstance(detectors[0], anomaly_detection.IsolationForest)
        assert detectors[0].window_size == 15

    def test_multiple_entries(self):
        detectors = interpret_detectors([
            {"type": "IsolationForest", 'window_size': 15},
            {"type": "MatrixProfileDetector", 'window_size': 25}
        ])
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
        preprocessors = interpret_preprocessing({"type": "MinMaxScaler"})
        assert isinstance(preprocessors, list)
        assert len(preprocessors) == 1
        assert isinstance(preprocessors[0], preprocessing.MinMaxScaler)

    def test_multiple_entries(self):
        preprocessors = interpret_preprocessing([{"type": "MinMaxScaler"}, {'type': 'MovingAverage', 'window_size': 40}])
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
        with pytest.raises(TypeError):
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


class TestAdditionalInformation:

    def test(self):
        additional_information = interpret_additional_information({'n_jobs': 3, 'error_log_path': 'test', 'something_else': 5})
        assert len(additional_information) == 2
        assert 'n_jobs' in additional_information
        assert additional_information['n_jobs'] == 3
        assert 'error_log_path' in additional_information
        assert additional_information['error_log_path'] == 'test'


def infer_entry_function(cls):
    if issubclass(cls, anomaly_detection.BaseDetector):
        return detector_entry
    elif issubclass(cls, data.LazyDataLoader):
        return data_entry
    elif issubclass(cls, evaluation.Metric):
        return metric_entry
    elif issubclass(cls, thresholding.Thresholding):
        return threshold_entry
    elif issubclass(cls, preprocessing.Preprocessor):
        return preprocessing_entry
    else:
        pytest.fail(f"An invalid object is given: {cls}")


def infer_minimal_entry(cls):
    kwargs = {
        'window_size': 15,
        'neighborhood_size_before': 15,
        'detector': {'type': 'IsolationForest', 'window_size': 15},
        'preprocessor': preprocessing.Identity(),
        'order': 2,
        'alpha': 0.7,
        'nb_samples': 500,
        'sampling_rate': 2,
        'n': 4,
        'metric': {'type': 'Precision'},
        'thresholder': {'type': 'FixedCutoff', 'cutoff': 0.9},
        'cutoff': 0.9,
        'contamination_rate': 0.1,
        'base_preprocessors': [{'type': 'Identity'}],
        'path': DATA_PATH
    }
    sig = inspect.signature(cls.__init__)
    accepted_params = set([
        p.name for p in sig.parameters.values()
        if p.name != 'self' and p.default is inspect.Parameter.empty
    ])
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    return {'type': cls.__name__} | filtered_kwargs


def infer_extensive_entry(cls):
    minimal_entry = infer_minimal_entry(cls)
    sig = inspect.signature(cls.__init__)
    optional_parameters = set(sig.parameters) - {"self", "kwargs", "args"} - set(minimal_entry.keys())
    return minimal_entry | {parameter: sig.parameters[parameter].default for parameter in optional_parameters}


@pytest.mark.parametrize("cls", utils.all_classes(return_names=False))
class TestInterpretEntries:

    @pytest.mark.parametrize("infer_entry", [infer_minimal_entry, infer_extensive_entry])
    def test(self, cls, infer_entry):
        entry_function = infer_entry_function(cls)
        entry = infer_entry(cls)
        read_object = entry_function(entry)
        assert isinstance(read_object, cls)
        for key, value in entry.items():
            if key in ['base_preprocessors', 'metric', 'thresholder', 'detector']:
                continue  # Simply assume it is correct, because test would become so difficult that errors can sneak in
            elif key == 'type':
                continue  # Skip this key as it is only necessary for the configuration
            elif hasattr(read_object, key):
                assert getattr(read_object, key) == value
            elif hasattr(read_object, 'kwargs'):
                assert getattr(read_object, 'kwargs')[key] == value
            else:
                pytest.fail(f"Object should either have '{key}' as attribute, or have 'kwargs' as attribute, which in turn has '{key}' as attribute!")

    def test_invalid_parameter(self, cls):
        entry_function = infer_entry_function(cls)
        minimal_entry = infer_minimal_entry(cls)
        minimal_entry['some-other-random-parameter'] = 0
        with pytest.raises(TypeError):
            entry_function(minimal_entry)

    def test_missing_obligated_parameters(self, cls):
        entry_function = infer_entry_function(cls)
        minimal_entry = infer_minimal_entry(cls)
        if len(minimal_entry) > 1:
            print(minimal_entry)
            with pytest.raises(TypeError):
                entry_function({'type': cls.__name__})


@pytest.mark.parametrize("entry_function", [threshold_entry, metric_entry, metric_entry, detector_entry, preprocessing_entry])
class TestInvalidEntries:

    def test_invalid_type(self, entry_function):
        with pytest.raises(ValueError):
            entry_function({'type': 'INVALID-TYPE'})

    def test_no_type(self, entry_function):
        with pytest.raises(KeyError):
            entry_function({})
