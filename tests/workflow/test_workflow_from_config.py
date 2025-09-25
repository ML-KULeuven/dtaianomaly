import inspect
import json
import pathlib
import sys
import types

import pytest
import toml

from dtaianomaly import (
    anomaly_detection,
    data,
    evaluation,
    in_time_ad,
    preprocessing,
    thresholding,
    utils,
)
from dtaianomaly.workflow import Workflow, interpret_config, workflow_from_config
from dtaianomaly.workflow._workflow_from_config import (
    _interpret_additional_information,
    _interpret_config,
    _interpret_entry,
)

DATA_PATH = f"{pathlib.Path(__file__).parent.parent.parent}/data"
ALL_CLASSES = utils.all_classes(
    return_names=False, exclude_types=in_time_ad.CustomDetectorVisualizer
)


@pytest.fixture
def valid_config():
    return {
        "dataloaders": [
            {
                "type": "UCRLoader",
                "path": f"{DATA_PATH}/UCR-time-series-anomaly-archive/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt",
            },
            {
                "type": "from_directory",
                "path": f"{DATA_PATH}/UCR-time-series-anomaly-archive",
                "base_type": "UCRLoader",
            },
        ],
        "metrics": [
            {"type": "Precision"},
            {"type": "Recall"},
            {"type": "AreaUnderROC"},
        ],
        "thresholds": [
            {"type": "TopNThreshold", "n": 10},
            {"type": "FixedCutoffThreshold", "cutoff": 0.5},
        ],
        "preprocessors": [
            {"type": "MovingAverage", "window_size": 15},
            {"type": "Identity"},
        ],
        "detectors": [
            {"type": "IsolationForest", "window_size": 50},
            {"type": "MatrixProfileDetector", "window_size": 50},
        ],
        "n_jobs": 4,
        "trace_memory": True,
    }


def infer_minimal_entry(cls):
    kwargs = {
        "window_size": 15,
        "neighborhood_size_before": 15,
        "detector": {"type": "IsolationForest", "window_size": 15},
        "preprocessor": preprocessing.Identity(),
        "order": 2,
        "alpha": 0.7,
        "nb_samples": 500,
        "sampling_rate": 2,
        "n": 4,
        "metric": {"type": "Precision"},
        "thresholder": {"type": "FixedCutoffThreshold", "cutoff": 0.9},
        "cutoff": 0.9,
        "contamination_rate": 0.1,
        "base_preprocessors": [{"type": "Identity"}],
        "path": DATA_PATH,
        "test_path": DATA_PATH,
        "train_path": DATA_PATH,
        "neighborhood": 20,
        "moving_average_window_size": 5,
        "decoder_dimensions": [64, 32, 16],
    }
    sig = inspect.signature(cls.__init__)
    accepted_params = set(
        [
            p.name
            for p in sig.parameters.values()
            if p.name != "self" and p.default is inspect.Parameter.empty
        ]
    )
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    return {"type": cls.__name__} | filtered_kwargs


def infer_extensive_entry(cls):
    minimal_entry = infer_minimal_entry(cls)
    sig = inspect.signature(cls.__init__)
    optional_parameters = (
        set(sig.parameters) - {"self", "kwargs", "args"} - set(minimal_entry.keys())
    )
    return minimal_entry | {
        parameter: sig.parameters[parameter].default
        for parameter in optional_parameters
    }


class TestWorkflowFromConfig:

    def test_non_str_path(self):
        with pytest.raises(TypeError):
            workflow_from_config({"detector": "IsolationForest"})

    def test_non_existing_file(self):
        with pytest.raises(FileNotFoundError):
            workflow_from_config("path-to-non-existing-file.json")

    def test_non_json_file(self, tmp_path):
        open(tmp_path / "config.txt", "a").close()  # Make the file
        with pytest.raises(ValueError):
            workflow_from_config(str(tmp_path / "config.txt"))

    def test_file_too_large(self, tmp_path, valid_config):
        with open(tmp_path / "config.json", "a") as file:
            valid_config["detectors"].extend(
                [{"type": "IsolationForest", "window_size": w} for w in range(1, 10)]
            )  # Make the file big
            json.dump(valid_config, file)
        workflow_from_config(
            str(tmp_path / "config.json"), 1000000
        )  # Does not throw an error
        with pytest.raises(ValueError):
            workflow_from_config(str(tmp_path / "config.json"), 1)

    def test_success_json(self, tmp_path, valid_config):
        with open(tmp_path / "config.json", "a") as file:
            json.dump(valid_config, file)
        workflow = workflow_from_config(str(tmp_path / "config.json"))
        assert isinstance(workflow, Workflow)

    def test_success_toml(self, tmp_path, valid_config):
        with open(tmp_path / "config.toml", "w") as file:
            toml.dump(valid_config, file)
        workflow = workflow_from_config(str(tmp_path / "config.toml"))
        assert isinstance(workflow, Workflow)


class TestInterpretConfig:

    def test(self, valid_config):
        workflow = interpret_config(valid_config)
        assert len(workflow.dataloaders) > 0
        assert len(workflow.detectors) * len(workflow.preprocessors) == 4
        assert len(workflow.jobs) == 4 * len(workflow.dataloaders)
        assert len(workflow.metrics) == 5
        assert workflow.n_jobs == 4
        assert workflow.trace_memory

    def test_no_n_jobs(self, valid_config):
        del valid_config["n_jobs"]
        workflow = interpret_config(valid_config)
        assert len(workflow.dataloaders) > 0
        assert len(workflow.detectors) * len(workflow.preprocessors) == 4
        assert len(workflow.jobs) == 4 * len(workflow.dataloaders)
        assert len(workflow.metrics) == 5
        assert workflow.n_jobs == 1
        assert workflow.trace_memory

    def test_no_trace_memory(self, valid_config):
        del valid_config["trace_memory"]
        workflow = interpret_config(valid_config)
        assert len(workflow.dataloaders) > 0
        assert len(workflow.detectors) * len(workflow.preprocessors) == 4
        assert len(workflow.jobs) == 4 * len(workflow.dataloaders)
        assert len(workflow.metrics) == 5
        assert workflow.n_jobs == 4
        assert not workflow.trace_memory

    def test_invalid_config(self, tmp_path):
        open(tmp_path / "config.json", "a").close()  # Make the file
        with pytest.raises(TypeError):
            interpret_config(str(tmp_path / "config.json"))


class Test_InterpretConfig:

    @pytest.mark.parametrize(
        "name,cls",
        [
            ("dataloaders", data.DemonstrationTimeSeriesLoader),
            ("preprocessors", preprocessing.StandardScaler),
            ("detectors", anomaly_detection.IsolationForest),
            ("metrics", evaluation.Precision),
            ("thresholds", thresholding.FixedCutoffThreshold),
        ],
    )
    @pytest.mark.parametrize("required", [True, False])
    def test_valid(self, name, cls, required):
        entry = infer_minimal_entry(cls)
        loaded_entry = _interpret_entry(entry)
        loaded_config = _interpret_config(name, {name: entry}, required)
        assert str(loaded_entry) == str(loaded_config[0])

    @pytest.mark.parametrize(
        "name,clss",
        [
            ("dataloaders", [data.DemonstrationTimeSeriesLoader, data.UCRLoader]),
            (
                "preprocessors",
                [preprocessing.StandardScaler, preprocessing.MinMaxScaler],
            ),
            (
                "detectors",
                [
                    anomaly_detection.IsolationForest,
                    anomaly_detection.LocalOutlierFactor,
                ],
            ),
            ("metrics", [evaluation.Precision, evaluation.FBeta]),
            (
                "thresholds",
                [thresholding.FixedCutoffThreshold, thresholding.TopNThreshold],
            ),
        ],
    )
    @pytest.mark.parametrize("required", [True, False])
    def test_valid_list(self, name, clss, required):
        config = {name: [infer_minimal_entry(cls) for cls in clss]}
        loaded_config = _interpret_config(name, config, required)
        assert len(loaded_config) == len(clss)
        for i in range(len(clss)):
            loaded_entry = _interpret_entry(infer_minimal_entry(clss[i]))
            assert str(loaded_entry) == str(loaded_config[i])

    def test_name_not_in_config(self):
        with pytest.raises(ValueError):
            _interpret_config("something", {"something-else": {}}, True)

    def test_name_not_in_config_not_required(self):
        assert _interpret_config("something", {"something-else": {}}, False) is None


class TestInterpretEntry:

    @pytest.mark.parametrize(
        "infer_entry", [infer_minimal_entry, infer_extensive_entry]
    )
    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test(self, cls, infer_entry, monkeypatch):
        if cls == anomaly_detection.MOMENTAnomalyDetector:
            monkeypatch.setattr(sys, "version_info", (3, 10, 7, "final", 0))
            sys.modules["momentfm"] = types.ModuleType("momentfm")

        entry = infer_entry(cls)
        read_object = _interpret_entry(entry)
        assert isinstance(read_object, cls)
        for key, value in entry.items():
            if key == "type":
                continue  # Skip this key as it is only necessary for the configuration
            elif key == "base_preprocessors":
                assert list(map(str, getattr(read_object, key))) == list(
                    map(str, map(_interpret_entry, value))
                )
            elif isinstance(value, dict):
                assert str(getattr(read_object, key)) == str(_interpret_entry(value))
            elif hasattr(read_object, key):
                assert getattr(read_object, key) == value
            else:
                pytest.fail(
                    f"Object should either have '{key}' as attribute, or have 'kwargs' as attribute, which in turn has '{key}' as attribute!"
                )

        if cls == anomaly_detection.MOMENTAnomalyDetector:
            del sys.modules["momentfm"]

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_invalid_parameter(self, cls):
        minimal_entry = infer_minimal_entry(cls)
        minimal_entry["some-other-random-parameter"] = 0
        with pytest.raises(TypeError):
            _interpret_entry(minimal_entry)

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_missing_obligated_parameters(self, cls):
        minimal_entry = infer_minimal_entry(cls)
        if len(minimal_entry) > 1:
            with pytest.raises(TypeError):
                _interpret_entry({"type": cls.__name__})

    @pytest.mark.parametrize("config", [{}, {"type": "INVALID-TYPE"}])
    def test_invalid_type(self, config):
        with pytest.raises(ValueError):
            _interpret_entry(config)


class TestAdditionalInformation:

    @staticmethod
    def additional_parameters():
        parameters = list(inspect.signature(Workflow.__init__).parameters.values())
        to_skip = [
            "self",
            "dataloaders",
            "metrics",
            "detectors",
            "preprocessors",
            "thresholds",
        ]
        return [param for param in parameters if param.name not in to_skip]

    @pytest.mark.parametrize("param", additional_parameters())
    @pytest.mark.parametrize("add_invalid", [True, False])
    def test(self, param, add_invalid):
        config = {param.name: param.default}
        if add_invalid:
            config["something-invalid"] = 0
        result = _interpret_additional_information(config)
        assert len(result) == 1
        assert param.name in result
        assert result[param.name] == param.default

    @pytest.mark.parametrize("param1", additional_parameters())
    @pytest.mark.parametrize("param2", additional_parameters())
    @pytest.mark.parametrize("add_invalid", [True, False])
    def test_two_params(self, param1, param2, add_invalid):
        if param1.name == param2.name:
            return
        config = {param1.name: param1.default, param2.name: param2.default}
        if add_invalid:
            config["something-invalid"] = 0
        result = _interpret_additional_information(config)
        assert len(result) == 2
        assert param1.name in result
        assert result[param1.name] == param1.default
        assert param2.name in result
        assert result[param2.name] == param2.default

    @pytest.mark.parametrize("nb_invalid", [1, 3, 10])
    def test_only_invalid(self, nb_invalid):
        config = {f"something-invalid-{i}": i for i in range(nb_invalid)}
        result = _interpret_additional_information(config)
        assert len(result) == 0
