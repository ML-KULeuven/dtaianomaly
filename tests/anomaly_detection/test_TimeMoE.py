import builtins
import sys
import types

import numpy as np
import pytest

from dtaianomaly.anomaly_detection import Supervision, TimeMoE
from dtaianomaly.anomaly_detection._TimeMoE import MODEL_PATHS


def setup():
    sys.modules["transformers"] = types.ModuleType("transformers")


def cleanup():
    del sys.modules["transformers"]


class TestTimeMoE:

    def test_supervision(self):
        setup()
        detector = TimeMoE(1)
        assert detector.supervision == Supervision.UNSUPERVISED
        cleanup()

    def test_str(self):
        setup()
        assert str(TimeMoE(5)) == "TimeMoE(window_size=5)"
        assert str(TimeMoE("fft")) == "TimeMoE(window_size='fft')"
        assert (
            str(TimeMoE(15, "TimeMoE-200M"))
            == "TimeMoE(window_size=15,model_path='TimeMoE-200M')"
        )
        assert str(TimeMoE(25, batch_size=3)) == "TimeMoE(window_size=25,batch_size=3)"
        cleanup()

    @pytest.mark.parametrize("model_path", MODEL_PATHS)
    def test_model_path_valid(self, model_path):
        setup()
        detector = TimeMoE(window_size="fft", model_path=model_path)
        assert detector.model_path == model_path
        cleanup()

    @pytest.mark.parametrize("model_path", [0, True, None, ["a", "list"]])
    def test_model_path_invalid_type(self, model_path):
        setup()
        with pytest.raises(TypeError):
            TimeMoE(window_size="fft", model_path=model_path)
        cleanup()

    @pytest.mark.parametrize("model_path", ["invalid"])
    def test_model_path_invalid_value(self, model_path):
        setup()
        with pytest.raises(ValueError):
            TimeMoE(window_size="fft", model_path=model_path)
        cleanup()

    @pytest.mark.parametrize("batch_size", [8, 16, 32])
    def test_batch_size_valid(self, batch_size):
        setup()
        detector = TimeMoE(window_size="fft", batch_size=batch_size)
        assert detector.batch_size == batch_size
        cleanup()

    @pytest.mark.parametrize("batch_size", ["8", 8.0])
    def test_batch_size_invalid_type(self, batch_size):
        setup()
        with pytest.raises(TypeError):
            TimeMoE(window_size="fft", batch_size=batch_size)
        cleanup()

    @pytest.mark.parametrize("batch_size", [0, -8])
    def test_batch_size_invalid_value(self, batch_size):
        setup()
        with pytest.raises(ValueError):
            TimeMoE(window_size="fft", batch_size=batch_size)
        cleanup()

    @pytest.mark.parametrize("prediction_length", [32, 16, 8])
    def test_prediction_length_valid(self, prediction_length):
        setup()
        detector = TimeMoE(window_size=16, prediction_length=prediction_length)
        assert detector.prediction_length == prediction_length
        cleanup()

    @pytest.mark.parametrize("prediction_length", ["32", 16.0, True])
    def test_prediction_length_invalid_type(self, prediction_length):
        setup()
        with pytest.raises(TypeError):
            TimeMoE(window_size=16, prediction_length=prediction_length)
        cleanup()

    @pytest.mark.parametrize("prediction_length", [0, -1, -16])
    def test_prediction_length_invalid_value(self, prediction_length):
        setup()
        with pytest.raises(ValueError):
            TimeMoE(window_size=16, prediction_length=prediction_length)
        cleanup()

    @pytest.mark.parametrize("normalize_sequences", [True, False])
    def test_do_fine_tuning_valid(self, normalize_sequences):
        setup()
        detector = TimeMoE(window_size="fft", normalize_sequences=normalize_sequences)
        assert detector.normalize_sequences == normalize_sequences
        cleanup()

    @pytest.mark.parametrize("normalize_sequences", [5, 1.0, "invalid"])
    def test_do_fine_tuning_invalid_type(self, normalize_sequences):
        setup()
        with pytest.raises(TypeError):
            TimeMoE(window_size="fft", normalize_sequences=normalize_sequences)
        cleanup()

    @pytest.mark.parametrize("min_std", [1e-5, 1e-3, 0.1, 1.0, 2.0])
    def test_min_std_valid(self, min_std):
        setup()
        detector = TimeMoE(3, min_std=min_std)
        assert detector.min_std == min_std
        cleanup()

    @pytest.mark.parametrize("min_std", ["0", True, None, ["a", "list"]])
    def test_min_std_invalid_type(self, min_std):
        setup()
        with pytest.raises(TypeError):
            TimeMoE(3, min_std=min_std)
        cleanup()

    @pytest.mark.parametrize("min_std", [-1e-16])
    def test_min_std_invalid_value(self, min_std):
        setup()
        with pytest.raises(ValueError):
            TimeMoE(3, min_std=min_std)
        cleanup()

    def test_raises_if_transformers_missing(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(Exception, match="Module 'transformers' is not available"):
            TimeMoE(15)

    def test_no_error_if_transformers_available(self, monkeypatch):
        sys.modules["transformers"] = types.ModuleType("transformers")
        TimeMoE(15)

        # cleanup (avoid side effects on other tests)
        del sys.modules["transformers"]


class TestGetBatchStarts:

    def test_simple(self):
        setup()
        time_moe = TimeMoE(window_size=2, batch_size=4, prediction_length=1)
        time_moe.window_size_ = 2  # Only for testing purposed!
        starts = time_moe._get_batch_starts(11)

        assert len(starts) == 3
        assert np.array_equal(starts[0], [0, 1, 2, 3])
        assert np.array_equal(starts[1], [4, 5, 6, 7])
        assert np.array_equal(starts[2], [8])
        cleanup()

    @pytest.mark.parametrize(
        "batch_size,window_size,length_time_series,prediction_length",
        [
            (4, 2, 11, 1),
            (16, 4, 100, 1),
            (16, 4, 100, 5),
            (17, 7, 101, 5),  # Annoying numbers
        ],
    )
    def test(self, batch_size, window_size, length_time_series, prediction_length):
        setup()
        time_moe = TimeMoE(
            window_size=window_size,
            batch_size=batch_size,
            prediction_length=prediction_length,
        )
        time_moe.window_size_ = window_size  # Only for testing purposed!
        starts = time_moe._get_batch_starts(length_time_series)

        total_nb_samples = 0
        for i, batch in enumerate(starts):
            if i != len(starts) - 1:
                assert len(batch) == batch_size
            assert np.array_equal(
                batch, np.arange(batch_size * i, batch_size * i + len(batch))
            )
            total_nb_samples += len(batch)

        assert (
            total_nb_samples == length_time_series - window_size - prediction_length + 1
        )
        cleanup()
