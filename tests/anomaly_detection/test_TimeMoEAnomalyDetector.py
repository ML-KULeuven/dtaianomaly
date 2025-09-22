import builtins
import sys
import types

import numpy as np
import pytest

from dtaianomaly.anomaly_detection import Supervision, TimeMoEAnomalyDetector


class TestTimeMoEAnomalyDetector:

    def test_supervision(self):
        detector = TimeMoEAnomalyDetector(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert str(TimeMoEAnomalyDetector(5)) == "TimeMoEAnomalyDetector(window_size=5)"
        assert (
            str(TimeMoEAnomalyDetector("fft"))
            == "TimeMoEAnomalyDetector(window_size='fft')"
        )
        assert (
            str(TimeMoEAnomalyDetector(15, "TimeMoE-200M"))
            == "TimeMoEAnomalyDetector(window_size=15,model_path='TimeMoE-200M')"
        )
        assert (
            str(TimeMoEAnomalyDetector(25, batch_size=3))
            == "TimeMoEAnomalyDetector(window_size=25,batch_size=3)"
        )

    @pytest.mark.parametrize("model_path", ["TimeMoE-50M", "TimeMoE-200M"])
    def test_model_path_valid(self, model_path):
        detector = TimeMoEAnomalyDetector(window_size="fft", model_path=model_path)
        assert detector.model_path == model_path

    @pytest.mark.parametrize("model_path", [0, True, None, ["a", "list"]])
    def test_model_path_invalid_type(self, model_path):
        with pytest.raises(TypeError):
            TimeMoEAnomalyDetector(window_size="fft", model_path=model_path)

    @pytest.mark.parametrize("model_path", ["invalid"])
    def test_model_path_invalid_value(self, model_path):
        with pytest.raises(ValueError):
            TimeMoEAnomalyDetector(window_size="fft", model_path=model_path)

    @pytest.mark.parametrize("batch_size", [8, 16, 32])
    def test_batch_size_valid(self, batch_size):
        detector = TimeMoEAnomalyDetector(window_size="fft", batch_size=batch_size)
        assert detector.batch_size == batch_size

    @pytest.mark.parametrize("batch_size", ["8", 8.0])
    def test_batch_size_invalid_type(self, batch_size):
        with pytest.raises(TypeError):
            TimeMoEAnomalyDetector(window_size="fft", batch_size=batch_size)

    @pytest.mark.parametrize("batch_size", [0, -8])
    def test_batch_size_invalid_value(self, batch_size):
        with pytest.raises(ValueError):
            TimeMoEAnomalyDetector(window_size="fft", batch_size=batch_size)

    @pytest.mark.parametrize("prediction_length", [32, 16, 8])
    def test_prediction_length_valid(self, prediction_length):
        detector = TimeMoEAnomalyDetector(
            window_size=16, prediction_length=prediction_length
        )
        assert detector.prediction_length == prediction_length

    @pytest.mark.parametrize("prediction_length", ["32", 16.0, True])
    def test_prediction_length_invalid_type(self, prediction_length):
        with pytest.raises(TypeError):
            TimeMoEAnomalyDetector(window_size=16, prediction_length=prediction_length)

    @pytest.mark.parametrize("prediction_length", [0, -1, -16])
    def test_prediction_length_invalid_value(self, prediction_length):
        with pytest.raises(ValueError):
            TimeMoEAnomalyDetector(window_size=16, prediction_length=prediction_length)

    @pytest.mark.parametrize("normalize_sequences", [True, False])
    def test_do_fine_tuning_valid(self, normalize_sequences):
        detector = TimeMoEAnomalyDetector(
            window_size="fft", normalize_sequences=normalize_sequences
        )
        assert detector.normalize_sequences == normalize_sequences

    @pytest.mark.parametrize("normalize_sequences", [5, 1.0, "invalid"])
    def test_do_fine_tuning_invalid_type(self, normalize_sequences):
        with pytest.raises(TypeError):
            TimeMoEAnomalyDetector(
                window_size="fft", normalize_sequences=normalize_sequences
            )

    @pytest.mark.parametrize("min_std", [1e-5, 1e-3, 0.1, 1.0, 2])
    def test_min_std_valid(self, min_std):
        detector = TimeMoEAnomalyDetector(3, min_std=min_std)
        assert detector.min_std == min_std

    @pytest.mark.parametrize("min_std", ["0", True, None, ["a", "list"]])
    def test_min_std_invalid_type(self, min_std):
        with pytest.raises(TypeError):
            TimeMoEAnomalyDetector(3, min_std=min_std)

    @pytest.mark.parametrize("min_std", [0.0, -1e-16])
    def test_min_std_invalid_value(self, min_std):
        with pytest.raises(ValueError):
            TimeMoEAnomalyDetector(3, min_std=min_std)

    def test_raises_if_transformers_missing(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(Exception, match="Module 'transformers' is not available"):
            TimeMoEAnomalyDetector(15)

    def test_no_error_if_transformers_available(self, monkeypatch):
        sys.modules["transformers"] = types.ModuleType("transformers")
        TimeMoEAnomalyDetector(15)

        # cleanup (avoid side effects on other tests)
        del sys.modules["transformers"]


class TestGetBatchStarts:

    def test_simple(self):
        time_moe = TimeMoEAnomalyDetector(
            window_size=2, batch_size=4, prediction_length=1
        )
        time_moe.window_size_ = 2  # Only for testing purposed!
        starts = time_moe._get_batch_starts(11)

        assert len(starts) == 3
        assert np.array_equal(starts[0], [0, 1, 2, 3])
        assert np.array_equal(starts[1], [4, 5, 6, 7])
        assert np.array_equal(starts[2], [8])

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
        time_moe = TimeMoEAnomalyDetector(
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
