import builtins
import sys
import types

import pytest

from dtaianomaly.anomaly_detection import ChronosAnomalyDetector, Supervision

_VALID_CHRONOS_MODEL_PATH = [
    "tiny",
    "mini",
    "small",
    "base",
    "large",
    "bolt_tiny",
    "bolt_mini",
    "bolt_small",
    "bolt_large",
]


class TestChronosAnomalyDetector:

    def test_supervision(self):
        detector = ChronosAnomalyDetector(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert str(ChronosAnomalyDetector(5)) == "ChronosAnomalyDetector(window_size=5)"
        assert (
            str(ChronosAnomalyDetector("fft"))
            == "ChronosAnomalyDetector(window_size='fft')"
        )
        assert (
            str(ChronosAnomalyDetector(15, "large"))
            == "ChronosAnomalyDetector(window_size=15,model_path='large')"
        )
        assert (
            str(ChronosAnomalyDetector(25, batch_size=3))
            == "ChronosAnomalyDetector(window_size=25,batch_size=3)"
        )

    @pytest.mark.parametrize("model_path", _VALID_CHRONOS_MODEL_PATH)
    def test_model_path_valid(self, model_path):
        detector = ChronosAnomalyDetector(window_size="fft", model_path=model_path)
        assert detector.model_path == model_path

    @pytest.mark.parametrize("model_path", [0, True, None, ["a", "list"]])
    def test_model_path_invalid_type(self, model_path):
        with pytest.raises(TypeError):
            ChronosAnomalyDetector(window_size="fft", model_path=model_path)

    @pytest.mark.parametrize("model_path", ["invalid"])
    def test_model_path_invalid_value(self, model_path):
        with pytest.raises(ValueError):
            ChronosAnomalyDetector(window_size="fft", model_path=model_path)

    @pytest.mark.parametrize("batch_size", [8, 16, 32])
    def test_batch_size_valid(self, batch_size):
        detector = ChronosAnomalyDetector(window_size="fft", batch_size=batch_size)
        assert detector.batch_size == batch_size

    @pytest.mark.parametrize("batch_size", ["8", 8.0])
    def test_batch_size_invalid_type(self, batch_size):
        with pytest.raises(TypeError):
            ChronosAnomalyDetector(window_size="fft", batch_size=batch_size)

    @pytest.mark.parametrize("batch_size", [0, -8])
    def test_batch_size_invalid_value(self, batch_size):
        with pytest.raises(ValueError):
            ChronosAnomalyDetector(window_size="fft", batch_size=batch_size)

    @pytest.mark.parametrize("forecast_horizon", [32, 16, 8])
    def test_forecast_horizon_valid(self, forecast_horizon):
        detector = ChronosAnomalyDetector(
            window_size=16, forecast_horizon=forecast_horizon
        )
        assert detector.forecast_horizon == forecast_horizon

    @pytest.mark.parametrize("forecast_horizon", ["32", 16.0, True])
    def test_forecast_horizon_invalid_type(self, forecast_horizon):
        with pytest.raises(TypeError):
            ChronosAnomalyDetector(window_size=16, forecast_horizon=forecast_horizon)

    @pytest.mark.parametrize("forecast_horizon", [0, -1, -16])
    def test_forecast_horizon_invalid_value(self, forecast_horizon):
        with pytest.raises(ValueError):
            ChronosAnomalyDetector(window_size=16, forecast_horizon=forecast_horizon)

    @pytest.mark.parametrize("do_fine_tuning", [True, False])
    def test_do_fine_tuning_valid(self, do_fine_tuning):
        detector = ChronosAnomalyDetector(
            window_size="fft", do_fine_tuning=do_fine_tuning
        )
        assert detector.do_fine_tuning == do_fine_tuning

    @pytest.mark.parametrize("do_fine_tuning", [5, 1.0, "invalid"])
    def test_do_fine_tuning_invalid_type(self, do_fine_tuning):
        with pytest.raises(TypeError):
            ChronosAnomalyDetector(window_size="fft", do_fine_tuning=do_fine_tuning)

    def test_raises_if_autogluon_missing(self, monkeypatch):
        # simulate ImportError when trying to import autogluon.timeseries
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "autogluon.timeseries":
                raise ImportError("No module named 'autogluon.timeseries'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(
            Exception, match="Module 'autogluon.timeseries' is not available"
        ):
            # call the constructor that runs your code
            ChronosAnomalyDetector(15)

    def test_no_error_if_autogluon_available(self, monkeypatch):
        # simulate a dummy autogluon.timeseries module
        sys.modules["autogluon.timeseries"] = types.ModuleType("autogluon.timeseries")

        # should NOT raise
        ChronosAnomalyDetector(15)  # just runs, no exception

        # cleanup (avoid side effects on other tests)
        del sys.modules["autogluon.timeseries"]
