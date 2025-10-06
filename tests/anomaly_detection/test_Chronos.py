import builtins
import sys
import types

import pytest

from dtaianomaly.anomaly_detection import Chronos, Supervision
from dtaianomaly.anomaly_detection._Chronos import MODEL_PATHS


def setup():
    # create dummy parent package and submodule
    autogluon = types.ModuleType("autogluon")
    timeseries = types.ModuleType("autogluon.timeseries")
    autogluon.timeseries = timeseries

    # register both in sys.modules
    sys.modules["autogluon"] = autogluon
    sys.modules["autogluon.timeseries"] = timeseries


def cleanup():
    del sys.modules["autogluon"]
    del sys.modules["autogluon.timeseries"]


class TestChronos:

    def test_supervision(self):
        setup()
        detector = Chronos(1)
        assert detector.supervision == Supervision.UNSUPERVISED
        cleanup()

    def test_str(self):
        setup()
        assert str(Chronos(5)) == "Chronos(window_size=5)"
        assert str(Chronos("fft")) == "Chronos(window_size='fft')"
        assert str(Chronos(15, "large")) == "Chronos(window_size=15,model_path='large')"
        assert str(Chronos(25, batch_size=3)) == "Chronos(window_size=25,batch_size=3)"
        cleanup()

    @pytest.mark.parametrize("model_path", MODEL_PATHS)
    def test_model_path_valid(self, model_path):
        setup()
        detector = Chronos(window_size="fft", model_path=model_path)
        assert detector.model_path == model_path
        cleanup()

    @pytest.mark.parametrize("model_path", [0, True, None, ["a", "list"]])
    def test_model_path_invalid_type(self, model_path):
        setup()
        with pytest.raises(TypeError):
            Chronos(window_size="fft", model_path=model_path)
        cleanup()

    @pytest.mark.parametrize("model_path", ["invalid"])
    def test_model_path_invalid_value(self, model_path):
        setup()
        with pytest.raises(ValueError):
            Chronos(window_size="fft", model_path=model_path)
        cleanup()

    @pytest.mark.parametrize("batch_size", [8, 16, 32])
    def test_batch_size_valid(self, batch_size):
        setup()
        detector = Chronos(window_size="fft", batch_size=batch_size)
        assert detector.batch_size == batch_size
        cleanup()

    @pytest.mark.parametrize("batch_size", ["8", 8.0])
    def test_batch_size_invalid_type(self, batch_size):
        setup()
        with pytest.raises(TypeError):
            Chronos(window_size="fft", batch_size=batch_size)
        cleanup()

    @pytest.mark.parametrize("batch_size", [0, -8])
    def test_batch_size_invalid_value(self, batch_size):
        setup()
        with pytest.raises(ValueError):
            Chronos(window_size="fft", batch_size=batch_size)
        cleanup()

    @pytest.mark.parametrize("forecast_horizon", [32, 16, 8])
    def test_forecast_horizon_valid(self, forecast_horizon):
        setup()
        detector = Chronos(window_size=16, forecast_horizon=forecast_horizon)
        assert detector.forecast_horizon == forecast_horizon
        cleanup()

    @pytest.mark.parametrize("forecast_horizon", ["32", 16.0, True])
    def test_forecast_horizon_invalid_type(self, forecast_horizon):
        setup()
        with pytest.raises(TypeError):
            Chronos(window_size=16, forecast_horizon=forecast_horizon)
        cleanup()

    @pytest.mark.parametrize("forecast_horizon", [0, -1, -16])
    def test_forecast_horizon_invalid_value(self, forecast_horizon):
        setup()
        with pytest.raises(ValueError):
            Chronos(window_size=16, forecast_horizon=forecast_horizon)
        cleanup()

    @pytest.mark.parametrize("do_fine_tuning", [True, False])
    def test_do_fine_tuning_valid(self, do_fine_tuning):
        setup()
        detector = Chronos(window_size="fft", do_fine_tuning=do_fine_tuning)
        assert detector.do_fine_tuning == do_fine_tuning
        cleanup()

    @pytest.mark.parametrize("do_fine_tuning", [5, 1.0, "invalid"])
    def test_do_fine_tuning_invalid_type(self, do_fine_tuning):
        setup()
        with pytest.raises(TypeError):
            Chronos(window_size="fft", do_fine_tuning=do_fine_tuning)
        cleanup()

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
            Chronos(15)

    def test_no_error_if_autogluon_available(self, monkeypatch):
        setup()
        Chronos(15)  # just runs, no exception
        cleanup()
