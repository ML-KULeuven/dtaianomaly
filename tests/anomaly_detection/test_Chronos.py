import builtins
import sys
import types

import pytest

from dtaianomaly.anomaly_detection import Chronos, Supervision


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

    def test_model_path_valid(self):
        setup()
        detector = Chronos(window_size="fft", model_path="large")
        assert detector.model_path == "large"
        cleanup()

    def test_model_path_invalid_type(self):
        setup()
        with pytest.raises(TypeError):
            Chronos(window_size="fft", model_path=0)
        cleanup()

    def test_model_path_invalid_value(self):
        setup()
        with pytest.raises(ValueError):
            Chronos(window_size="fft", model_path="invalid")
        cleanup()

    def test_batch_size_valid(self):
        setup()
        detector = Chronos(window_size="fft", batch_size=8)
        assert detector.batch_size == 8
        cleanup()

    def test_batch_size_invalid_type(self):
        setup()
        with pytest.raises(TypeError):
            Chronos(window_size="fft", batch_size=8.0)
        cleanup()

    def test_batch_size_invalid_value(self):
        setup()
        with pytest.raises(ValueError):
            Chronos(window_size="fft", batch_size=0)
        cleanup()

    def test_forecast_horizon_valid(self):
        setup()
        detector = Chronos(window_size=16, forecast_horizon=8)
        assert detector.forecast_horizon == 8
        cleanup()

    def test_forecast_horizon_invalid_type(self):
        setup()
        with pytest.raises(TypeError):
            Chronos(window_size=16, forecast_horizon=16.0)
        cleanup()

    def test_forecast_horizon_invalid_value(self):
        setup()
        with pytest.raises(ValueError):
            Chronos(window_size=16, forecast_horizon=0)
        cleanup()

    def test_do_fine_tuning_valid(self):
        setup()
        detector = Chronos(window_size="fft", do_fine_tuning=True)
        assert detector.do_fine_tuning is True
        cleanup()

    def test_do_fine_tuning_invalid_type(self):
        setup()
        with pytest.raises(TypeError):
            Chronos(window_size="fft", do_fine_tuning="invalid")
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
