import builtins
import sys
import types

import pytest

from dtaianomaly.anomaly_detection import MOMENT, Supervision


def setup(mp):
    mp.setattr(sys, "version_info", (3, 11, 7, "final", 0))
    sys.modules["momentfm"] = types.ModuleType("momentfm")


def cleanup():
    del sys.modules["momentfm"]


class TestChronosAnomalyDetector:

    def test_supervision(self, monkeypatch):
        setup(monkeypatch)
        detector = MOMENT(1)
        assert detector.supervision == Supervision.UNSUPERVISED
        cleanup()

    def test_str(self, monkeypatch):
        setup(monkeypatch)
        assert str(MOMENT(5)) == "MOMENT(window_size=5)"
        assert str(MOMENT("fft")) == "MOMENT(window_size='fft')"
        assert str(MOMENT(15, "large")) == "MOMENT(window_size=15,model_size='large')"
        assert str(MOMENT(25, batch_size=3)) == "MOMENT(window_size=25,batch_size=3)"
        cleanup()

    def test_model_size_valid(self, monkeypatch):
        setup(monkeypatch)
        detector = MOMENT(window_size="fft", model_size="large")
        assert detector.model_size == "large"
        cleanup()

    def test_model_size_invalid_type(self, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(TypeError):
            MOMENT(window_size="fft", model_size=0)
        cleanup()

    def test_model_size_invalid_value(self, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(ValueError):
            MOMENT(window_size="fft", model_size="invalid")
        cleanup()

    def test_batch_size_valid(self, monkeypatch):
        setup(monkeypatch)
        detector = MOMENT(window_size="fft", batch_size=8)
        assert detector.batch_size == 8
        cleanup()

    def test_batch_size_invalid_type(self, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(TypeError):
            MOMENT(window_size="fft", batch_size=8.0)
        cleanup()

    def test_batch_size_invalid_value(self, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(ValueError):
            MOMENT(window_size="fft", batch_size=0)
        cleanup()

    def test_do_fine_tuning_valid(self, monkeypatch):
        setup(monkeypatch)
        detector = MOMENT(window_size="fft", do_fine_tuning=True)
        assert detector.do_fine_tuning is True
        cleanup()

    def test_do_fine_tuning_invalid_type(self, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(TypeError):
            MOMENT(window_size="fft", do_fine_tuning="invalid")
        cleanup()

    def test_learning_rate_valid(self, monkeypatch):
        setup(monkeypatch)
        detector = MOMENT(3, learning_rate=0.1)
        assert detector.learning_rate == 0.1
        cleanup()

    def test_learning_rate_invalid_type(self, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(TypeError):
            MOMENT(3, learning_rate="0")
        cleanup()

    def test_learning_rate_invalid_value(self, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(ValueError):
            MOMENT(3, learning_rate=0.0)
        cleanup()

    def test_nb_epochs_valid(self, monkeypatch):
        setup(monkeypatch)
        detector = MOMENT(window_size="fft", nb_epochs=8)
        assert detector.nb_epochs == 8
        cleanup()

    def test_nb_epochs_invalid_type(self, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(TypeError):
            MOMENT(window_size="fft", nb_epochs=8.0)
        cleanup()

    def test_nb_epochs_invalid_value(self, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(ValueError):
            MOMENT(window_size="fft", nb_epochs=0)
        cleanup()

    def test_raises_error_if_not_3_11(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 10, 7, "final", 0))
        with pytest.raises(EnvironmentError):
            MOMENT(15)

    def test_raises_if_momentfm_missing(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 11, 7, "final", 0))
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "momentfm":
                raise ImportError("No module named 'autogluon.timeseries'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(Exception, match="Module 'momentfm' is not available"):
            # call the constructor that runs your code
            MOMENT(15)

    def test_no_error_if_momentfm_available(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 11, 7, "final", 0))

        sys.modules["momentfm"] = types.ModuleType("momentfm")

        # should NOT raise
        MOMENT(15)  # just runs, no exception

        # cleanup (avoid side effects on other tests)
        del sys.modules["momentfm"]
