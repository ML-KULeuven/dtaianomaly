
import builtins
import sys
import types

import pytest

from dtaianomaly.anomaly_detection import MOMENTAnomalyDetector, Supervision


def setup(mp):
    mp.setattr(sys, "version_info", (3, 10, 7, "final", 0))
    sys.modules["momentfm"] = types.ModuleType("momentfm")


def cleanup():
    del sys.modules["momentfm"]


class TestChronosAnomalyDetector:

    def test_supervision(self, monkeypatch):
        setup(monkeypatch)
        detector = MOMENTAnomalyDetector(1)
        assert detector.supervision == Supervision.UNSUPERVISED
        cleanup()

    def test_str(self, monkeypatch):
        setup(monkeypatch)
        assert str(MOMENTAnomalyDetector(5)) == "MOMENTAnomalyDetector(window_size=5)"
        assert str(MOMENTAnomalyDetector('fft')) == "MOMENTAnomalyDetector(window_size='fft')"
        assert str(MOMENTAnomalyDetector(15, 'large')) == "MOMENTAnomalyDetector(window_size=15,model_size='large')"
        assert str(MOMENTAnomalyDetector(25, batch_size=3)) == "MOMENTAnomalyDetector(window_size=25,batch_size=3)"
        cleanup()

    @pytest.mark.parametrize('model_size', ['small', 'base', 'large'])
    def test_model_size_valid(self, model_size, monkeypatch):
        setup(monkeypatch)
        detector = MOMENTAnomalyDetector(window_size='fft', model_size=model_size)
        assert detector.model_size == model_size
        cleanup()

    @pytest.mark.parametrize('model_size', [0, True, None, ['a', 'list']])
    def test_model_size_invalid_type(self, model_size, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(TypeError):
            MOMENTAnomalyDetector(window_size='fft', model_size=model_size)
        cleanup()

    @pytest.mark.parametrize('model_size', ['invalid'])
    def test_model_size_invalid_value(self, model_size, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(ValueError):
            MOMENTAnomalyDetector(window_size='fft', model_size=model_size)
        cleanup()

    @pytest.mark.parametrize('batch_size', [8, 16, 32])
    def test_batch_size_valid(self, batch_size, monkeypatch):
        setup(monkeypatch)
        detector = MOMENTAnomalyDetector(window_size='fft', batch_size=batch_size)
        assert detector.batch_size == batch_size
        cleanup()

    @pytest.mark.parametrize('batch_size', ['8', 8.0])
    def test_batch_size_invalid_type(self, batch_size, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(TypeError):
            MOMENTAnomalyDetector(window_size='fft', batch_size=batch_size)
        cleanup()

    @pytest.mark.parametrize('batch_size', [0, -8])
    def test_batch_size_invalid_value(self, batch_size, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(ValueError):
            MOMENTAnomalyDetector(window_size='fft', batch_size=batch_size)
        cleanup()

    @pytest.mark.parametrize('do_fine_tuning', [True, False])
    def test_do_fine_tuning_valid(self, do_fine_tuning, monkeypatch):
        setup(monkeypatch)
        detector = MOMENTAnomalyDetector(window_size='fft', do_fine_tuning=do_fine_tuning)
        assert detector.do_fine_tuning == do_fine_tuning
        cleanup()

    @pytest.mark.parametrize('do_fine_tuning', [5, 1.0, 'invalid'])
    def test_do_fine_tuning_invalid_type(self, do_fine_tuning, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(TypeError):
            MOMENTAnomalyDetector(window_size='fft', do_fine_tuning=do_fine_tuning)
        cleanup()

    @pytest.mark.parametrize('learning_rate', [1e-5, 1e-3, 0.1, 1.0, 2])
    def test_learning_rate_valid(self, learning_rate, monkeypatch):
        setup(monkeypatch)
        detector = MOMENTAnomalyDetector(3, learning_rate=learning_rate)
        assert detector.learning_rate == learning_rate
        cleanup()

    @pytest.mark.parametrize('learning_rate', ["0", True, None, ['a', 'list']])
    def test_learning_rate_invalid_type(self, learning_rate, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(TypeError):
            MOMENTAnomalyDetector(3, learning_rate=learning_rate)
        cleanup()

    @pytest.mark.parametrize('learning_rate', [0.0, -1e-16])
    def test_learning_rate_invalid_value(self, learning_rate, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(ValueError):
            MOMENTAnomalyDetector(3, learning_rate=learning_rate)
        cleanup()

    @pytest.mark.parametrize('nb_epochs', [8, 16, 32])
    def test_nb_epochs_valid(self, nb_epochs, monkeypatch):
        setup(monkeypatch)
        detector = MOMENTAnomalyDetector(window_size='fft', nb_epochs=nb_epochs)
        assert detector.nb_epochs == nb_epochs
        cleanup()

    @pytest.mark.parametrize('nb_epochs', ['8', 8.0])
    def test_nb_epochs_invalid_type(self, nb_epochs, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(TypeError):
            MOMENTAnomalyDetector(window_size='fft', nb_epochs=nb_epochs)
        cleanup()

    @pytest.mark.parametrize('nb_epochs', [0, -8])
    def test_nb_epochs_invalid_value(self, nb_epochs, monkeypatch):
        setup(monkeypatch)
        with pytest.raises(ValueError):
            MOMENTAnomalyDetector(window_size='fft', nb_epochs=nb_epochs)
        cleanup()

    def test_raises_error_if_not_3_10(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 11, 7, "final", 0))
        with pytest.raises(EnvironmentError):
            MOMENTAnomalyDetector(15)

    def test_raises_if_momentfm_missing(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 10, 7, "final", 0))
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "momentfm":
                raise ImportError("No module named 'autogluon.timeseries'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(Exception, match="Module 'momentfm' is not available"):
            # call the constructor that runs your code
            MOMENTAnomalyDetector(15)

    def test_no_error_if_momentfm_available(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 10, 7, "final", 0))

        sys.modules["momentfm"] = types.ModuleType("momentfm")

        # should NOT raise
        MOMENTAnomalyDetector(15)  # just runs, no exception

        # cleanup (avoid side effects on other tests)
        del sys.modules["momentfm"]
