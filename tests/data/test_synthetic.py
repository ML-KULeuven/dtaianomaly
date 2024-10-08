
import numpy as np
from dtaianomaly.data.synthetic import make_sine_wave, inject_anomalies


class TestInjectAnomalies:

    def test_univariate(self, univariate_time_series):
        np.random.seed(0)
        anomalies = inject_anomalies(univariate_time_series, nb_anomalies=15)
        assert anomalies.sum() == 15
        assert anomalies.shape[0] == univariate_time_series.shape[0]
        assert len(anomalies.shape) == 1
        assert ((anomalies == 0) | (anomalies == 1)).all()

    def test_amplitude_univariate(self):
        np.random.seed(0)
        time_series = np.zeros(1000)
        anomalies = inject_anomalies(time_series, 950, 1.0, 2.0)
        assert ((anomalies == 0) | ((1.0 <= time_series) & (time_series <= 2.0)) | ((-2.0 <= time_series) & (time_series <= -1.0))).all()

    def test_multivariate(self, multivariate_time_series):
        np.random.seed(0)
        anomalies = inject_anomalies(multivariate_time_series, nb_anomalies=15)
        assert anomalies.sum() == 15
        assert anomalies.shape[0] == multivariate_time_series.shape[0]
        assert len(anomalies.shape) == 1
        assert ((anomalies == 0) | (anomalies == 1)).all()

    def test_amplitude_multivariate(self):
        np.random.seed(0)
        time_series = np.zeros((1000, 3))
        anomalies = inject_anomalies(time_series, 950, 1.0, 2.0)
        for i in range(1000):
            for j in range(3):
                if anomalies[i]:
                    assert 1.0 <= abs(time_series[i, j]) <= 2.0
                else:
                    assert time_series[i, j] == 0.0


class TestMakeSineWave:

    def test(self):
        x, y = make_sine_wave(1234)
        assert x.shape[0] == 1234
        assert x.shape[1] == 1
        assert y.shape[0] == 1234
        assert len(y.shape) == 1

    def test_nb_anomalies(self):
        x, y = make_sine_wave(1234, nb_anomalies=20)
        assert y.sum() == 20

    def test_seed(self):
        seed = 1
        x, y = make_sine_wave(1234, seed=seed)
        x_, y_ = make_sine_wave(1234, seed=seed)
        assert np.array_equal(x, x_)
        assert np.array_equal(y, y_)
