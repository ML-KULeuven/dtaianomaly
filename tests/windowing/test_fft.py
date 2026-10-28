import numpy as np
import pytest

from dtaianomaly.windowing import dominant_fourier_frequency


class TestDominantFourrierFrequency:

    @pytest.mark.parametrize("nb_periods", [10, 20])
    def test_simple(self, nb_periods):
        X = np.sin(np.linspace(0, nb_periods * 2 * np.pi, 5000))
        window_size = dominant_fourier_frequency(X)
        assert window_size == 5000 / nb_periods

    def test_upper_bound_exclusive(self):
        X = np.sin(2 * np.pi * np.arange(1000) / 50)
        assert dominant_fourier_frequency(X, upper_bound=51) == 50
        assert dominant_fourier_frequency(X, upper_bound=50) < 50
