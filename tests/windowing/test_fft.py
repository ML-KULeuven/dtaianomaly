import numpy as np
import pytest

from dtaianomaly.windowing import dominant_fourier_frequency


class TestDominantFourrierFrequency:

    @pytest.mark.parametrize("nb_periods", [5, 10])
    def test_simple(self, nb_periods):
        X = np.sin(np.linspace(0, nb_periods * 2 * np.pi, 5000))
        window_size = dominant_fourier_frequency(X)
        assert window_size == 5000 / nb_periods
