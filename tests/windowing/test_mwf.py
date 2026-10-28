import numpy as np
import pytest

from dtaianomaly.windowing import multi_window_finder


class TestMultiWindowFinder:

    def test_mwf_three_periods(self):
        X = np.sin(np.linspace(0, 1.5 * 2 * np.pi, 500))
        window_size = multi_window_finder(X, upper_bound=500)
        assert window_size == pytest.approx(500 // 3, abs=5)

    def test_upper_bound_exclusive(self):
        X = np.sin(2 * np.pi * np.arange(1000) / 50)
        assert multi_window_finder(X, upper_bound=60) == 50
        assert multi_window_finder(X, upper_bound=50) != 50
