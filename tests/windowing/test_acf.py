import numpy as np
import pytest

from dtaianomaly.windowing import highest_autocorrelation


class TestHighestAutocorrelation:

    @pytest.mark.parametrize("period_size", [25, 42])
    @pytest.mark.parametrize("nb_periods", [5, 10])
    def test_acf_simple(self, period_size, nb_periods):
        rng = np.random.default_rng(42)
        period = rng.uniform(size=period_size)
        X = np.tile(period, nb_periods)

        # Check if X is correctly formatted
        assert X.shape == (period_size * nb_periods,)
        assert np.array_equal(X[:period_size], period)

        window_size = highest_autocorrelation(X)
        assert window_size == period_size
