import numpy as np
import pytest

from dtaianomaly.data import demonstration_time_series
from dtaianomaly.utils import is_valid_array_like
from dtaianomaly.windowing import AUTO_WINDOW_SIZE_COMPUTATION, compute_window_size


class TestComputeWindowSize:

    def test_integer(self):
        for i in range(1, 100):
            assert i == compute_window_size(np.array([1, 2, 3]), i)

    @pytest.mark.parametrize("window_size", [1] + AUTO_WINDOW_SIZE_COMPUTATION)
    def test_invalid_x(self, window_size):
        assert not is_valid_array_like([1, 2, 3, 4, "5"])
        with pytest.raises(ValueError):
            compute_window_size([1, 2, 3, 4, "5"], window_size)

    def test_multivariate_integer(self, multivariate_time_series):
        assert 16 == compute_window_size(multivariate_time_series, 16)

    def test_multivariate_non_integer(self, multivariate_time_series):
        with pytest.raises(ValueError):
            compute_window_size(multivariate_time_series, "fft")

    @pytest.mark.parametrize("window_size", AUTO_WINDOW_SIZE_COMPUTATION)
    def test_demonstration_time_series(self, window_size):
        X, _ = demonstration_time_series()
        assert compute_window_size(X, window_size, threshold=0.95) == pytest.approx(
            1400 / (25 / 2), abs=10
        )

    @pytest.mark.parametrize("window_size", AUTO_WINDOW_SIZE_COMPUTATION)
    def test_no_window_size(self, window_size):
        flat = np.ones(shape=1000)
        with pytest.raises(ValueError):
            compute_window_size(flat, window_size)

    @pytest.mark.parametrize("window_size", AUTO_WINDOW_SIZE_COMPUTATION)
    def test_no_window_size_but_default_window_size(self, window_size):
        flat = np.ones(shape=1000)
        assert compute_window_size(flat, window_size, default_window_size=16) == 16

    @pytest.mark.parametrize("window_size", AUTO_WINDOW_SIZE_COMPUTATION)
    def test_invalid_bounds_default_window_size(
        self, window_size, univariate_time_series
    ):
        window_size_ = compute_window_size(
            univariate_time_series,
            window_size,
            lower_bound=int(univariate_time_series.shape[0] // 2),
            upper_bound=int(
                univariate_time_series.shape[0] // 3
            ),  # Smaller than lower_bound
            default_window_size=16,
        )
        assert window_size_ == 16

    @pytest.mark.parametrize("window_size", AUTO_WINDOW_SIZE_COMPUTATION)
    def test_invalid_bounds_no_default_window_size(
        self, window_size, univariate_time_series
    ):
        with pytest.raises(ValueError):
            compute_window_size(
                univariate_time_series,
                window_size,
                lower_bound=int(univariate_time_series.shape[0] // 2),
                upper_bound=int(
                    univariate_time_series.shape[0] // 3
                ),  # Smaller than lower_bound
                default_window_size=None,
            )

    @pytest.mark.parametrize("window_size", AUTO_WINDOW_SIZE_COMPUTATION)
    def test_too_small_lower_bound(self, window_size, univariate_time_series):
        with pytest.raises(ValueError):
            compute_window_size(
                univariate_time_series,
                window_size,
                lower_bound=-1,
                relative_upper_bound=-0.1,
                default_window_size=None,
            )

    @pytest.mark.parametrize("window_size", AUTO_WINDOW_SIZE_COMPUTATION)
    def test_too_large_upper_bound(self, window_size, univariate_time_series):
        with pytest.raises(ValueError):
            compute_window_size(
                univariate_time_series,
                window_size,
                upper_bound=2 * univariate_time_series.shape[0],
                relative_upper_bound=1.1,
                default_window_size=None,
            )
