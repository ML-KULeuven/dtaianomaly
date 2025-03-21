
import pytest
import numpy as np
from dtaianomaly import utils
from dtaianomaly.data import demonstration_time_series
from dtaianomaly.anomaly_detection.windowing_utils import sliding_window, reverse_sliding_window, check_is_valid_window_size, compute_window_size


class TestSlidingWindow:

    def test_stride_1_odd_window_size_univariate(self):
        x = np.arange(10)
        windows = sliding_window(x, 3, 1)
        assert windows.shape == (8, 3)
        assert np.array_equal(windows[0], [0, 1, 2])
        assert np.array_equal(windows[1], [1, 2, 3])
        assert np.array_equal(windows[2], [2, 3, 4])
        assert np.array_equal(windows[3], [3, 4, 5])
        assert np.array_equal(windows[4], [4, 5, 6])
        assert np.array_equal(windows[5], [5, 6, 7])
        assert np.array_equal(windows[6], [6, 7, 8])
        assert np.array_equal(windows[7], [7, 8, 9])

    def test_stride_1_even_window_size_univariate(self):
        x = np.arange(10)
        windows = sliding_window(x, 4, 1)
        assert windows.shape == (7, 4)
        assert np.array_equal(windows[0], [0, 1, 2, 3])
        assert np.array_equal(windows[1], [1, 2, 3, 4])
        assert np.array_equal(windows[2], [2, 3, 4, 5])
        assert np.array_equal(windows[3], [3, 4, 5, 6])
        assert np.array_equal(windows[4], [4, 5, 6, 7])
        assert np.array_equal(windows[5], [5, 6, 7, 8])
        assert np.array_equal(windows[6], [6, 7, 8, 9])

    def test_nice_fit_univariate(self):
        x = np.arange(11)
        windows = sliding_window(x, 3, 2)
        assert windows.shape == (5, 3)
        assert np.array_equal(windows[0], [0, 1, 2])
        assert np.array_equal(windows[1], [2, 3, 4])
        assert np.array_equal(windows[2], [4, 5, 6])
        assert np.array_equal(windows[3], [6, 7, 8])
        assert np.array_equal(windows[4], [8, 9, 10])

    def test_not_nice_fit_univariate(self):
        x = np.arange(10)
        windows = sliding_window(x, 3, 2)
        assert windows.shape == (5, 3)
        assert np.array_equal(windows[0], [0, 1, 2])
        assert np.array_equal(windows[1], [2, 3, 4])
        assert np.array_equal(windows[2], [4, 5, 6])
        assert np.array_equal(windows[3], [6, 7, 8])
        assert np.array_equal(windows[4], [7, 8, 9])

    def test_not_nice_fit_large_stride_univariate(self):
        x = np.arange(20)
        windows = sliding_window(x, 6, 4)
        assert windows.shape == (5, 6)
        assert np.array_equal(windows[0], [0, 1, 2, 3, 4, 5])
        assert np.array_equal(windows[1], [4, 5, 6, 7, 8, 9])
        assert np.array_equal(windows[2], [8, 9, 10, 11, 12, 13])
        assert np.array_equal(windows[3], [12, 13, 14, 15, 16, 17])
        assert np.array_equal(windows[4], [14, 15, 16, 17, 18, 19])

    def test_stride_1_odd_window_size_multivariate(self):
        x = np.array([np.arange(10), np.arange(10) * 10]).T
        windows = sliding_window(x, 3, 1)
        assert windows.shape == (8, 6)
        assert np.array_equal(windows[0], [0, 0, 1, 10, 2, 20])
        assert np.array_equal(windows[1], [1, 10, 2, 20, 3, 30])
        assert np.array_equal(windows[2], [2, 20, 3, 30, 4, 40])
        assert np.array_equal(windows[3], [3, 30, 4, 40, 5, 50])
        assert np.array_equal(windows[4], [4, 40, 5, 50, 6, 60])
        assert np.array_equal(windows[5], [5, 50, 6, 60, 7, 70])
        assert np.array_equal(windows[6], [6, 60, 7, 70, 8, 80])
        assert np.array_equal(windows[7], [7, 70, 8, 80, 9, 90])

    def test_stride_1_even_window_size_multivariate(self):
        x = np.array([np.arange(10), np.arange(10) * 10]).T
        windows = sliding_window(x, 4, 1)
        assert windows.shape == (7, 8)
        assert np.array_equal(windows[0], [0, 0, 1, 10, 2, 20, 3, 30])
        assert np.array_equal(windows[1], [1, 10, 2, 20, 3, 30, 4, 40])
        assert np.array_equal(windows[2], [2, 20, 3, 30, 4, 40, 5, 50])
        assert np.array_equal(windows[3], [3, 30, 4, 40, 5, 50, 6, 60])
        assert np.array_equal(windows[4], [4, 40, 5, 50, 6, 60, 7, 70])
        assert np.array_equal(windows[5], [5, 50, 6, 60, 7, 70, 8, 80])
        assert np.array_equal(windows[6], [6, 60, 7, 70, 8, 80, 9, 90])

    def test_nice_fit_multivariate(self):
        x = np.array([np.arange(11), np.arange(11) * 10]).T
        windows = sliding_window(x, 3, 2)
        assert windows.shape == (5, 6)
        assert np.array_equal(windows[0], [0, 0, 1, 10, 2, 20])
        assert np.array_equal(windows[1], [2, 20, 3, 30, 4, 40])
        assert np.array_equal(windows[2], [4, 40, 5, 50, 6, 60])
        assert np.array_equal(windows[3], [6, 60, 7, 70, 8, 80])
        assert np.array_equal(windows[4], [8, 80, 9, 90, 10, 100])

    def test_not_nice_fit_multivariate(self):
        x = np.array([np.arange(10), np.arange(10) * 10]).T
        windows = sliding_window(x, 3, 2)
        assert windows.shape == (5, 6)
        assert np.array_equal(windows[0], [0, 0, 1, 10, 2, 20])
        assert np.array_equal(windows[1], [2, 20, 3, 30, 4, 40])
        assert np.array_equal(windows[2], [4, 40, 5, 50, 6, 60])
        assert np.array_equal(windows[3], [6, 60, 7, 70, 8, 80])
        assert np.array_equal(windows[4], [7, 70, 8, 80, 9, 90])

    def test_not_nice_fit_large_stride_multivariate(self):
        x = np.array([np.arange(20), np.arange(20) * 10]).T
        windows = sliding_window(x, 6, 4)
        assert windows.shape == (5, 12)
        assert np.array_equal(windows[0], [0, 0, 1, 10, 2, 20, 3, 30, 4, 40, 5, 50])
        assert np.array_equal(windows[1], [4, 40, 5, 50, 6, 60, 7, 70, 8, 80, 9, 90])
        assert np.array_equal(windows[2], [8, 80, 9, 90, 10, 100, 11, 110, 12, 120, 13, 130])
        assert np.array_equal(windows[3], [12, 120, 13, 130, 14, 140, 15, 150, 16, 160, 17, 170])
        assert np.array_equal(windows[4], [14, 140, 15, 150, 16, 160, 17, 170, 18, 180, 19, 190])


class TestReverseSlidingWindow:

    def test_window_size_1(self):
        scores = np.arange(10)
        reverse_windows = reverse_sliding_window(scores, window_size=1, stride=1, length_time_series=10)
        assert len(reverse_windows.shape) == 1
        assert reverse_windows.shape[0] == 10
        assert reverse_windows[0] == 0
        assert reverse_windows[1] == 1
        assert reverse_windows[2] == 2
        assert reverse_windows[3] == 3
        assert reverse_windows[4] == 4
        assert reverse_windows[5] == 5
        assert reverse_windows[6] == 6
        assert reverse_windows[7] == 7
        assert reverse_windows[8] == 8
        assert reverse_windows[9] == 9

    def test_stride_1(self):
        scores = np.arange(8)
        reverse_windows = reverse_sliding_window(scores, window_size=3, stride=1, length_time_series=10)
        assert len(reverse_windows.shape) == 1
        assert reverse_windows.shape[0] == 10
        assert reverse_windows[0] == 0  # [0]
        assert reverse_windows[1] == 0.5  # [0, 1]
        assert reverse_windows[2] == 1  # [0, 1, 2]
        assert reverse_windows[3] == 2  # [1, 2, 3]
        assert reverse_windows[4] == 3  # [2, 3, 4]
        assert reverse_windows[5] == 4  # [3, 4, 5]
        assert reverse_windows[6] == 5  # [4, 5, 6]
        assert reverse_windows[7] == 6  # [5, 6, 7]
        assert reverse_windows[8] == 6.5  # [6, 7]
        assert reverse_windows[9] == 7  # [7]

    def test_stride_1_bigger_numbers(self):
        # Mean and median is the for np.arange
        scores = 2 ** np.arange(8)
        reverse_windows = reverse_sliding_window(scores, window_size=3, stride=1, length_time_series=10)
        assert len(reverse_windows.shape) == 1
        assert reverse_windows.shape[0] == 10
        assert reverse_windows[0] == 1  # [1]
        assert reverse_windows[1] == 1.5  # [1, 2]
        assert reverse_windows[2] == pytest.approx(7 / 3)  # [1, 2, 4]
        assert reverse_windows[3] == pytest.approx(14 / 3)  # [2, 4, 8]
        assert reverse_windows[4] == pytest.approx(28 / 3)  # [4, 8, 16]
        assert reverse_windows[5] == pytest.approx(56 / 3)  # [8, 16, 32]
        assert reverse_windows[6] == pytest.approx(112 / 3)  # [16, 32, 64]
        assert reverse_windows[7] == pytest.approx(224 / 3)  # [32 64, 128]
        assert reverse_windows[8] == 96  # [64, 128]
        assert reverse_windows[9] == 128  # [128]

    def test_nice_fit(self):
        scores = np.arange(5)
        reverse_windows = reverse_sliding_window(scores, window_size=3, stride=2, length_time_series=11)
        assert len(reverse_windows.shape) == 1
        assert reverse_windows.shape[0] == 11
        assert reverse_windows[0] == 0  # [0]
        assert reverse_windows[1] == 0  # [0]
        assert reverse_windows[2] == 0.5  # [0, 1]
        assert reverse_windows[3] == 1  # [1]
        assert reverse_windows[4] == 1.5  # [1, 2]
        assert reverse_windows[5] == 2  # [2]
        assert reverse_windows[6] == 2.5  # [2, 3]
        assert reverse_windows[7] == 3  # [3]
        assert reverse_windows[8] == 3.5  # [3, 4]
        assert reverse_windows[9] == 4  # [4]
        assert reverse_windows[10] == 4  # [4]

    def test_not_nice_fit(self):
        scores = np.arange(5)
        reverse_windows = reverse_sliding_window(scores, window_size=3, stride=2, length_time_series=10)
        assert len(reverse_windows.shape) == 1
        assert reverse_windows.shape[0] == 10
        assert reverse_windows[0] == 0  # [0]
        assert reverse_windows[1] == 0  # [0]
        assert reverse_windows[2] == 0.5  # [0, 1]
        assert reverse_windows[3] == 1  # [1]
        assert reverse_windows[4] == 1.5  # [1, 2]
        assert reverse_windows[5] == 2  # [2]
        assert reverse_windows[6] == 2.5  # [2, 3]
        assert reverse_windows[7] == 3.5  # [3, 4]
        assert reverse_windows[8] == 3.5  # [3, 4]
        assert reverse_windows[9] == 4  # [4]

    def test_non_overlapping_windows(self):
        scores = np.arange(5)
        reverse_windows = reverse_sliding_window(scores, window_size=3, stride=3, length_time_series=15)
        assert len(reverse_windows.shape) == 1
        assert reverse_windows.shape[0] == 15
        assert reverse_windows[0] == 0  # [0]
        assert reverse_windows[1] == 0  # [0]
        assert reverse_windows[2] == 0  # [0]
        assert reverse_windows[3] == 1  # [1]
        assert reverse_windows[4] == 1  # [1]
        assert reverse_windows[5] == 1  # [1]
        assert reverse_windows[6] == 2  # [2]
        assert reverse_windows[7] == 2  # [2]
        assert reverse_windows[8] == 2  # [2]
        assert reverse_windows[9] == 3  # [3]
        assert reverse_windows[10] == 3  # [3]
        assert reverse_windows[11] == 3  # [3]
        assert reverse_windows[12] == 4  # [4]
        assert reverse_windows[13] == 4  # [4]
        assert reverse_windows[14] == 4  # [4]


class TestCheckIsValidWindowSize:

    def test_valid_integer(self):
        for i in range(1, 100):
            check_is_valid_window_size(i)

    @pytest.mark.parametrize('window_size', ['fft', 'acf'])
    def test_valid_string(self, window_size):
        check_is_valid_window_size(window_size)

    def test_invalid_integer(self):
        for i in [-10, -1, 0]:
            with pytest.raises(ValueError):
                check_is_valid_window_size(i)

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            check_is_valid_window_size('something_invalid')

    def test_invalid_float(self):
        with pytest.raises(ValueError):
            check_is_valid_window_size(1.0)

    def test_invalid_bool(self):
        with pytest.raises(ValueError):
            check_is_valid_window_size(True)
        with pytest.raises(ValueError):
            check_is_valid_window_size(False)


class TestComputeWindowSize:

    def test_integer(self):
        for i in range(1, 100):
            assert i == compute_window_size(np.array([1, 2, 3]), i)

    @pytest.mark.parametrize('window_size', [1, 'fft', 'acf', 'mwf', 'suss'])
    def test_invalid_x(self, window_size):
        check_is_valid_window_size(window_size)
        assert not utils.is_valid_array_like([1, 2, 3, 4, '5'])
        with pytest.raises(ValueError):
            compute_window_size([1, 2, 3, 4, '5'], window_size)

    def test_multivariate_integer(self, multivariate_time_series):
        assert 16 == compute_window_size(multivariate_time_series, 16)

    def test_multivariate_non_integer(self, multivariate_time_series):
        with pytest.raises(ValueError):
            compute_window_size(multivariate_time_series, 'fft')

    @pytest.mark.parametrize('window_size', ['fft', 'acf', 'mwf', 'suss'])
    def test_demonstration_time_series(self, window_size):
        X, _ = demonstration_time_series()
        assert compute_window_size(X, window_size, threshold=0.95) == pytest.approx(1400 / (25 / 2), abs=10)

    @pytest.mark.parametrize('window_size', ['fft', 'acf', 'mwf', 'suss'])
    def test_no_window_size(self, window_size):
        flat = np.ones(shape=1000)
        with pytest.raises(ValueError):
            compute_window_size(flat, window_size)

    @pytest.mark.parametrize('window_size', ['fft', 'acf', 'mwf', 'suss'])
    def test_no_window_size_but_default_window_size(self, window_size):
        flat = np.ones(shape=1000)
        assert compute_window_size(flat, window_size, default_window_size=16) == 16

    @pytest.mark.parametrize('nb_periods', [5, 10])
    def test_fft_simple(self, nb_periods):
        X = np.sin(np.linspace(0, nb_periods * 2 * np.pi, 5000))
        window_size = compute_window_size(X, window_size='fft')
        assert window_size == 5000/nb_periods

    @pytest.mark.parametrize('period_size', [25, 42])
    @pytest.mark.parametrize('nb_periods', [5, 10])
    def test_acf_simple(self, period_size, nb_periods):
        rng = np.random.default_rng(42)
        period = rng.uniform(size=period_size)
        X = np.tile(period, nb_periods)

        # Check if X is correctly formatted
        assert X.shape == (period_size * nb_periods,)
        assert np.array_equal(X[:period_size], period)

        window_size = compute_window_size(X, window_size='acf')
        assert window_size == period_size

    def test_mwf_three_periods(self):
        X = np.sin(np.linspace(0, 1.5 * 2 * np.pi, 500))

        window_size = compute_window_size(X, window_size='mwf', upper_bound=500)
        assert window_size == pytest.approx(500 // 3, abs=5)

    def test_suss_exact_threshold(self):
        X, _ = demonstration_time_series()
        assert compute_window_size(X, 'suss', threshold=0.9437091537824681) == 104

    @pytest.mark.parametrize('window_size', ['fft', 'acf', 'mwf', 'suss'])
    def test_invalid_bounds_default_window_size(self, window_size, univariate_time_series):
        window_size_ = compute_window_size(
            univariate_time_series, window_size,
            lower_bound=int(univariate_time_series.shape[0] // 2),
            upper_bound=int(univariate_time_series.shape[0] // 3),  # Smaller than lower_bound
            default_window_size=16
        )
        assert window_size_ == 16

    @pytest.mark.parametrize('window_size', ['fft', 'acf', 'mwf', 'suss'])
    def test_invalid_bounds_no_default_window_size(self, window_size, univariate_time_series):
        with pytest.raises(ValueError):
            compute_window_size(
                univariate_time_series, window_size,
                lower_bound=int(univariate_time_series.shape[0] // 2),
                upper_bound=int(univariate_time_series.shape[0] // 3),  # Smaller than lower_bound
                default_window_size=None
            )

    @pytest.mark.parametrize('window_size', ['fft', 'acf', 'mwf', 'suss'])
    def test_too_small_lower_bound(self, window_size, univariate_time_series):
        with pytest.raises(ValueError):
            compute_window_size(
                univariate_time_series, window_size,
                lower_bound=-1,
                relative_upper_bound=-0.1,
                default_window_size=None
            )

    @pytest.mark.parametrize('window_size', ['fft', 'acf', 'mwf', 'suss'])
    def test_too_large_upper_bound(self, window_size, univariate_time_series):
        with pytest.raises(ValueError):
            compute_window_size(
                univariate_time_series, window_size,
                upper_bound=2*univariate_time_series.shape[0],
                relative_upper_bound=1.1,
                default_window_size=None
            )
