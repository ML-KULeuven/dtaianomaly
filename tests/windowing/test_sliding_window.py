import numpy as np

from dtaianomaly.windowing import sliding_window


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
        assert np.array_equal(
            windows[2], [8, 80, 9, 90, 10, 100, 11, 110, 12, 120, 13, 130]
        )
        assert np.array_equal(
            windows[3], [12, 120, 13, 130, 14, 140, 15, 150, 16, 160, 17, 170]
        )
        assert np.array_equal(
            windows[4], [14, 140, 15, 150, 16, 160, 17, 170, 18, 180, 19, 190]
        )
