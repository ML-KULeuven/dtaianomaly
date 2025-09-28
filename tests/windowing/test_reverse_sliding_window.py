import numpy as np
import pytest

from dtaianomaly.windowing import reverse_sliding_window


class TestReverseSlidingWindow:

    def test_window_size_1(self):
        scores = np.arange(10)
        reverse_windows = reverse_sliding_window(
            scores, window_size=1, stride=1, length_time_series=10
        )
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
        reverse_windows = reverse_sliding_window(
            scores, window_size=3, stride=1, length_time_series=10
        )
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
        reverse_windows = reverse_sliding_window(
            scores, window_size=3, stride=1, length_time_series=10
        )
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
        reverse_windows = reverse_sliding_window(
            scores, window_size=3, stride=2, length_time_series=11
        )
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
        reverse_windows = reverse_sliding_window(
            scores, window_size=3, stride=2, length_time_series=10
        )
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
        reverse_windows = reverse_sliding_window(
            scores, window_size=3, stride=3, length_time_series=15
        )
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
