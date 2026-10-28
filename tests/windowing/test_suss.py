import numpy as np

from dtaianomaly.data import demonstration_time_series
from dtaianomaly.windowing import summary_statistics_subsequences


class TestSummaryStatisticsSubsequences:

    def test_suss_exact_threshold(self):
        X, _ = demonstration_time_series()
        assert summary_statistics_subsequences(X, threshold=0.9437091537824681) == 104

    def test_upper_bound_exclusive(self):
        X = np.sin(2 * np.pi * np.arange(1000) / 50)
        assert summary_statistics_subsequences(X, lower_bound=10, upper_bound=37) == 36
        assert summary_statistics_subsequences(X, lower_bound=10, upper_bound=36) == -1

    def test_no_window_within_upper_bound(self):
        X = np.sin(np.arange(80) / 8)
        assert summary_statistics_subsequences(X, lower_bound=2, upper_bound=10) == -1

    def test_upper_bound_capped_by_length(self):
        X = np.sin(np.arange(80) / 8)
        window_size = summary_statistics_subsequences(
            X, lower_bound=2, upper_bound=1000
        )
        assert 2 <= window_size < X.shape[0] - 1

    def test_lower_bound_not_below_length(self):
        X = np.sin(np.arange(20))
        assert summary_statistics_subsequences(X, lower_bound=19) == -1
