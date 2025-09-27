from dtaianomaly.data import demonstration_time_series
from dtaianomaly.windowing import summary_statistics_subsequences


class TestSummaryStatisticsSubsequences:

    def test_suss_exact_threshold(self):
        X, _ = demonstration_time_series()
        assert summary_statistics_subsequences(X, threshold=0.9437091537824681) == 104
