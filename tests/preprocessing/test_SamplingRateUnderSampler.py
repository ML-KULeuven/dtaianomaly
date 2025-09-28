import numpy as np

from dtaianomaly.preprocessing import NbSamplesUnderSampler


class TestNbSamplesUnderSampler:

    def test_nb_samples_equal_to_length(self, univariate_time_series):
        x_, _ = NbSamplesUnderSampler(univariate_time_series.shape[0]).fit_transform(
            univariate_time_series
        )
        assert np.array_equal(x_, univariate_time_series)

    def test_nb_samples_longer_than_length(self, univariate_time_series):
        x_, _ = NbSamplesUnderSampler(
            univariate_time_series.shape[0] + 1
        ).fit_transform(univariate_time_series)
        assert np.array_equal(x_, univariate_time_series)

    def test_only_two_samples(self, univariate_time_series):
        x_, _ = NbSamplesUnderSampler(2).fit_transform(univariate_time_series)
        assert x_.shape[0] == 2
        assert x_[0] == univariate_time_series[0]
        assert x_[1] == univariate_time_series[-1]
