import numpy as np
import pytest

from dtaianomaly.preprocessing import SamplingRateUnderSampler


class TestSamplingRateUnderSampler:

    def test(self):
        x = np.array([4, 9, 2, 5, 4, 7, 4, 5])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        preprocessor = SamplingRateUnderSampler(2)
        x_, y_ = preprocessor.fit_transform(x, y)
        assert np.array_equal(x_, np.array([4, 2, 4, 4]))
        assert np.array_equal(y_, np.array([0, 0, 0, 0]))

    def test_too_large_sampling_rate(self, univariate_time_series):
        with pytest.raises(ValueError):
            SamplingRateUnderSampler(univariate_time_series.shape[0]).fit_transform(
                univariate_time_series
            )
