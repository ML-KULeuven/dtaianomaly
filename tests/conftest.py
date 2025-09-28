import numpy as np
import pytest

from dtaianomaly.data import demonstration_time_series


@pytest.fixture
def univariate_time_series() -> np.array:
    x, _ = demonstration_time_series()
    return x


@pytest.fixture
def multivariate_time_series() -> np.array:
    rng = np.random.default_rng()
    return rng.standard_normal(size=(500, 3))
