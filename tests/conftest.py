import os

# Tests never need an interactive window; the Tk backend fails intermittently.
os.environ.setdefault("MPLBACKEND", "Agg")

# Inside pytest-xdist workers, force third-party libraries (numba, torch,
# BLAS) to run single-threaded to avoid CPU oversubscription across workers.
# This must happen before numpy or dtaianomaly are imported.
if "PYTEST_XDIST_WORKER" in os.environ:
    for _var in (
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dtaianomaly.data import demonstration_time_series  # noqa: E402


@pytest.fixture(scope="session")
def _demonstration_time_series() -> np.array:
    x, _ = demonstration_time_series()
    return x


@pytest.fixture
def univariate_time_series(_demonstration_time_series) -> np.array:
    return _demonstration_time_series.copy()


@pytest.fixture
def multivariate_time_series() -> np.array:
    rng = np.random.default_rng()
    return rng.standard_normal(size=(500, 3))
