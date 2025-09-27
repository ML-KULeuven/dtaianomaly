import numpy as np

from dtaianomaly.preprocessing import Identity


class TestIdentity:

    def test(self):
        rng = np.random.default_rng()
        x = rng.uniform(size=1000)
        y = rng.choice([0, 1], size=1000, replace=True)
        x_, y_ = Identity().fit_transform(x, y)
        assert np.array_equal(x, x_)
        assert np.array_equal(y, y_)
