import numpy as np

from dtaianomaly.preprocessing import MovingAverage


class TestMovingAverage:

    def test_odd_window_size(self):
        x = np.array([1, 5, 3, 7, 8, 6, 4, 11])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        preprocessor = MovingAverage(3)
        x_, y_ = preprocessor.fit_transform(x, y)
        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, np.array([3, 3, 5, 6, 7, 6, 7, 7.5]))
        assert np.array_equal(y_, y)

    def test_even_window_size(self):
        x = np.array([1, 5, 3, 7, 8, 6, 4, 11])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        preprocessor = MovingAverage(4)
        x_, y_ = preprocessor.fit_transform(x, y)
        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, np.array([3, 3, 4, 5.75, 6, 6.25, 7.25, 7]))
        assert np.array_equal(y_, y)

    def test_simple_multivariate(self):
        x = np.array(
            [[1, 10], [5, 50], [3, 30], [7, 70], [8, 80], [6, 60], [4, 40], [11, 110]]
        )
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        preprocessor = MovingAverage(3)
        x_, y_ = preprocessor.fit_transform(x, y)
        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(
            x_,
            np.array(
                [
                    [3, 30],
                    [3, 30],
                    [5, 50],
                    [6, 60],
                    [7, 70],
                    [6, 60],
                    [7, 70],
                    [7.5, 75],
                ]
            ),
        )
        assert np.array_equal(y_, y)
