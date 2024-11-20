
import pytest
import numpy as np
from dtaianomaly.anomaly_detection import MedianMethod, Supervision


class TestMedianMethod:

    def test_supervision(self):
        detector = MedianMethod(15)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_initialize(self):
        detector = MedianMethod(neighborhood_size_before=15, neighborhood_size_after=10)
        assert detector.neighborhood_size_before == 15
        assert detector.neighborhood_size_after == 10

    def test_initialize_no_neighborhood_size_after(self):
        detector = MedianMethod(neighborhood_size_before=15)
        assert detector.neighborhood_size_before == 15
        assert detector.neighborhood_size_after is None

    def test_initialize_too_small_neighborhood_size_before(self):
        with pytest.raises(ValueError):
            MedianMethod(neighborhood_size_before=0)
        MedianMethod(neighborhood_size_before=1)  # Doesn't raise an error with float
        MedianMethod(neighborhood_size_before=15)  # Doesn't raise an error with int

    def test_initialize_float_neighborhood_size_before(self):
        with pytest.raises(TypeError):
            MedianMethod(neighborhood_size_before=5.5)

    def test_initialize_string_neighborhood_size_before(self):
        with pytest.raises(TypeError):
            MedianMethod(neighborhood_size_before='15')

    def test_initialize_too_small_neighborhood_size_after(self):
        with pytest.raises(ValueError):
            MedianMethod(neighborhood_size_before=15, neighborhood_size_after=-1)
        MedianMethod(neighborhood_size_before=15, neighborhood_size_after=0)  # Doesn't raise an error with float
        MedianMethod(neighborhood_size_before=15, neighborhood_size_after=1)  # Doesn't raise an error with float
        MedianMethod(neighborhood_size_before=15, neighborhood_size_after=5)  # Doesn't raise an error with int

    def test_initialize_float_neighborhood_size_after(self):
        with pytest.raises(TypeError):
            MedianMethod(neighborhood_size_before=10, neighborhood_size_after=2.5)

    def test_initialize_string_neighborhood_size_after(self):
        with pytest.raises(TypeError):
            MedianMethod(neighborhood_size_before=10, neighborhood_size_after='1')

    def test_str(self):
        assert str(MedianMethod(neighborhood_size_before=5)) == "MedianMethod(neighborhood_size_before=5)"
        assert str(MedianMethod(neighborhood_size_before=15, neighborhood_size_after=3)) == "MedianMethod(neighborhood_size_before=15,neighborhood_size_after=3)"

    @staticmethod
    def check_prediction(pred, actual, neighborhood):
        if np.std(neighborhood) == 0.0:
            assert pred == 0.0
        else:
            assert pred == abs(actual - np.mean(neighborhood)) / np.std(neighborhood)

    def test_simple_example(self):
        detector = MedianMethod(2, 1)
        X = np.array([0, 1, 2, 3, 4, 15, 6, 7, 8])
        y_pred = detector.decision_function(X)
        self.check_prediction(y_pred[0], X[0], [0, 1])
        self.check_prediction(y_pred[1], X[1], [0, 1, 2])
        self.check_prediction(y_pred[2], X[2], [0, 1, 2, 3])
        self.check_prediction(y_pred[3], X[3], [1, 2, 3, 4])
        self.check_prediction(y_pred[4], X[4], [2, 3, 4, 15])
        self.check_prediction(y_pred[5], X[5], [3, 4, 15, 6])
        self.check_prediction(y_pred[6], X[6], [4, 15, 6, 7])
        self.check_prediction(y_pred[7], X[7], [15, 6, 7, 8])
        self.check_prediction(y_pred[8], X[8], [6, 7, 8])

    def test_simple_example_no_neighborhood_size_after(self):
        detector = MedianMethod(2)
        X = np.array([0, 1, 2, 3, 4, 15, 6, 7, 8])
        y_pred = detector.decision_function(X)
        self.check_prediction(y_pred[0], X[0], [0, 1, 2])
        self.check_prediction(y_pred[1], X[1], [0, 1, 2, 3])
        self.check_prediction(y_pred[2], X[2], [0, 1, 2, 3, 4])
        self.check_prediction(y_pred[3], X[3], [1, 2, 3, 4, 15])
        self.check_prediction(y_pred[4], X[4], [2, 3, 4, 15, 6])
        self.check_prediction(y_pred[5], X[5], [3, 4, 15, 6, 7])
        self.check_prediction(y_pred[6], X[6], [4, 15, 6, 7, 8])
        self.check_prediction(y_pred[7], X[7], [15, 6, 7, 8])
        self.check_prediction(y_pred[8], X[8], [6, 7, 8])

    def test_simple_example_zero_neighborhood_size_after(self):
        detector = MedianMethod(2, 0)
        X = np.array([0, 1, 2, 3, 4, 15, 6, 7, 8])
        y_pred = detector.decision_function(X)
        self.check_prediction(y_pred[0], X[0], [0])
        self.check_prediction(y_pred[1], X[1], [0, 1])
        self.check_prediction(y_pred[2], X[2], [0, 1, 2])
        self.check_prediction(y_pred[3], X[3], [1, 2, 3])
        self.check_prediction(y_pred[4], X[4], [2, 3, 4])
        self.check_prediction(y_pred[5], X[5], [3, 4, 15])
        self.check_prediction(y_pred[6], X[6], [4, 15, 6])
        self.check_prediction(y_pred[7], X[7], [15, 6, 7])
        self.check_prediction(y_pred[8], X[8], [6, 7, 8])

    def test_simple_example_constant(self):
        detector = MedianMethod(2)
        X = np.array([1, 1, 1, 1, 1])
        y_pred = detector.decision_function(X)
        assert np.all(y_pred == 0)
