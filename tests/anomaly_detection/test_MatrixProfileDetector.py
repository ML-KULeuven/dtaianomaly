import pytest
from sklearn.exceptions import NotFittedError

from dtaianomaly.anomaly_detection import MatrixProfileDetector, Supervision


class TestMatrixProfileDetector:

    def test_supervision(self):
        detector = MatrixProfileDetector(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_initialize_non_int_window_size(self):
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=True)
        with pytest.raises(ValueError):
            MatrixProfileDetector(window_size="a string")
        MatrixProfileDetector(5)  # Doesn't raise an error

    def test_initialize_too_small_window_size(self):
        with pytest.raises(ValueError):
            MatrixProfileDetector(window_size=0)
        MatrixProfileDetector(5)  # Doesn't raise an error

    def test_valid_window_size(self):
        MatrixProfileDetector(1)
        MatrixProfileDetector(10)
        MatrixProfileDetector(100)
        MatrixProfileDetector("fft")

    def test_initialize_non_bool_normalize(self):
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, normalize=0)
        MatrixProfileDetector(5, normalize=False)  # Doesn't raise an error

    def test_initialize_non_numeric_p(self):
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, p=False)
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, p="a string")
        MatrixProfileDetector(5, p=2.5)  # Doesn't raise an error with float
        MatrixProfileDetector(5, p=3.0)  # Doesn't raise an error with int

    def test_initialize_too_small_p(self):
        with pytest.raises(ValueError):
            MatrixProfileDetector(window_size=15, p=0.5)
        MatrixProfileDetector(5, p=2.5)  # Doesn't raise an error with float
        MatrixProfileDetector(5, p=3.0)  # Doesn't raise an error with int

    def test_initialize_non_integer_k(self):
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, k=1.5)
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, k=True)
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, k="a string")
        MatrixProfileDetector(5, k=2)  # Doesn't raise an error

    def test_initialize_too_small_k(self):
        with pytest.raises(ValueError):
            MatrixProfileDetector(window_size=15, k=0)
        MatrixProfileDetector(5, k=2)  # Doesn't raise an error

    def test_initialize_non_bool_novelty(self):
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, novelty=0)
        MatrixProfileDetector(5, novelty=False)  # Doesn't raise an error

    @pytest.mark.slow
    def test_novelty_univariate(self, univariate_time_series):
        detector = MatrixProfileDetector(window_size=15, novelty=True)
        y_pred = detector.fit(univariate_time_series).decision_function(
            univariate_time_series
        )
        assert y_pred.shape == (univariate_time_series.shape[0],)

    @pytest.mark.slow
    def test_novelty_multivariate(self, multivariate_time_series):
        detector = MatrixProfileDetector(window_size=15, novelty=True)
        y_pred = detector.fit(multivariate_time_series).decision_function(
            multivariate_time_series
        )
        assert y_pred.shape == (multivariate_time_series.shape[0],)

    @pytest.mark.slow
    def test_not_fitted_no_novelty(self, univariate_time_series):
        detector = MatrixProfileDetector(window_size=15, novelty=False)
        detector.fit(univariate_time_series)
        detector.decision_function(univariate_time_series)

    def test_not_fitted_novelty(self, univariate_time_series):
        detector = MatrixProfileDetector(window_size=15, novelty=True)
        with pytest.raises(NotFittedError):
            detector.decision_function(univariate_time_series)

    def test_novelty_different_dimensions(
        self, univariate_time_series, multivariate_time_series
    ):
        detector = MatrixProfileDetector(window_size=15, novelty=True)
        detector.fit(univariate_time_series)
        with pytest.raises(ValueError):
            detector.decision_function(multivariate_time_series)

    @pytest.mark.slow
    def test_no_novelty_different_dimension(
        self, univariate_time_series, multivariate_time_series
    ):
        detector = MatrixProfileDetector(window_size=15, novelty=False)
        detector.fit(multivariate_time_series)
        detector.decision_function(univariate_time_series)  # No error

    @pytest.mark.slow
    def test_novelty_univariate_but_multi_dimensional(self, univariate_time_series):
        data_2d = univariate_time_series.reshape(-1, 1)
        assert data_2d.shape == (univariate_time_series.shape[0], 1)
        assert len(data_2d.shape) == 2
        detector = MatrixProfileDetector(window_size=15, novelty=True)
        detector.fit(univariate_time_series).decision_function(data_2d)  # No error
        detector.fit(data_2d).decision_function(univariate_time_series)  # No error

    def test_str(self):
        assert str(MatrixProfileDetector(5)) == "MatrixProfileDetector(window_size=5)"
        assert (
            str(MatrixProfileDetector(15, normalize=False, p=1.5))
            == "MatrixProfileDetector(window_size=15,normalize=False,p=1.5)"
        )
        assert (
            str(MatrixProfileDetector(15, p=1.5, normalize=False))
            == "MatrixProfileDetector(window_size=15,normalize=False,p=1.5)"
        )
        assert (
            str(MatrixProfileDetector(25, k=2))
            == "MatrixProfileDetector(window_size=25,k=2)"
        )
