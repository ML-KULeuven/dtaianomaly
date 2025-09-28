import numpy as np
import pytest
from scipy.spatial.distance import pdist

from dtaianomaly.anomaly_detection import LocalOutlierFactor, Supervision


def is_valid_scipy_distance(metric: str) -> bool:
    try:
        # Use dummy data with at least two rows
        pdist(np.array([[0, 0], [1, 1]]), metric=metric)
        return True
    except ValueError:
        return False


class TestLocalOutlierFactor:

    def test_supervision(self):
        detector = LocalOutlierFactor(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert str(LocalOutlierFactor(5)) == "LocalOutlierFactor(window_size=5)"
        assert (
            str(LocalOutlierFactor(15, 3))
            == "LocalOutlierFactor(window_size=15,stride=3)"
        )
        assert (
            str(LocalOutlierFactor(25, n_neighbors=42))
            == "LocalOutlierFactor(window_size=25,n_neighbors=42)"
        )

    def test_n_neighbors(self):
        LocalOutlierFactor(15, n_neighbors=1)
        LocalOutlierFactor(15, n_neighbors=2)
        LocalOutlierFactor(15, n_neighbors=4)
        LocalOutlierFactor(15, n_neighbors=8)
        with pytest.raises(TypeError):
            LocalOutlierFactor(15, n_neighbors="6")
        with pytest.raises(TypeError):
            LocalOutlierFactor(15, n_neighbors=6.0)
        with pytest.raises(TypeError):
            LocalOutlierFactor(15, n_neighbors=True)
        with pytest.raises(ValueError):
            LocalOutlierFactor(15, n_neighbors=0)
        with pytest.raises(ValueError):
            LocalOutlierFactor(15, n_neighbors=-1)

    def test_metric(self):
        LocalOutlierFactor(15, metric="minkowski")
        LocalOutlierFactor(15, metric="euclidean")
        LocalOutlierFactor(15, metric="jaccard")
        with pytest.raises(TypeError):
            LocalOutlierFactor(15, metric=6)
        with pytest.raises(TypeError):
            LocalOutlierFactor(15, metric=6.0)
        with pytest.raises(TypeError):
            LocalOutlierFactor(15, metric=True)
        with pytest.raises(ValueError):
            LocalOutlierFactor(15, metric="some-invalid-string")
