import pytest

from dtaianomaly.anomaly_detection import KNearestNeighbors, Supervision


class TestKNearestNeighbors:

    def test_supervision(self):
        detector = KNearestNeighbors(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert str(KNearestNeighbors(5)) == "KNearestNeighbors(window_size=5)"
        assert str(KNearestNeighbors("fft")) == "KNearestNeighbors(window_size='fft')"
        assert (
            str(KNearestNeighbors(15, 3))
            == "KNearestNeighbors(window_size=15,stride=3)"
        )
        assert (
            str(KNearestNeighbors(25, n_neighbors=42))
            == "KNearestNeighbors(window_size=25,n_neighbors=42)"
        )
        assert (
            str(KNearestNeighbors(25, metric="euclidean"))
            == "KNearestNeighbors(window_size=25,metric='euclidean')"
        )

    def test_n_neighbors(self):
        KNearestNeighbors(15, n_neighbors=1)
        KNearestNeighbors(15, n_neighbors=2)
        KNearestNeighbors(15, n_neighbors=4)
        KNearestNeighbors(15, n_neighbors=8)
        with pytest.raises(TypeError):
            KNearestNeighbors(15, n_neighbors="6")
        with pytest.raises(TypeError):
            KNearestNeighbors(15, n_neighbors=6.0)
        with pytest.raises(TypeError):
            KNearestNeighbors(15, n_neighbors=True)
        with pytest.raises(ValueError):
            KNearestNeighbors(15, n_neighbors=0)
        with pytest.raises(ValueError):
            KNearestNeighbors(15, n_neighbors=-1)

    def test_method(self):
        KNearestNeighbors(15, method="largest")
        KNearestNeighbors(15, method="mean")
        KNearestNeighbors(15, method="median")
        with pytest.raises(TypeError):
            KNearestNeighbors(15, method=6)
        with pytest.raises(TypeError):
            KNearestNeighbors(15, method=6.0)
        with pytest.raises(TypeError):
            KNearestNeighbors(15, method=True)
        with pytest.raises(ValueError):
            KNearestNeighbors(15, method="some-invalid-string")

    def test_metric(self):
        KNearestNeighbors(15, metric="minkowski")
        KNearestNeighbors(15, metric="euclidean")
        KNearestNeighbors(15, metric="jaccard")
        with pytest.raises(TypeError):
            KNearestNeighbors(15, metric=6)
        with pytest.raises(TypeError):
            KNearestNeighbors(15, metric=6.0)
        with pytest.raises(TypeError):
            KNearestNeighbors(15, metric=True)
        with pytest.raises(ValueError):
            KNearestNeighbors(15, metric="some-invalid-string")
