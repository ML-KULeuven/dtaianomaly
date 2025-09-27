import pytest

from dtaianomaly.anomaly_detection import OneClassSupportVectorMachine, Supervision


class TestOneClassSupportVectorMachine:

    def test_supervision(self):
        detector = OneClassSupportVectorMachine(1)
        assert detector.supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert (
            str(OneClassSupportVectorMachine(5))
            == "OneClassSupportVectorMachine(window_size=5)"
        )
        assert (
            str(OneClassSupportVectorMachine("fft"))
            == "OneClassSupportVectorMachine(window_size='fft')"
        )
        assert (
            str(OneClassSupportVectorMachine(15, 3))
            == "OneClassSupportVectorMachine(window_size=15,stride=3)"
        )
        assert (
            str(OneClassSupportVectorMachine(25, kernel="poly"))
            == "OneClassSupportVectorMachine(window_size=25,kernel='poly')"
        )

    def test_kernel(self):
        OneClassSupportVectorMachine(15, kernel="sigmoid")
        OneClassSupportVectorMachine(15, kernel="linear")
        OneClassSupportVectorMachine(15, kernel="rbf")
        OneClassSupportVectorMachine(15, kernel="poly")
        OneClassSupportVectorMachine(15, kernel="cosine")
        with pytest.raises(TypeError):
            OneClassSupportVectorMachine(15, kernel=0.5)
        with pytest.raises(TypeError):
            OneClassSupportVectorMachine(15, kernel=True)
        with pytest.raises(TypeError):
            OneClassSupportVectorMachine(15, kernel=1)
        with pytest.raises(ValueError):
            OneClassSupportVectorMachine(15, kernel="something-invalid")
