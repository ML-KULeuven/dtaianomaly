
from dtaianomaly.anomaly_detection import OneClassSupportVectorMachine, Supervision


class TestOneClassSupportVectorMachine:

    def test_supervision(self):
        detector = OneClassSupportVectorMachine(1)
        assert detector.supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert str(OneClassSupportVectorMachine(5)) == "OneClassSupportVectorMachine(window_size=5)"
        assert str(OneClassSupportVectorMachine('fft')) == "OneClassSupportVectorMachine(window_size='fft')"
        assert str(OneClassSupportVectorMachine(15, 3)) == "OneClassSupportVectorMachine(window_size=15,stride=3)"
        assert str(OneClassSupportVectorMachine(25, kernel='poly')) == "OneClassSupportVectorMachine(window_size=25,kernel='poly')"
