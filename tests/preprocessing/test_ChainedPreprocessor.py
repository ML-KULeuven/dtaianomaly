
import pytest

from dtaianomaly.preprocessing import ChainedPreprocessor, Identity, MinMaxScaler, StandardScaler


class TestChainedPreprocessor:

    def test_wrong_input(self):
        with pytest.raises(ValueError):
            ChainedPreprocessor([Identity(), 'five'])

    def test_single_input(self):
        ChainedPreprocessor(Identity())

    def test_multiple_inputs(self):
        ChainedPreprocessor(Identity(), Identity())

    def test_list_input(self):
        ChainedPreprocessor([Identity(), Identity()])

    def test_multiple_inputs_and_list(self):
        with pytest.raises(ValueError):
            ChainedPreprocessor(Identity(), [Identity(), Identity()])

    def test_zero_inputs(self):
        with pytest.raises(ValueError):
            ChainedPreprocessor()

    def test_str(self):
        preprocessor = ChainedPreprocessor(Identity(), MinMaxScaler(), StandardScaler())
        assert str(preprocessor) == 'Identity()->MinMaxScaler()->StandardScaler()'
