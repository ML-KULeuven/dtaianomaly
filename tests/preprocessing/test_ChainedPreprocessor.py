from dtaianomaly.preprocessing import (
    ChainedPreprocessor,
    Identity,
    MinMaxScaler,
    StandardScaler,
)


class TestChainedPreprocessor:

    def test_piped_print(self):
        preprocessor = ChainedPreprocessor(Identity(), MinMaxScaler(), StandardScaler())
        assert (
            preprocessor.piped_print() == "Identity()->MinMaxScaler()->StandardScaler()"
        )
