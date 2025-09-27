from dtaianomaly.preprocessing import (
    ChainedPreprocessor,
    Identity,
    MinMaxScaler,
    StandardScaler,
)


class TestChainedPreprocessor:

    def test_input_list(self):
        preprocessor = ChainedPreprocessor(
            [Identity(), MinMaxScaler(), StandardScaler()]
        )
        assert len(preprocessor.base_preprocessors) == 3
        assert isinstance(preprocessor.base_preprocessors[0], Identity)
        assert isinstance(preprocessor.base_preprocessors[1], MinMaxScaler)
        assert isinstance(preprocessor.base_preprocessors[2], StandardScaler)

    def test_input_sequence(self):
        preprocessor = ChainedPreprocessor(Identity(), MinMaxScaler(), StandardScaler())
        assert len(preprocessor.base_preprocessors) == 3
        assert isinstance(preprocessor.base_preprocessors[0], Identity)
        assert isinstance(preprocessor.base_preprocessors[1], MinMaxScaler)
        assert isinstance(preprocessor.base_preprocessors[2], StandardScaler)

    def test_piped_str(self):
        preprocessor = ChainedPreprocessor(Identity(), MinMaxScaler(), StandardScaler())
        assert (
            preprocessor.piped_str() == "Identity()->MinMaxScaler()->StandardScaler()"
        )
