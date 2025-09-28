from dtaianomaly.anomaly_detection import IsolationForest
from dtaianomaly.pipeline import Pipeline
from dtaianomaly.preprocessing import ChainedPreprocessor, Identity, StandardScaler


class TestPipeline:

    def test_piped_str(self):
        assert (
            Pipeline(StandardScaler(), IsolationForest(15, 3)).piped_str()
            == "StandardScaler()->IsolationForest(window_size=15,stride=3)"
        )
        assert (
            Pipeline(
                ChainedPreprocessor(Identity(), StandardScaler()),
                IsolationForest(15, 3),
            ).piped_str()
            == "Identity()->StandardScaler()->IsolationForest(window_size=15,stride=3)"
        )

    def test_simple_execution(self, univariate_time_series):
        pipeline = Pipeline(StandardScaler(), IsolationForest(16))
        y_pred = pipeline.fit(univariate_time_series).decision_function(
            univariate_time_series
        )
        assert univariate_time_series.shape[0] == y_pred.shape[0]
