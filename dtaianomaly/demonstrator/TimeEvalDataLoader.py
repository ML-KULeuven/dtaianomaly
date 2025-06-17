import pandas as pd

from dtaianomaly.data import DataSet, PathDataLoader


class TimeEvalDataLoader(PathDataLoader):

    def _load(self) -> DataSet:
        df = pd.read_csv(self.path)
        feature_names = [
            col for col in df.columns if col not in ["timestamp", "is_anomaly"]
        ]
        return DataSet(
            X_test=df[feature_names].values,
            y_test=df["is_anomaly"].values,
            feature_names=feature_names,
            time_steps_test=df["timestamp"].values,
        )
