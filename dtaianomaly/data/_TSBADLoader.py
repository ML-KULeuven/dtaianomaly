import os

import pandas as pd

from dtaianomaly.data import DataSet, PathDataLoader

__all__ = ["TSBADLoader"]


class TSBADLoader(PathDataLoader):
    """
    The TSB-AD benchmark comprises 870 univariate time series and 200 multivariate
    time series :cite:`liu2024elephant`, as well as predefined tune and evaluation
    sets. The benchmark has been constructed through the following steps:

    1. **Dataset construction.** Collect 13 univariate and 20 multivariate publicly
       vailable datasets. The multivariate time series are also transformed to multiple
       univariate time series by assuming channel independence, in which channels with
       low correlation to the anomaly scores are ignored.
    2. **Flaw identification.** A human annotator excludes several time series that have
       some common flaws and could induce a bias and inaccurate evaluation.
    3. **Label quality assessment.** The quality of the anomaly scores are assessed through
       an algorithmic test, in which at least one anomaly detector should be able to locate
       the anomalies.

    A tune and evaluation set have been defined. These are available at https://github.com/TheDatumOrg/TSB-AD/tree/main/Datasets/File_List.

    Notes
    -----
    This implementation expects the file names to have the following form:
    ``<index>_<Dataset Name>_id_<id>_<Domain>_tr_<Train Index>_1st_<First Anomaly Index>.csv``.

    Examples
    --------
    >>> from dtaianomaly.data import TSBADLoader
    >>> path_to_tsb_ad = "001_NAB_id_1_Facility_tr_1007_1st_2014.csv"
    >>> ucr_data_set = TSBADLoader(path_to_tsb_ad).load()  # doctest: +SKIP
    """

    def _load(self) -> DataSet:
        df = pd.read_csv(self.path)

        if self.train_index > self.first_anomaly:
            y_train = df["Label"].values[: self.train_index]
        else:
            y_train = None

        return DataSet(
            X_test=df.drop(columns="Label").values.squeeze(),
            y_test=df["Label"].values,
            X_train=df.drop(columns="Label").values[: self.train_index].squeeze(),
            y_train=y_train,
        )

    def _format_path(self) -> list[str]:
        return os.path.basename(self.path).split(".")[0].split("_")

    @property
    def index(self) -> int:
        return int(self._format_path()[0])

    @property
    def dataset_name(self) -> str:
        return self._format_path()[1]

    @property
    def id(self) -> int:
        return int(self._format_path()[3])

    @property
    def domain(self) -> str:
        return self._format_path()[4]

    @property
    def train_index(self) -> int:
        return int(self._format_path()[6])

    @property
    def first_anomaly(self) -> int:
        return int(self._format_path()[8])
