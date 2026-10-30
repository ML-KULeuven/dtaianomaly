import os
import re
from pathlib import Path

import pandas as pd

from dtaianomaly.data._DataSet import DataSet
from dtaianomaly.data._PathDataLoader import PathDataLoader

__all__ = ["TSBADLoader"]

_FILE_NAME_PATTERN = re.compile(
    r"^(\d+)_([^_]+)_id_(\d+)_([^_]+)_tr_(\d+)_1st_(\d+)\.csv$"
)


class TSBADLoader(PathDataLoader):
    """
    Lazy dataloader for the TSB-AD benchmark :cite:`liu2024elephant`.

    The TSB-AD benchmark comprises 870 univariate time series and 200 multivariate
    time series, as well as predefined tune and evaluation sets. The benchmark has
    been constructed through the following steps:

    1. **Dataset construction.** Collect 13 univariate and 20 multivariate publicly
       available datasets. The multivariate time series are also transformed to multiple
       univariate time series by assuming channel independence, in which channels with
       low correlation to the anomaly scores are ignored.
    2. **Flaw identification.** A human annotator excludes several time series that have
       some common flaws and could induce a bias and inaccurate evaluation.
    3. **Label quality assessment.** The quality of the anomaly scores are assessed through
       an algorithmic test, in which at least one anomaly detector should be able to locate
       the anomalies.

    A tune and evaluation set have been defined. These are available at
    https://github.com/TheDatumOrg/TSB-AD/tree/main/Datasets/File_List.

    Parameters
    ----------
    path : str
        The path at which the data set is located.
    do_caching : bool, default=False
        Whether to cache the loaded data or not.

    Raises
    ------
    ValueError
        If the name of the file does not have the expected format.

    Notes
    -----
    This implementation expects the file names to have the following form:
    ``<index>_<Dataset Name>_id_<id>_<Domain>_tr_<Train Index>_1st_<First Anomaly Index>.csv``.

    Following the TSB-AD protocol, ``X_test`` is the full time series, including
    the training part. ``X_train`` consists of the first ``<Train Index>`` observations.
    The training labels ``y_train`` are only given if the training part contains an
    anomaly, i.e., if ``<Train Index>`` is larger than ``<First Anomaly Index>``.
    The data always has two dimensions, also for univariate time series.

    Examples
    --------
    >>> from dtaianomaly.data import TSBADLoader
    >>> path_to_tsb_ad = "001_NAB_id_1_Facility_tr_1007_1st_2014.csv"
    >>> tsb_ad_data_set = TSBADLoader(path_to_tsb_ad).load()  # doctest: +SKIP
    """

    def __init__(self, path: str | Path, do_caching: bool = False):
        super().__init__(path, do_caching)
        if _FILE_NAME_PATTERN.match(os.path.basename(self.path)) is None:
            raise ValueError(
                f"The file name of '{self.path}' does not have the TSB-AD format "
                "'<index>_<Dataset Name>_id_<id>_<Domain>_tr_<Train Index>_1st_<First Anomaly Index>.csv'."
            )

    def _load(self) -> DataSet:
        df = pd.read_csv(self.path)
        X = df.drop(columns="Label").values
        y = df["Label"].values

        if self.train_index > self.first_anomaly:
            y_train = y[: self.train_index]
        else:
            y_train = None

        return DataSet(
            X_test=X,
            y_test=y,
            X_train=X[: self.train_index],
            y_train=y_train,
        )

    def _format_path(self) -> tuple[str, ...]:
        return _FILE_NAME_PATTERN.match(os.path.basename(self.path)).groups()

    @property
    def index(self) -> int:
        return int(self._format_path()[0])

    @property
    def dataset_name(self) -> str:
        return self._format_path()[1]

    @property
    def id(self) -> int:
        return int(self._format_path()[2])

    @property
    def domain(self) -> str:
        return self._format_path()[3]

    @property
    def train_index(self) -> int:
        return int(self._format_path()[4])

    @property
    def first_anomaly(self) -> int:
        return int(self._format_path()[5])
