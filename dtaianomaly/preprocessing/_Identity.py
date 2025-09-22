import numpy as np

from dtaianomaly.preprocessing._Preprocessor import Preprocessor


class Identity(Preprocessor):
    """
    Identity preprocessor.

    A dummy preprocessor which does not do any processing at all.

    Examples
    --------
    >>> from dtaianomaly.preprocessing import Identity
    >>> from dtaianomaly.data import demonstration_time_series
    >>> X, y = demonstration_time_series()
    >>> preprocessor = Identity()
    >>> X_, y_ = preprocessor.fit_transform(X, y)
    """

    def _fit(self, X: np.ndarray, y: np.ndarray = None) -> "Preprocessor":
        return self

    def _transform(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> (np.ndarray, np.ndarray | None):
        return X, y
