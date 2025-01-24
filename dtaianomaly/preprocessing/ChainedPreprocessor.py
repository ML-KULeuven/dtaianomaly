import numpy as np
from typing import List, Optional, Tuple, Union

from dtaianomaly import utils
from dtaianomaly.preprocessing.Preprocessor import (
    Preprocessor,
    check_preprocessing_inputs,
)


class ChainedPreprocessor(Preprocessor):
    """
    Wrapper chaining multiple `Preprocessor` objects.

    Parameters
    ----------
    base_preprocessors: list of `Preprocessor` objects
        The preprocessors to chain. These preprocessors can be passed as a single
        list argument or as multiple independent arguments to the constructor.
    """

    base_preprocessors: List[Preprocessor]

    def __init__(self, *base_preprocessors: Union[Preprocessor, List[Preprocessor]]):
        # Format the base processors
        if len(base_preprocessors) == 1 and isinstance(base_preprocessors[0], list):
            base_preprocessors = base_preprocessors[0]
        else:
            base_preprocessors = list(base_preprocessors)
        # Check the preprocessors
        if (
            not utils.is_valid_list(base_preprocessors, Preprocessor)
            or len(base_preprocessors) == 0
        ):
            raise ValueError("Expected a list of Preprocessors")
        self.base_preprocessors = base_preprocessors

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Preprocessor":
        for preprocessor in self.base_preprocessors:
            preprocessor._fit(X, y)
            X, y = preprocessor._transform(X, y)
        return self

    def _transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        for preprocessor in self.base_preprocessors:
            X, y = preprocessor._transform(X, y)
        return X, y

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        check_preprocessing_inputs(X, y)
        for preprocessor in self.base_preprocessors:
            preprocessor._fit(X, y)
            X, y = preprocessor._transform(X, y)
        return X, y

    def __str__(self):
        return "->".join(map(str, self.base_preprocessors))
