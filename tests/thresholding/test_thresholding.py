import inspect

import numpy as np
import pytest

from dtaianomaly.thresholding import Thresholding
from dtaianomaly.utils import all_classes


def initialize(cls):
    kwargs = {
        "cutoff": 0.9,
        "contamination_rate": 0.1,
        "n": 1,
    }
    sig = inspect.signature(cls.__init__)
    accepted_params = set(sig.parameters) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    return cls(**filtered_kwargs)


@pytest.mark.parametrize("cls", all_classes(Thresholding, return_names=False))
class TestThresholding:

    def test_is_valid(self, cls):
        thresholding = initialize(cls)
        scores = np.random.default_rng().uniform(0, 1, size=1000)
        labels = thresholding.threshold(scores)
        assert labels.shape[0] == 1000
        assert labels.min() == 0
        assert labels.max() == 1
        assert set(labels) == {0, 1}

    def test_list(self, cls):
        thresholding = initialize(cls)
        scores = np.random.default_rng().uniform(0, 1, size=(1000, 2))
        with pytest.raises(ValueError):
            thresholding.threshold(list(scores))

    def test_invalid(self, cls):
        thresholding = initialize(cls)
        with pytest.raises(ValueError):
            thresholding.threshold([0.0, "0.9", 1.0])

    def test_multivariate(self, cls):
        thresholding = initialize(cls)
        scores = np.random.default_rng().uniform(0, 1, size=(1000, 2))
        with pytest.raises(ValueError):
            thresholding.threshold(scores)

    def test_is_invalid_array_like_first_dim_1(self, cls):
        thresholding = initialize(cls)
        scores = np.random.default_rng().uniform(0, 1, size=(1, 1000))
        labels = thresholding.threshold(scores)
        assert labels.shape[0] == 1000
        assert labels.min() == 0
        assert labels.max() == 1
        assert set(labels) == {0, 1}

    def test_is_invalid_array_like_second_dim_1(self, cls):
        thresholding = initialize(cls)
        scores = np.random.default_rng().uniform(0, 1, size=(1000, 1))
        labels = thresholding.threshold(scores)
        assert labels.shape[0] == 1000
        assert labels.min() == 0
        assert labels.max() == 1
        assert set(labels) == {0, 1}
