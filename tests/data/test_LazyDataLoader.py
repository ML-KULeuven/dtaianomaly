
import numpy as np
import time
from dtaianomaly.data import DataSet, LazyDataLoader


class CostlyDummyLoader(LazyDataLoader):
    NB_SECONDS_SLEEP = 1.5

    def _load(self) -> DataSet:
        time.sleep(self.NB_SECONDS_SLEEP)
        return DataSet(X_test=np.random.uniform(size=(1000, 3)), y_test=np.random.choice([0, 1], p=[0.95, 0.05], size=1000, replace=True))


class TestLazyDataLoader:

    def test_caching(self):
        loader = CostlyDummyLoader(do_caching=True)
        assert not hasattr(loader, 'cache_')

        # First load takes a long time
        start = time.time()
        loader.load()
        assert time.time() - start >= loader.NB_SECONDS_SLEEP
        assert hasattr(loader, 'cache_')

        # Second load is fast
        start = time.time()
        loader.load()
        assert time.time() - start < loader.NB_SECONDS_SLEEP

    def test_no_caching(self):
        loader = CostlyDummyLoader(do_caching=False)
        assert not hasattr(loader, 'cache_')

        # First load takes a long time
        start = time.time()
        loader.load()
        assert time.time() - start >= loader.NB_SECONDS_SLEEP
        assert not hasattr(loader, 'cache_')

        # Second load is also slow
        start = time.time()
        loader.load()
        assert time.time() - start >= loader.NB_SECONDS_SLEEP

