
import numpy as np
import pytest
import time
from dtaianomaly.data import LazyDataLoader, DataSet, from_directory


class DummyLoader(LazyDataLoader):

    def _load(self) -> DataSet:
        return DataSet(x=np.array([]), y=np.array([]))


class TestLazyDataLoader:

    def test_invalid_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader = DummyLoader(tmp_path / 'some' / 'invalid' / 'path')

    def test_valid_file(self, tmp_path):
        open(tmp_path / 'valid-file', 'a').close()  # Make the file
        loader = DummyLoader(tmp_path / 'valid-file')
        assert loader.path == str(tmp_path / 'valid-file')

    def test_valid_directory(self, tmp_path):
        loader = DummyLoader(tmp_path)
        assert loader.path == str(tmp_path)

    def test_str(self, tmp_path):
        assert str(DummyLoader(tmp_path)) == f"DummyLoader(path='{tmp_path}')"


class CostlyDummyLoader(LazyDataLoader):
    NB_SECONDS_SLEEP = 1.5

    def _load(self) -> DataSet:
        time.sleep(self.NB_SECONDS_SLEEP)
        return DataSet(x=np.array([]), y=np.array([]))


class TestCaching:

    def test_caching(self):
        loader = CostlyDummyLoader(path='.', do_caching=True)
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
        loader = CostlyDummyLoader(path='.', do_caching=False)
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


class TestFromDirectory:

    def test_no_directory(self):
        with pytest.raises(FileNotFoundError):
            from_directory('some-invalid-path', DummyLoader)

    def test_file_given(self, tmp_path):
        open(tmp_path / 'a-file', 'a').close()  # Make the file
        with pytest.raises(FileNotFoundError):
            from_directory(tmp_path / 'a-file', DummyLoader)

    def test_valid(self, tmp_path):
        paths = [str(tmp_path / f'file-{i}') for i in range(5)]
        for path in paths:
            open(path, 'a').close()  # Make the file
        data_loaders = from_directory(tmp_path, DummyLoader)

        assert len(data_loaders) == len(paths)
        for data_loader in data_loaders:
            assert data_loader.path in paths
