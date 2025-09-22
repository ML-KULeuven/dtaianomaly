import numpy as np
import pytest

from dtaianomaly.data import DataSet, PathDataLoader, from_directory


class DummyLoader(PathDataLoader):

    def _load(self) -> DataSet:
        return DataSet(
            X_test=np.random.uniform(size=(1000, 3)),
            y_test=np.random.choice([0, 1], p=[0.95, 0.05], size=1000, replace=True),
        )


class TestPathDataLoader:

    def test_invalid_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DummyLoader(tmp_path / "some" / "invalid" / "path")

    def test_valid_file(self, tmp_path):
        open(tmp_path / "valid-file", "a").close()  # Make the file
        loader = DummyLoader(tmp_path / "valid-file")
        assert loader.path == str(tmp_path / "valid-file")

    def test_valid_directory(self, tmp_path):
        loader = DummyLoader(tmp_path)
        assert loader.path == str(tmp_path)

    def test_str(self, tmp_path):
        assert str(DummyLoader(tmp_path)) == f"DummyLoader(path='{tmp_path}')"


class TestFromDirectory:

    def test_no_directory(self):
        with pytest.raises(FileNotFoundError):
            from_directory("some-invalid-path", DummyLoader)

    def test_file_given(self, tmp_path):
        open(tmp_path / "a-file", "a").close()  # Make the file
        with pytest.raises(FileNotFoundError):
            from_directory(tmp_path / "a-file", DummyLoader)

    def test_valid(self, tmp_path):
        paths = [str(tmp_path / f"file-{i}") for i in range(5)]
        for path in paths:
            open(path, "a").close()  # Make the file
        data_loaders = from_directory(tmp_path, DummyLoader)

        assert len(data_loaders) == len(paths)
        for data_loader in data_loaders:
            assert data_loader.path in paths

    @pytest.mark.parametrize("do_caching", [True, False])
    def test_kwargs(self, tmp_path, do_caching):
        paths = [str(tmp_path / f"file-{i}") for i in range(5)]
        for path in paths:
            open(path, "a").close()  # Make the file
        data_loaders = from_directory(tmp_path, DummyLoader, do_caching=do_caching)

        assert len(data_loaders) == len(paths)
        for data_loader in data_loaders:
            assert data_loader.path in paths
            assert data_loader.do_caching == do_caching
