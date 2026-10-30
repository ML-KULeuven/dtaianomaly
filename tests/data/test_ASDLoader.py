import collections
import os
import pickle

import numpy as np
import pytest

from dtaianomaly.data import ASDLoader

ASD_DATA_PATH = "data/ASD"
ASD_MACHINE = "omi-1"

data_available = pytest.mark.skipif(
    not all(
        os.path.isfile(f"{ASD_DATA_PATH}/processed/{ASD_MACHINE}_{part}.pkl")
        for part in ("train", "test", "test_label")
    ),
    reason="ASD files unavailable",
)


class _PrintOnUnpickle:
    def __reduce__(self):
        return print, ("x",)


def _write_asd_machine(
    root, machine="omi-99", n_attributes=1, skip=(), overrides=None, protocol=None
):
    rng = np.random.default_rng(0)
    data = {
        "train": rng.normal(size=(8, n_attributes)).squeeze(),
        "test": rng.normal(size=(10, n_attributes)).squeeze(),
        "test_label": np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
    }
    data.update(overrides or {})
    (root / "processed").mkdir(exist_ok=True)
    for part, values in data.items():
        if part not in skip:
            with open(root / "processed" / f"{machine}_{part}.pkl", "wb") as f:
                pickle.dump(values, f, protocol=protocol)
    return root


@pytest.fixture
def loader():
    return ASDLoader(ASD_DATA_PATH, ASD_MACHINE)


@pytest.fixture
def loaded(loader):
    return loader.load()


class TestASDLoader:

    @data_available
    def test_train_shape(self, loaded):
        assert loaded.X_train.shape == (8640, 19)

    @data_available
    def test_test_shape(self, loaded):
        assert loaded.X_test.shape == (4320, 19)

    @data_available
    def test_samples_match(self, loaded):
        assert len(loaded.y_test) == len(loaded.X_test)

    @data_available
    def test_labels_are_binary(self, loaded):
        assert set(np.unique(loaded.y_test)) == {0, 1}

    @data_available
    def test_number_of_anomalies(self, loaded):
        assert np.sum(loaded.y_test == 1) == 441

    @data_available
    def test_no_train_labels(self, loaded):
        assert loaded.y_train is None

    @data_available
    def test_dtypes(self, loaded):
        assert np.issubdtype(loaded.X_train.dtype, np.floating)
        assert np.issubdtype(loaded.X_test.dtype, np.floating)
        assert np.issubdtype(loaded.y_test.dtype, np.integer)

    def test_single_attribute_keeps_two_dimensions(self, tmp_path):
        loaded = ASDLoader(_write_asd_machine(tmp_path), "omi-99").load()
        assert loaded.X_test.shape == (10, 1)
        assert loaded.X_train.shape == (8, 1)

    def test_multiple_attributes(self, tmp_path):
        loaded = ASDLoader(
            _write_asd_machine(tmp_path, n_attributes=3), "omi-99"
        ).load()
        assert loaded.X_test.shape == (10, 3)
        assert loaded.X_train.shape == (8, 3)

    def test_labels_are_one_dimensional(self, tmp_path):
        loaded = ASDLoader(_write_asd_machine(tmp_path), "omi-99").load()
        assert np.array_equal(loaded.y_test, [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])

    def test_synthetic_dtypes(self, tmp_path):
        overrides = {"train": np.arange(8, dtype=np.int32)}
        loaded = ASDLoader(
            _write_asd_machine(tmp_path, overrides=overrides), "omi-99"
        ).load()
        assert np.issubdtype(loaded.X_train.dtype, np.floating)
        assert np.issubdtype(loaded.y_test.dtype, np.integer)

    def test_synthetic_no_train_labels(self, tmp_path):
        loaded = ASDLoader(_write_asd_machine(tmp_path), "omi-99").load()
        assert loaded.y_train is None

    @pytest.mark.parametrize("protocol", [2, 3, 4, 5])
    def test_pickle_protocols(self, tmp_path, protocol):
        path = _write_asd_machine(tmp_path, protocol=protocol)
        assert ASDLoader(path, "omi-99").load().X_test.shape == (10, 1)

    def test_faulty_path(self):
        with pytest.raises(ValueError, match="must be an existing path"):
            ASDLoader(path="bollocks", machine=ASD_MACHINE)

    @pytest.mark.parametrize(
        "machine",
        ["machine-1-1", "omi-x", "omi-", "omi-1.pkl", "OMI-1", "omi-1_train", "", 1],
    )
    def test_invalid_machine(self, tmp_path, machine):
        with pytest.raises(ValueError, match="Invalid machine"):
            ASDLoader(_write_asd_machine(tmp_path), machine)

    def test_missing_machine(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="omi-2"):
            ASDLoader(_write_asd_machine(tmp_path), "omi-2")

    @pytest.mark.parametrize("part", ["train", "test", "test_label"])
    def test_missing_file(self, tmp_path, part):
        path = _write_asd_machine(tmp_path, skip=(part,))
        with pytest.raises(FileNotFoundError, match=f"omi-99_{part}.pkl"):
            ASDLoader(path, "omi-99")

    @pytest.mark.parametrize("part", ["train", "test", "test_label"])
    @pytest.mark.parametrize(
        "value", [collections.OrderedDict(a=1), _PrintOnUnpickle()]
    )
    def test_disallowed_global(self, tmp_path, part, value):
        path = _write_asd_machine(tmp_path, overrides={part: value})
        loader = ASDLoader(path, "omi-99")
        with pytest.raises(pickle.UnpicklingError, match="is not allowed"):
            loader.load()

    def test_disallowed_global_is_not_called(self, tmp_path, capsys):
        path = _write_asd_machine(tmp_path, overrides={"train": _PrintOnUnpickle()})
        with pytest.raises(pickle.UnpicklingError):
            ASDLoader(path, "omi-99").load()
        assert capsys.readouterr().out == ""

    def test_str_contains_machine(self, tmp_path):
        loader = ASDLoader(_write_asd_machine(tmp_path), "omi-99")
        assert "machine='omi-99'" in str(loader)
