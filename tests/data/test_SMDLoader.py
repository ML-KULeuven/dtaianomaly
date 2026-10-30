import os

import numpy as np
import pytest

from dtaianomaly.data import SMDLoader

SMD_DATA_PATH = "data/SMD"
SMD_MACHINE = "machine-1-1"

data_available = pytest.mark.skipif(
    not all(
        os.path.isfile(f"{SMD_DATA_PATH}/{folder}/{SMD_MACHINE}.txt")
        for folder in ("train", "test", "test_label")
    ),
    reason="SMD files unavailable",
)


def _write_smd_machine(root, machine="machine-9-9", n_attributes=1, skip=()):
    rng = np.random.default_rng(0)
    data = {
        "train": rng.normal(size=(8, n_attributes)),
        "test": rng.normal(size=(10, n_attributes)),
        "test_label": np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
    }
    for folder, values in data.items():
        (root / folder).mkdir(exist_ok=True)
        if folder not in skip:
            fmt = "%d" if folder == "test_label" else "%.6f"
            np.savetxt(root / folder / f"{machine}.txt", values, fmt=fmt, delimiter=",")
    return root


@pytest.fixture
def loader():
    return SMDLoader(SMD_DATA_PATH, SMD_MACHINE)


@pytest.fixture
def loaded(loader):
    return loader.load()


class TestSMDLoader:

    @data_available
    def test_train_shape(self, loaded):
        assert loaded.X_train.shape == (28479, 38)

    @data_available
    def test_test_shape(self, loaded):
        assert loaded.X_test.shape == (28479, 38)

    @data_available
    def test_samples_match(self, loaded):
        assert loaded.X_test.shape[0] == loaded.y_test.shape[0]

    @data_available
    def test_contains_anomaly(self, loaded):
        assert np.sum(loaded.y_test == 1) > 0

    @data_available
    def test_no_train_labels(self, loaded):
        assert loaded.y_train is None

    def test_single_attribute_keeps_two_dimensions(self, tmp_path):
        loaded = SMDLoader(_write_smd_machine(tmp_path), "machine-9-9").load()
        assert loaded.X_test.shape == (10, 1)
        assert loaded.X_train.shape == (8, 1)

    def test_labels_are_one_dimensional(self, tmp_path):
        loaded = SMDLoader(_write_smd_machine(tmp_path), "machine-9-9").load()
        assert np.array_equal(loaded.y_test, [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])

    def test_faulty_path(self):
        with pytest.raises(ValueError):
            SMDLoader(path="bollocks", machine=SMD_MACHINE)

    @pytest.mark.parametrize(
        "machine",
        ["machine-1", "machine-1-1.txt", "machine-a-1", "Machine-1-1", "", 11, None],
    )
    def test_invalid_machine(self, tmp_path, machine):
        with pytest.raises(ValueError, match="Invalid machine"):
            SMDLoader(_write_smd_machine(tmp_path), machine)

    def test_missing_machine(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="machine-1-2"):
            SMDLoader(_write_smd_machine(tmp_path), "machine-1-2")

    @pytest.mark.parametrize("folder", ["train", "test", "test_label"])
    def test_missing_file(self, tmp_path, folder):
        path = _write_smd_machine(tmp_path, skip=(folder,))
        with pytest.raises(FileNotFoundError, match="No such file"):
            SMDLoader(path, "machine-9-9")

    def test_str_contains_machine(self, tmp_path):
        loader = SMDLoader(_write_smd_machine(tmp_path), "machine-9-9")
        assert "machine='machine-9-9'" in str(loader)
