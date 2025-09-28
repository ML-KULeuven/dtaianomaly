from pathlib import Path

import pytest

from dtaianomaly.type_validation import PathAttribute


class TestPathAttribute:

    def test_initialize_default(self):
        assert PathAttribute().must_exist

    @pytest.mark.parametrize("must_exist", [True, False])
    def test_must_exist_valid(self, must_exist):
        assert PathAttribute(must_exist=must_exist).must_exist == must_exist

    @pytest.mark.parametrize(
        "must_exist", [1, 1.5, "auto", [0, 1, 2, 2], {"a": 1, "b": 2}]
    )
    def test_must_exist_invalid(self, must_exist):
        with pytest.raises(TypeError):
            PathAttribute(must_exist=must_exist)

    def test_must_exist_immutable(self):
        validator = PathAttribute(must_exist=True)
        with pytest.raises(AttributeError):
            validator.must_exist = False

    @pytest.mark.parametrize(
        "value,expected",
        [
            (5, False),
            (1.0, False),
            (True, False),
            (Path("."), True),
            (".", True),
            (
                "nonexistent_file_should_not_exist_12345.txt",
                True,
            ),  # valid type, existence checked separately
            ([0, 1, 2, 3], False),
            (None, False),
            ({"a": 1, "b": 2}, False),
        ],
    )
    @pytest.mark.parametrize("must_exist", [True, False])
    def test_is_valid_type(self, value, expected, must_exist):
        assert PathAttribute(must_exist)._is_valid_type(value) == expected

    @pytest.mark.parametrize("must_exist", [True, False])
    def test_get_valid_type_description(self, must_exist):
        assert (
            PathAttribute(must_exist)._get_valid_type_description()
            == "str or pathlib.Path"
        )

    @pytest.mark.parametrize("must_exist", [True, False])
    def test_is_valid_value_existing_path(self, tmp_path, must_exist):
        validator = PathAttribute(must_exist)
        assert validator._is_valid_value(tmp_path) is True
        assert validator._is_valid_value(str(tmp_path)) is True

    @pytest.mark.parametrize("must_exist", [True, False])
    def test_is_valid_value_non_existing_path(self, tmp_path, must_exist):
        non_existing = tmp_path / "nonexistent_file.txt"
        validator = PathAttribute(must_exist=must_exist)
        assert validator._is_valid_value(non_existing) is not must_exist
        assert validator._is_valid_value(str(non_existing)) is not must_exist

    def test_get_valid_value_description_must_exist(self):
        assert PathAttribute(True)._get_valid_value_description() == "an existing path"

    def test_get_valid_value_description_must_not_exist(self):
        assert PathAttribute(False)._get_valid_value_description() == "a path"
