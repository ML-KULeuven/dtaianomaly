from pathlib import Path

import pytest

from dtaianomaly.type_validation import PathAttribute


class TestPathAttribute:

    def test_initialize_default(self):
        PathAttribute()  # Should not raise an error

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
    def test_is_valid_type(self, value, expected):
        assert PathAttribute()._is_valid_type(value) == expected

    def test_get_valid_type_description(self):
        assert PathAttribute()._get_valid_type_description() == "str or pathlib.Path"

    def test_is_valid_value_existing_path(self, tmp_path):
        validator = PathAttribute()
        assert validator._is_valid_value(tmp_path) is True
        assert validator._is_valid_value(str(tmp_path)) is True

    def test_is_valid_value_non_existing_path(self, tmp_path):
        non_existing = tmp_path / "nonexistent_file.txt"
        validator = PathAttribute()
        assert validator._is_valid_value(non_existing) is False
        assert validator._is_valid_value(str(non_existing)) is False

    def test_get_valid_value_description(self):
        assert PathAttribute()._get_valid_value_description() == "an existing path"
