import pytest

from dtaianomaly.type_validation import LiteralAttribute


class TestLiteralAttribute:

    def test_initialize_default(self):
        with pytest.raises(ValueError):
            LiteralAttribute()

    @pytest.mark.parametrize(
        "values",
        [
            ["one", "two", "three"],
            ["auto"],
        ],
    )
    def test_values_valid(self, values):
        assert LiteralAttribute(*values).values == set(values)

    @pytest.mark.parametrize(
        "values",
        [
            ["one", "two", "three"],
            ["auto"],
        ],
    )
    def test_values_valid_list(self, values):
        assert LiteralAttribute(values).values == set(values)

    @pytest.mark.parametrize("values", [5, 1.0, True, [0, 1, 2, 2], {"a": 1, "b": 2}])
    def test_values_invalid(self, values):
        with pytest.raises(TypeError):
            LiteralAttribute(values)

    def test_values_empty_list(self):
        with pytest.raises(ValueError):
            LiteralAttribute([])

    def test_values_immutable(self):
        validator = LiteralAttribute("one", "two", "three")
        with pytest.raises(AttributeError):
            validator.values = {"one", "two"}

    @pytest.mark.parametrize(
        "value,expected",
        [
            (5, False),
            (1.0, False),
            (True, False),
            ("literal", True),
            ([0, 1, 2, 3], False),
            (None, False),
            ({"a": 1, "b": 2}, False),
        ],
    )
    def test_is_valid_type(self, value, expected):
        assert LiteralAttribute("auto")._is_valid_type(value) == expected

    def test_get_valid_type_description(self):
        assert LiteralAttribute("auto")._get_valid_type_description() == "string"

    @pytest.mark.parametrize(
        "values,value,is_valid",
        [
            (["a", "b", "c"], "a", True),
            (["a", "b", "c"], "b", True),
            (["a", "b", "c"], "c", True),
            (["a", "b", "c"], "d", False),
        ],
    )
    def test_is_valid_value(self, values, value, is_valid):
        assert LiteralAttribute(*values)._is_valid_value(value) == is_valid

    @pytest.mark.parametrize(
        "values,string",
        [
            (["a"], "in {'a'}"),
            (["a", "b"], "in {'a', 'b'}"),
            (["a", "b", "c"], "in {'a', 'b', 'c'}"),
            (["a", "b", "c", "d"], "in {'a', 'b', 'c', 'd'}"),
        ],
    )
    def test_get_valid_value_description(self, values, string):
        assert LiteralAttribute(values)._get_valid_value_description() == string
