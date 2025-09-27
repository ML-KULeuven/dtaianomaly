import pytest

from dtaianomaly.type_validation import ObjectAttribute


class TestLiteralAttribute:

    values_to_check_if_valid = [
        (int, 5, True),
        (int, 5.0, False),
        (int, True, False),
        (str, "a-string", True),
        (str, 5, False),
        (bool, True, True),
        (bool, 5, False),
    ]

    def test_initialize_default(self):
        with pytest.raises(TypeError):
            ObjectAttribute()

    @pytest.mark.parametrize("object_type", [int, str, list, bool, float])
    def test_values_valid(self, object_type):
        assert ObjectAttribute(object_type).object_type == object_type

    @pytest.mark.parametrize(
        "object_type", [5, 1.0, True, [0, 1, 2, 2], {"a": 1, "b": 2}]
    )
    def test_values_invalid(self, object_type):
        with pytest.raises(TypeError):
            ObjectAttribute(object_type)

    def test_values_immutable(self):
        validator = ObjectAttribute(int)
        with pytest.raises(AttributeError):
            validator.object_type = str

    @pytest.mark.parametrize("object_type,value,expected", values_to_check_if_valid)
    def test_is_valid_type(self, object_type, value, expected):
        assert ObjectAttribute(object_type)._is_valid_type(value) == expected

    def test_get_valid_type_description(self):
        assert ObjectAttribute(int)._get_valid_type_description() == f"'type {int}'"

    @pytest.mark.parametrize("object_type,value,expected", values_to_check_if_valid)
    def test_is_valid_value(self, object_type, value, expected):
        assert ObjectAttribute(object_type)._is_valid_value(value) == expected

    def test_get_valid_value_description(self):
        assert ObjectAttribute(int)._get_valid_value_description() == f"'type {int}'"

    def test_custom_object(self):

        class MyObject:
            pass

        assert ObjectAttribute(MyObject)._is_valid_type(MyObject())

    def test_inheritance(self):

        class Parent:
            pass

        class Child(Parent):
            pass

        assert ObjectAttribute(Parent)._is_valid_type(Child())
        assert not ObjectAttribute(Child)._is_valid_type(Parent())
