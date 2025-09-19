
import pytest

from conftest import ATTRIBUTE_VALIDATION_CONFIGS

from dtaianomaly.type_validation import (
    IntegerAttribute,
    BoolAttribute,
    LiteralAttribute,
    NoneAttribute,
    UnionAttribute
)


class TestUnionAttribute:

    @pytest.mark.parametrize("first", ATTRIBUTE_VALIDATION_CONFIGS)
    @pytest.mark.parametrize("second", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_attribute_validators_valid(self, first, second):
        if isinstance(first['validator'], UnionAttribute) or isinstance(second['validator'], UnionAttribute):
            return
        validator = UnionAttribute(first['validator'], second['validator'])
        assert len(validator.attribute_validators) == 2
        assert isinstance(validator.attribute_validators[0], first['validator'].__class__)
        assert isinstance(validator.attribute_validators[1], second['validator'].__class__)

    @pytest.mark.parametrize("first", ATTRIBUTE_VALIDATION_CONFIGS)
    @pytest.mark.parametrize("second", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_attribute_validators_valid_list(self, first, second):
        with pytest.raises(TypeError):
            UnionAttribute([first['validator'], second['validator']])

    @pytest.mark.parametrize("validators", [
        [5, LiteralAttribute('auto'), NoneAttribute()],
        [IntegerAttribute(minimum=5), "auto", NoneAttribute()],
        [IntegerAttribute(minimum=5), LiteralAttribute('auto'), None],
        [5, "auto", NoneAttribute()],
        [IntegerAttribute(minimum=5), "auto", None],
        [5, LiteralAttribute('auto'), None],
        [5, "auto", None],
    ])
    def test_attribute_validators_invalid_type(self, validators):
        with pytest.raises(TypeError):
            UnionAttribute(*validators)

    def test_flatten(self):
        validator = UnionAttribute(
            IntegerAttribute(),
            UnionAttribute(
                BoolAttribute(),
                NoneAttribute()
            )
        )
        assert len(validator.attribute_validators) == 3
        assert isinstance(validator.attribute_validators[0], IntegerAttribute)
        assert isinstance(validator.attribute_validators[1], BoolAttribute)
        assert isinstance(validator.attribute_validators[2], NoneAttribute)

    def test_flatten_multiple_times(self):
        validator = UnionAttribute(
            UnionAttribute(
                IntegerAttribute(),
                BoolAttribute()
            ),
            UnionAttribute(
                NoneAttribute(),
                LiteralAttribute("auto")
            )
        )
        assert len(validator.attribute_validators) == 4
        assert isinstance(validator.attribute_validators[0], IntegerAttribute)
        assert isinstance(validator.attribute_validators[1], BoolAttribute)
        assert isinstance(validator.attribute_validators[2], NoneAttribute)
        assert isinstance(validator.attribute_validators[3], LiteralAttribute)

    def test_flatten_multiple_layers(self):
        validator = UnionAttribute(
            IntegerAttribute(),
            UnionAttribute(
                BoolAttribute(),
                UnionAttribute(
                    NoneAttribute(),
                    LiteralAttribute("auto")
                )
            )
        )
        assert len(validator.attribute_validators) == 4
        assert isinstance(validator.attribute_validators[0], IntegerAttribute)
        assert isinstance(validator.attribute_validators[1], BoolAttribute)
        assert isinstance(validator.attribute_validators[2], NoneAttribute)
        assert isinstance(validator.attribute_validators[3], LiteralAttribute)

    @pytest.mark.parametrize("validators", [[], [NoneAttribute()]])
    def test_initialize_too_few(self, validators):
        with pytest.raises(ValueError):
            UnionAttribute(*validators)

    @pytest.mark.parametrize("validator,value,expected", [
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), 5, True),
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), 1.0, False),
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), True, False),
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), "auto", True),
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), [0, 1, 2, 3], False),
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), None, True),
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), {'a': 1, 'b': 2}, False),
    ])
    def test_is_valid_type(self, validator, value, expected):
        assert validator._is_valid_type(value) == expected

    def test_get_valid_type_description(self):
        validator = IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute()
        assert validator._get_valid_type_description() == "int, string or None"

    @pytest.mark.parametrize("validator,value,is_valid", [
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), 5, True),
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), 4, False),
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), "auto", True),
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), "other", False),
        (IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute(), None, True),
    ])
    def test_is_valid_value(self, validator, value, is_valid):
        assert validator._is_valid_value(value) == is_valid

    def test_get_valid_value_description(self):
        validator = IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute()
        assert validator._get_valid_value_description() == "greater than or equal to 5, in {'auto'} or None"
