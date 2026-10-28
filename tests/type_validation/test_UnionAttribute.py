import pytest
from conftest import ATTRIBUTE_VALIDATION_CONFIGS

from dtaianomaly.type_validation import (
    BoolAttribute,
    IntegerAttribute,
    LiteralAttribute,
    NoneAttribute,
    UnionAttribute,
)

_NON_UNION_VALIDATORS = [
    config["validator"]
    for config in ATTRIBUTE_VALIDATION_CONFIGS
    if not isinstance(config["validator"], UnionAttribute)
]

# Every validator once as first and once as second element, plus one identical pair
_VALIDATOR_PAIRS = [
    pytest.param(first, second, id=f"{type(first).__name__}-{type(second).__name__}")
    for first, second in zip(
        _NON_UNION_VALIDATORS, _NON_UNION_VALIDATORS[1:] + _NON_UNION_VALIDATORS[:1]
    )
] + [pytest.param(IntegerAttribute(), IntegerAttribute(), id="same-class")]


class TestUnionAttribute:

    @pytest.mark.parametrize("first,second", _VALIDATOR_PAIRS)
    def test_attribute_validators_valid(self, first, second):
        validator = UnionAttribute(first, second)
        assert len(validator.attribute_validators) == 2
        assert isinstance(validator.attribute_validators[0], first.__class__)
        assert isinstance(validator.attribute_validators[1], second.__class__)

    @pytest.mark.parametrize("first,second", _VALIDATOR_PAIRS)
    def test_attribute_validators_valid_list(self, first, second):
        with pytest.raises(TypeError):
            UnionAttribute([first, second])

    @pytest.mark.parametrize(
        "validators",
        [
            [5, LiteralAttribute("auto"), NoneAttribute()],
            [IntegerAttribute(minimum=5), "auto", NoneAttribute()],
            [IntegerAttribute(minimum=5), LiteralAttribute("auto"), None],
            [5, "auto", NoneAttribute()],
            [IntegerAttribute(minimum=5), "auto", None],
            [5, LiteralAttribute("auto"), None],
            [5, "auto", None],
        ],
    )
    def test_attribute_validators_invalid_type(self, validators):
        with pytest.raises(TypeError):
            UnionAttribute(*validators)

    def test_flatten(self):
        validator = UnionAttribute(
            IntegerAttribute(), UnionAttribute(BoolAttribute(), NoneAttribute())
        )
        assert len(validator.attribute_validators) == 3
        assert isinstance(validator.attribute_validators[0], IntegerAttribute)
        assert isinstance(validator.attribute_validators[1], BoolAttribute)
        assert isinstance(validator.attribute_validators[2], NoneAttribute)

    def test_flatten_multiple_times(self):
        validator = UnionAttribute(
            UnionAttribute(IntegerAttribute(), BoolAttribute()),
            UnionAttribute(NoneAttribute(), LiteralAttribute("auto")),
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
                UnionAttribute(NoneAttribute(), LiteralAttribute("auto")),
            ),
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

    @pytest.mark.parametrize(
        "validator,value,expected",
        [
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                5,
                True,
            ),
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                1.0,
                False,
            ),
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                True,
                False,
            ),
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                "auto",
                True,
            ),
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                [0, 1, 2, 3],
                False,
            ),
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                None,
                True,
            ),
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                {"a": 1, "b": 2},
                False,
            ),
        ],
    )
    def test_is_valid_type(self, validator, value, expected):
        assert validator._is_valid_type(value) == expected

    def test_get_valid_type_description(self):
        validator = (
            IntegerAttribute(minimum=5) | LiteralAttribute("auto") | NoneAttribute()
        )
        assert validator._get_valid_type_description() == "int, string or None"

    @pytest.mark.parametrize(
        "validator,value,is_valid",
        [
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                5,
                True,
            ),
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                4,
                False,
            ),
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                "auto",
                True,
            ),
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                "other",
                False,
            ),
            (
                IntegerAttribute(minimum=5)
                | LiteralAttribute("auto")
                | NoneAttribute(),
                None,
                True,
            ),
        ],
    )
    def test_is_valid_value(self, validator, value, is_valid):
        assert validator._is_valid_value(value) == is_valid

    def test_get_valid_value_description(self):
        validator = (
            IntegerAttribute(minimum=5) | LiteralAttribute("auto") | NoneAttribute()
        )
        assert (
            validator._get_valid_value_description()
            == "greater than or equal to 5, in {'auto'} or None"
        )
