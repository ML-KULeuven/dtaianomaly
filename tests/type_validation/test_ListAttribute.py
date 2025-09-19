
import pytest

from conftest import ATTRIBUTE_VALIDATION_CONFIGS

from dtaianomaly.type_validation import (
    IntegerAttribute,
    LiteralAttribute,
    ListAttribute,
    NoneAttribute,
    UnionAttribute
)


class TestListAttribute:

    def test_initialize_default(self):
        with pytest.raises(TypeError):
            ListAttribute()

    @pytest.mark.parametrize('config', ATTRIBUTE_VALIDATION_CONFIGS)
    def test_validator_valid(self, config):
        assert isinstance(ListAttribute(config['validator']).validator, config['validator'].__class__)

    @pytest.mark.parametrize('validator', [5, 1.0, True, "auto", [0, 1, 2, 2], {'a': 1, 'b': 2}])
    def test_validator_invalid(self, validator):
        with pytest.raises(TypeError):
            ListAttribute(validator)

    def test_validator_immutable(self):
        validator = ListAttribute(IntegerAttribute())
        with pytest.raises(AttributeError):
            validator.validator = NoneAttribute()

    @pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_is_valid_type(self, config):
        # Valid
        assert ListAttribute(config['validator'])._is_valid_type([config['valid']])
        assert ListAttribute(config['validator'])._is_valid_type(5 * [config['valid']])

        # Invalid
        assert not ListAttribute(config['validator'])._is_valid_type(config['valid'])  # Not a list of the requested type
        assert not ListAttribute(config['validator'])._is_valid_type([config['invalid_type']])
        assert not ListAttribute(config['validator'])._is_valid_type(5 * [config['invalid_type']])
        one_invalid = 5 * [config['valid']]
        one_invalid[3] = config['invalid_type']
        assert not ListAttribute(config['validator'])._is_valid_type(one_invalid)

    @pytest.mark.parametrize('config', ATTRIBUTE_VALIDATION_CONFIGS)
    def test_get_valid_type_description(self, config):
        assert ListAttribute(config['validator'])._get_valid_type_description() == f"list of {config['validator']._get_valid_type_description()}"

    def test_get_valid_type_description_defined(self):
        assert ListAttribute(IntegerAttribute())._get_valid_type_description() == "list of int"

    @pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_is_valid_value(self, config):
        # Valid
        assert ListAttribute(config['validator'])._is_valid_value([config['valid']])
        assert ListAttribute(config['validator'])._is_valid_value(5 * [config['valid']])

        # Invalid
        assert not ListAttribute(config['validator'])._is_valid_value([config['invalid_value']])
        assert not ListAttribute(config['validator'])._is_valid_value(5 * [config['invalid_value']])
        one_invalid = 5 * [config['valid']]
        one_invalid[3] = config['invalid_value']
        assert not ListAttribute(config['validator'])._is_valid_value(one_invalid)

    @pytest.mark.parametrize('config', ATTRIBUTE_VALIDATION_CONFIGS)
    def test_get_valid_value_description(self, config):
        if isinstance(config['validator'], (NoneAttribute, UnionAttribute)):
            return
        assert ListAttribute(config['validator'])._get_valid_value_description() == f"list of {config['validator']._get_valid_type_description()} {config['validator']._get_valid_value_description()}"

    def test_get_valid_value_description_defined(self):
        assert ListAttribute(IntegerAttribute(minimum=5))._get_valid_value_description() == "list of int greater than or equal to 5"

    def test_get_valid_value_description_none(self):
        assert ListAttribute(NoneAttribute())._get_valid_value_description() == "list of None"

    def test_get_valid_value_description_union(self):
        assert ListAttribute(IntegerAttribute(minimum=5) | LiteralAttribute("auto"))._get_valid_value_description() \
               == "list of int greater than or equal to 5 or string in {'auto'}"

    def test_get_valid_value_description_union_3(self):
        assert ListAttribute(IntegerAttribute(minimum=5) | LiteralAttribute("auto") | NoneAttribute())._get_valid_value_description() \
               == "list of int greater than or equal to 5, string in {'auto'} or None"
