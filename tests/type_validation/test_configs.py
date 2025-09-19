
import pytest
from conftest import ATTRIBUTE_VALIDATION_CONFIGS


@pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
def test_configuration(config):
    assert not config['validator']._is_valid_type(config['invalid_type'])
    assert not config['validator']._is_valid_value(config['invalid_value'])
    assert config['validator']._is_valid_type(config['valid'])
    assert config['validator']._is_valid_value(config['valid'])
