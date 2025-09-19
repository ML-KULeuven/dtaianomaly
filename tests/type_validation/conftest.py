
from dtaianomaly.type_validation import (
    IntegerAttribute,
    FloatAttribute,
    BoolAttribute,
    LiteralAttribute,
    ListAttribute,
    NoneAttribute
)

ATTRIBUTE_VALIDATION_CONFIGS = [
    {
        "validator": NoneAttribute(),
        "invalid_type": 0,
        "invalid_value": 0,
        "valid": None
    },
    {
        "validator": IntegerAttribute(minimum=0, maximum=10),
        "invalid_type": 5.0,
        "invalid_value": 42,
        "valid": 5
    },
    {
        "validator": FloatAttribute(minimum=1.0),
        "invalid_type": ["a", "list"],
        "invalid_value": 0.5,
        "valid": 1.5
    },
    {
        "validator": BoolAttribute(),
        "invalid_type": 5,
        "invalid_value": 5,
        "valid": True
    },
    {
        "validator": LiteralAttribute("auto"),
        "invalid_type": 1,
        "invalid_value": "not-auto",
        "valid": "auto"
    },
    {
        "validator": ListAttribute(IntegerAttribute(minimum=1)),
        "invalid_type": 5,
        "invalid_value": [1, 2, 3, -1, 4, 5],
        "valid": [1, 2, 3, 4, 5]
    },
    {
        "validator": IntegerAttribute(minimum=1) | NoneAttribute(),
        "invalid_type": "auto",
        "invalid_value": 0,
        "valid": None
    }
]
