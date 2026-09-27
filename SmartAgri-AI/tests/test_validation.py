"""
Comprehensive Automated Unit Tests for Input Validation and Boundary Enforcement.
Verifies rejection of non-numeric, out-of-bounds, boolean, NaN, and malformed queries.
"""

import pytest
import math
from src.predict import validate_input_data, predict_crop


@pytest.fixture
def valid_input():
    return {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.88,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.94
    }


def test_validation_rejects_non_dict():
    """Verify that passing non-dictionary types (list, int, str, None) is rejected."""
    for invalid_payload in [[1, 2, 3], "N=50", 12345, None]:
        is_valid, _, errors = validate_input_data(invalid_payload)
        assert is_valid is False
        assert "general" in errors


def test_validation_rejects_empty_dict():
    """Verify that an empty dictionary is rejected."""
    is_valid, _, errors = validate_input_data({})
    assert is_valid is False
    assert "general" in errors


def test_validation_rejects_boolean_values(valid_input):
    """
    CRITICAL PYTHON SECURITY TEST:
    In Python, isinstance(True, int) evaluates to True and float(True) == 1.0.
    Verify that boolean inputs are explicitly rejected.
    """
    bad_input = valid_input.copy()
    bad_input["temperature"] = True
    is_valid, _, errors = validate_input_data(bad_input)
    assert is_valid is False
    assert "temperature" in errors
    assert "boolean" in errors["temperature"].lower()


def test_validation_rejects_nan_and_infinity(valid_input):
    """Verify that NaN and +/- Infinity are rejected."""
    for bad_val in [float("nan"), float("inf"), float("-inf")]:
        bad_input = valid_input.copy()
        bad_input["ph"] = bad_val
        is_valid, _, errors = validate_input_data(bad_input)
        assert is_valid is False
        assert "ph" in errors
        assert "finite" in errors["ph"].lower()


def test_validation_rejects_negative_nutrients(valid_input):
    """Verify that physically negative nutrient values are rejected."""
    for nutrient in ["N", "P", "K", "rainfall", "humidity"]:
        bad_input = valid_input.copy()
        bad_input[nutrient] = -1.0
        is_valid, _, errors = validate_input_data(bad_input)
        assert is_valid is False
        assert nutrient in errors
        assert "outside realistic agronomic range" in errors[nutrient]


def test_validation_rejects_extreme_temperatures(valid_input):
    """Verify that extreme temperatures (<8°C or >45°C) are rejected."""
    bad_low = valid_input.copy()
    bad_low["temperature"] = -5.0
    is_valid, _, errors = validate_input_data(bad_low)
    assert is_valid is False
    assert "temperature" in errors

    bad_high = valid_input.copy()
    bad_high["temperature"] = 65.0
    is_valid, _, errors = validate_input_data(bad_high)
    assert is_valid is False
    assert "temperature" in errors


def test_validation_rejects_impossible_ph(valid_input):
    """Verify that impossible soil pH values (<3.5 or >10.0) are rejected."""
    bad_acidic = valid_input.copy()
    bad_acidic["ph"] = 1.0
    is_valid, _, errors = validate_input_data(bad_acidic)
    assert is_valid is False
    assert "ph" in errors

    bad_alkaline = valid_input.copy()
    bad_alkaline["ph"] = 14.0
    is_valid, _, errors = validate_input_data(bad_alkaline)
    assert is_valid is False
    assert "ph" in errors


def test_validation_rejects_impossible_humidity(valid_input):
    """Verify that humidity >100% or <10% is rejected."""
    bad_high = valid_input.copy()
    bad_high["humidity"] = 120.0
    is_valid, _, errors = validate_input_data(bad_high)
    assert is_valid is False
    assert "humidity" in errors


def test_predict_crop_error_structure():
    """Verify that predict_crop returns structured error with all flagged fields without raising unhandled exceptions."""
    bad_payload = {
        "N": -99,
        "P": "abc",
        "K": 50,
        "temperature": 999,
        "humidity": 50,
        "ph": 6.5,
        "rainfall": 100
    }
    response = predict_crop(bad_payload)
    assert response["status"] == "error"
    assert response["error_type"] == "ValidationError"
    assert "N" in response["errors"]
    assert "P" in response["errors"]
    assert "temperature" in response["errors"]
