"""
Automated unit tests for production prediction service, input validation, and inference.
"""

import pytest
from src.predict import (
    CropPredictor,
    predict_crop,
    validate_input_data,
    InputValidationError
)


@pytest.fixture
def valid_rice_input():
    return {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.88,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.94
    }


@pytest.fixture
def valid_cotton_input():
    return {
        "N": 120.0,
        "P": 40.0,
        "K": 20.0,
        "temperature": 24.0,
        "humidity": 80.0,
        "ph": 6.8,
        "rainfall": 80.0
    }


def test_validate_input_data_success(valid_rice_input):
    """Verify that a properly formatted agricultural dictionary passes validation."""
    is_valid, sanitized, errors = validate_input_data(valid_rice_input)
    assert is_valid is True
    assert len(errors) == 0
    assert len(sanitized) == 7
    assert sanitized["N"] == 90.0


def test_validate_input_data_missing_field(valid_rice_input):
    """Verify that omitting a mandatory feature triggers a validation error."""
    invalid_input = valid_rice_input.copy()
    del invalid_input["rainfall"]
    is_valid, _, errors = validate_input_data(invalid_input)
    assert is_valid is False
    assert "rainfall" in errors
    assert "Missing required parameter" in errors["rainfall"]


def test_validate_input_data_non_numeric(valid_rice_input):
    """Verify that non-numeric strings are rejected."""
    invalid_input = valid_rice_input.copy()
    invalid_input["ph"] = "acidic_soil"
    is_valid, _, errors = validate_input_data(invalid_input)
    assert is_valid is False
    assert "ph" in errors
    assert "Invalid numeric format" in errors["ph"]


def test_validate_input_data_out_of_bounds(valid_rice_input):
    """Verify that physiologically impossible values are caught and flagged."""
    invalid_input = valid_rice_input.copy()
    invalid_input["N"] = -25.0       # Negative nitrogen impossible
    invalid_input["ph"] = 14.5       # Beyond pH scale
    invalid_input["humidity"] = 150  # Over 100% humidity impossible

    is_valid, _, errors = validate_input_data(invalid_input)
    assert is_valid is False
    assert "N" in errors
    assert "ph" in errors
    assert "humidity" in errors


def test_predict_crop_success(valid_rice_input):
    """Verify end-to-end prediction returns recommended crop and calibrated probabilities."""
    result = predict_crop(valid_rice_input)
    assert result["status"] == "success"
    assert result["recommended_crop"] == "rice"
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert len(result["top_alternatives"]) == 3
    assert result["top_alternatives"][0]["crop"] == "rice"
    assert "explainability" in result
    assert len(result["explainability"]) == 7


def test_predict_crop_cotton(valid_cotton_input):
    """Verify end-to-end prediction on high-nitrogen crop (cotton)."""
    result = predict_crop(valid_cotton_input)
    assert result["status"] == "success"
    assert result["recommended_crop"] == "cotton"
    assert result["confidence_score"] > 0.50


def test_predict_crop_error_handling():
    """Verify that passing invalid data returns a safe error dictionary without crashing."""
    bad_input = {"N": -50.0, "P": "abc"}
    result = predict_crop(bad_input)
    assert result["status"] == "error"
    assert result["error_type"] == "ValidationError"
    assert "errors" in result


def test_crop_predictor_singleton():
    """Verify that CropPredictor singleton returns the same instance without re-instantiation."""
    p1 = CropPredictor.get_instance()
    p2 = CropPredictor.get_instance()
    assert p1 is p2


def test_prediction_output_format_exact_schema(valid_rice_input):
    """Verify exact schema and types of prediction output dictionary."""
    result = predict_crop(valid_rice_input)
    assert isinstance(result, dict)

    required_keys = {
        "status",
        "recommended_crop",
        "confidence_score",
        "confidence_percentage",
        "top_alternatives",
        "input_summary",
        "explainability",
        "explainability_report"
    }
    assert required_keys.issubset(result.keys()), f"Missing keys in output: {required_keys - result.keys()}"

    assert result["status"] == "success"
    assert isinstance(result["recommended_crop"], str)
    assert isinstance(result["confidence_score"], float)
    assert isinstance(result["confidence_percentage"], str)
    assert result["confidence_percentage"].endswith("%")
    assert isinstance(result["top_alternatives"], list)
    assert isinstance(result["input_summary"], dict)
    assert isinstance(result["explainability"], list)
    assert isinstance(result["explainability_report"], dict)


def test_top_alternatives_ordering_and_format(valid_rice_input):
    """Verify alternatives are ranked descending with valid probabilities and percentages."""
    result = predict_crop(valid_rice_input)
    alternatives = result["top_alternatives"]
    assert len(alternatives) == 3

    probs = [alt["probability"] for alt in alternatives]
    # Check descending order
    assert probs == sorted(probs, reverse=True)

    for alt in alternatives:
        assert "crop" in alt and isinstance(alt["crop"], str)
        assert "probability" in alt and 0.0 <= alt["probability"] <= 1.0
        assert "confidence_percentage" in alt and alt["confidence_percentage"].endswith("%")


def test_explainability_comparison_fields(valid_rice_input):
    """Verify feature comparison fields within the explainability report."""
    result = predict_crop(valid_rice_input)
    comparison_list = result["explainability"]
    assert len(comparison_list) == 7

    expected_fields = {"feature", "label", "unit", "user_value", "crop_median", "difference", "status"}
    for item in comparison_list:
        assert expected_fields.issubset(item.keys())
        assert item["status"] in {"Optimal", "Higher than typical", "Lower than typical"}
