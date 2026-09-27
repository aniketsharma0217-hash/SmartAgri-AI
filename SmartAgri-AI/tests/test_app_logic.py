"""
Automated unit tests for application business logic, presets, and model metadata consistency.
"""

import pytest
from src.predict import predict_crop, validate_input_data
from src.config import (
    MODEL_METADATA_PATH,
    MODEL_COMPARISON_JSON,
    FEATURE_BOUNDS,
    NUMERICAL_FEATURES
)
from src.utils import load_json


# Presets defined in the UI
APPLICATION_PRESETS = {
    "Rice (Monsoon / High Rain)": {
        "input": {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 20.88, "humidity": 82.0, "ph": 6.50, "rainfall": 202.94},
        "expected_crop": "rice"
    },
    "Cotton (Warm / High Nitrogen)": {
        "input": {"N": 120.0, "P": 40.0, "K": 20.0, "temperature": 24.0, "humidity": 80.0, "ph": 6.8, "rainfall": 80.0},
        "expected_crop": "cotton"
    },
    "Coffee (Sub-tropical High Potassium)": {
        "input": {"N": 100.0, "P": 30.0, "K": 30.0, "temperature": 25.5, "humidity": 58.0, "ph": 6.8, "rainfall": 160.0},
        "expected_crop": "coffee"
    },
    "Apple (Cool / High Potash)": {
        "input": {"N": 20.0, "P": 135.0, "K": 200.0, "temperature": 22.0, "humidity": 92.0, "ph": 5.9, "rainfall": 110.0},
        "expected_crop": "apple"
    },
    "Chickpea (Low Nitrogen / Dry Pulse)": {
        "input": {"N": 40.0, "P": 68.0, "K": 80.0, "temperature": 18.9, "humidity": 17.0, "ph": 7.4, "rainfall": 80.0},
        "expected_crop": "chickpea"
    }
}


@pytest.mark.parametrize("preset_name,preset_data", APPLICATION_PRESETS.items())
def test_application_presets_validity_and_recommendation(preset_name, preset_data):
    """Verify all application presets pass validation and yield expected recommendations."""
    raw_input = preset_data["input"]
    expected = preset_data["expected_crop"]

    # 1. Validation check
    is_valid, sanitized, errors = validate_input_data(raw_input)
    assert is_valid is True, f"Preset '{preset_name}' failed validation: {errors}"
    assert len(errors) == 0

    # 2. Prediction check
    result = predict_crop(raw_input)
    assert result["status"] == "success"
    assert result["recommended_crop"].lower() == expected.lower(), (
        f"Preset '{preset_name}' yielded '{result['recommended_crop']}', expected '{expected}'"
    )
    assert result["confidence_score"] > 0.50


def test_feature_bounds_coverage():
    """Verify that FEATURE_BOUNDS has complete coverage for all 7 NUMERICAL_FEATURES."""
    for feat in NUMERICAL_FEATURES:
        assert feat in FEATURE_BOUNDS
        spec = FEATURE_BOUNDS[feat]
        assert "min" in spec and "max" in spec and "unit" in spec
        assert spec["min"] < spec["max"]
        assert isinstance(spec["unit"], str) and len(spec["unit"]) > 0


def test_metadata_consistency_with_benchmarks():
    """Verify consistency between models/metadata.json and reports/model_comparison.json."""
    assert MODEL_METADATA_PATH.exists()
    assert MODEL_COMPARISON_JSON.exists()

    meta = load_json(MODEL_METADATA_PATH)
    benchmarks = load_json(MODEL_COMPARISON_JSON)

    rf_benchmark = benchmarks.get("Random Forest")
    assert rf_benchmark is not None

    # Check accuracy and F1 match exactly
    meta_acc = meta["performance_summary"]["test_accuracy"]
    bench_acc = rf_benchmark["test_accuracy"]
    assert abs(meta_acc - bench_acc) < 1e-6

    meta_f1 = meta["performance_summary"]["macro_f1"]
    bench_f1 = rf_benchmark["f1_macro"]
    assert abs(meta_f1 - bench_f1) < 1e-6


def test_boundary_inputs_min_and_max():
    """Verify that predictions with extreme boundary values execute safely."""
    # Construct minimum bounded values
    min_input = {feat: FEATURE_BOUNDS[feat]["min"] for feat in NUMERICAL_FEATURES}
    is_valid, _, errors = validate_input_data(min_input)
    assert is_valid is True, f"Min boundary failed validation: {errors}"
    res_min = predict_crop(min_input)
    assert res_min["status"] == "success"

    # Construct maximum bounded values
    max_input = {feat: FEATURE_BOUNDS[feat]["max"] for feat in NUMERICAL_FEATURES}
    is_valid, _, errors = validate_input_data(max_input)
    assert is_valid is True, f"Max boundary failed validation: {errors}"
    res_max = predict_crop(max_input)
    assert res_max["status"] == "success"
