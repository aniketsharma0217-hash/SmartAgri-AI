"""
Automated unit tests for the model explainability framework, global attribution, and instance narratives.
"""

import pytest
import numpy as np
import joblib

from src.explainability import (
    compute_global_feature_importance,
    compute_permutation_importance_metrics,
    ExplanationEngine,
    run_explainability_audit
)
from src.config import BEST_MODEL_PATH, NUMERICAL_FEATURES
from src.predict import predict_crop


def test_global_feature_importance_validity():
    """Verify that global MDI feature importance is computed for all 7 features and sums to ~100%."""
    model = joblib.load(BEST_MODEL_PATH)
    importances = compute_global_feature_importance(model, NUMERICAL_FEATURES)

    assert len(importances) == 7
    for feat in NUMERICAL_FEATURES:
        assert feat in importances
        assert 0.0 <= importances[feat] <= 100.0

    total_pct = sum(importances.values())
    assert 99.0 <= total_pct <= 101.0, f"Expected total ~100%, got {total_pct}"


def test_explanation_engine_instance_narrative():
    """Verify that instance explanation returns structured breakdowns and non-causal disclaimer."""
    engine = ExplanationEngine()
    sample_input = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.88,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.94
    }
    exp = engine.explain_instance(sample_input, "rice")

    assert exp["recommended_crop"] == "rice"
    assert "narrative_summary" in exp
    assert "causal_disclaimer" in exp
    assert "causality" in exp["causal_disclaimer"].lower()
    assert len(exp["feature_breakdown"]) == 7

    for item in exp["feature_breakdown"]:
        assert "feature" in item
        assert "user_value" in item
        assert "crop_median" in item
        assert "delta" in item
        assert "alignment" in item


def test_predict_crop_integration_with_explainability():
    """Verify that predict_crop includes the full explainability report."""
    sample_input = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.88,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.94
    }
    result = predict_crop(sample_input)
    assert result["status"] == "success"
    assert "explainability" in result
    assert "explainability_report" in result
    assert result["explainability_report"] is not None
    assert "causal_disclaimer" in result["explainability_report"]
