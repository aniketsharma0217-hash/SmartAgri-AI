"""
Automated unit tests for machine learning model training, candidate architectures, and metric integrity.
"""

import pytest
import numpy as np
import joblib
from pathlib import Path

from src.train import get_candidate_models, train_and_benchmark_models
from src.config import (
    CANDIDATE_MODELS,
    CANDIDATE_MODELS_DIR,
    MODEL_COMPARISON_JSON,
    RANDOM_STATE
)
from src.utils import load_json


def test_candidate_models_structure():
    """Verify that all 5 candidate model families are properly instantiated."""
    models = get_candidate_models(random_state=RANDOM_STATE)
    assert len(models) == 5
    for expected_name in CANDIDATE_MODELS:
        assert expected_name in models, f"Missing candidate model: {expected_name}"


def test_candidate_model_serialization():
    """Verify that all candidate models have been serialized to joblib artifacts."""
    assert CANDIDATE_MODELS_DIR.exists()
    expected_slugs = [
        "random_forest.joblib",
        "decision_tree.joblib",
        "logistic_regression.joblib",
        "support_vector_machine.joblib",
        "k_nearest_neighbors.joblib"
    ]
    for slug in expected_slugs:
        model_file = CANDIDATE_MODELS_DIR / slug
        assert model_file.exists(), f"Model artifact missing: {slug}"
        loaded = joblib.load(model_file)
        assert hasattr(loaded, "predict"), f"Loaded artifact {slug} has no predict method"


def test_model_comparison_metrics_validity():
    """Verify that the model comparison report exists and contains genuine, bounded metrics."""
    assert MODEL_COMPARISON_JSON.exists(), f"Comparison file missing: {MODEL_COMPARISON_JSON}"
    results = load_json(MODEL_COMPARISON_JSON)

    assert len(results) == 5
    for model_name, metrics in results.items():
        assert "test_accuracy" in metrics
        assert "f1_macro" in metrics
        assert "cv_accuracy_mean" in metrics
        assert "fit_time_seconds" in metrics

        # Validate mathematical metric boundaries [0.0, 1.0]
        assert 0.0 <= metrics["test_accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0
        assert 0.0 <= metrics["cv_accuracy_mean"] <= 1.0

        # High benchmark check: all models should exceed 90% accuracy on this standard dataset
        assert metrics["test_accuracy"] > 0.90, f"{model_name} accuracy below expected benchmark"


def test_selected_production_model_exists():
    """Verify that the selected production model exists and can be loaded."""
    from src.config import BEST_MODEL_PATH
    assert BEST_MODEL_PATH.exists(), f"Production model missing at {BEST_MODEL_PATH}"
    model = joblib.load(BEST_MODEL_PATH)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_production_model_metadata():
    """Verify that production metadata accurately reflects selection criteria and metrics."""
    from src.config import MODEL_METADATA_PATH
    assert MODEL_METADATA_PATH.exists(), "Metadata file missing"
    meta = load_json(MODEL_METADATA_PATH)
    assert meta["model_name"] == "Random Forest"
    assert "selection_criteria" in meta
    assert "performance_summary" in meta
    assert meta["performance_summary"]["test_accuracy"] > 0.99
    assert len(meta["target_classes"]) == 22

