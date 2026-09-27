"""
Model Explainability Module for SmartAgri-AI.
Implements both Global Attribution (MDI Feature Importance & Permutation Importance)
and Local Instance Explainability (Agronomic Profile Deviation Analysis)
with strict adherence to non-causal statistical communication standards.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

from src.config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    MODEL_METADATA_PATH,
    REPORTS_DIR,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    FEATURE_BOUNDS,
    RANDOM_STATE
)
from src.data_loader import load_raw_data
from src.utils import get_logger, save_json, load_json

logger = get_logger("Explainability")

EXPLAINABILITY_JSON = REPORTS_DIR / "explainability.json"


def compute_global_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """
    Computes global Mean Decrease in Impurity (MDI) feature importances from Random Forest.

    Args:
        model: Trained Random Forest model.
        feature_names: Ordered list of feature names.

    Returns:
        Dict[str, float]: Sorted feature importance percentages.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"Model {type(model)} does not provide feature_importances_.")

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    return {
        feature_names[i]: round(float(importances[i]) * 100, 2)
        for i in sorted_idx
    }


def compute_permutation_importance_metrics(
    model: Any,
    X_test_scaled: np.ndarray,
    y_test_encoded: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Computes Permutation Feature Importance on held-out test data to validate MDI importances.

    Args:
        model: Trained classifier.
        X_test_scaled: Standardized test feature matrix.
        y_test_encoded: Encoded test targets.
        feature_names: List of feature names.

    Returns:
        Dict[str, Any]: Mean importance and standard deviation per feature.
    """
    logger.info("Computing Permutation Feature Importance on held-out test partition...")
    perm = permutation_importance(
        model,
        X_test_scaled,
        y_test_encoded,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    perm_results = {}
    for i in np.argsort(perm.importances_mean)[::-1]:
        feat = feature_names[i]
        perm_results[feat] = {
            "mean_importance_drop": round(float(perm.importances_mean[i]), 4),
            "std_importance_drop": round(float(perm.importances_std[i]), 4)
        }

    return perm_results


class ExplanationEngine:
    """
    Generates human-readable, non-causal explanations for individual predictions.
    """

    def __init__(self):
        self.model = joblib.load(BEST_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.crop_profiles = load_json(MODELS_DIR / "crop_profiles.json")
        self.global_importance = compute_global_feature_importance(self.model, NUMERICAL_FEATURES)

    def explain_instance(
        self,
        sanitized_input: Dict[str, float],
        recommended_crop: str
    ) -> Dict[str, Any]:
        """
        Generates local explanation comparing user inputs to typical crop profiles.

        Args:
            sanitized_input: Validated input dictionary.
            recommended_crop: Output predicted crop label.

        Returns:
            Dict[str, Any]: Structured explanation report.
        """
        crop_name = recommended_crop.lower()
        ideal_profile = self.crop_profiles.get(crop_name, {})

        influential_features = []
        feature_breakdown = []

        for feature in NUMERICAL_FEATURES:
            user_val = sanitized_input.get(feature, 0.0)
            median_val = ideal_profile.get(feature, user_val)
            bounds = FEATURE_BOUNDS.get(feature, {})
            label = bounds.get("label", feature)
            unit = bounds.get("unit", "")

            diff = user_val - median_val
            pct_diff = (diff / median_val * 100) if median_val != 0 else 0.0

            # Categorize agronomic alignment
            if abs(pct_diff) <= 15.0:
                alignment = "Optimal Match"
                narrative = f"Within typical range for {recommended_crop}."
            elif pct_diff > 15.0:
                alignment = "Above Typical"
                narrative = f"{abs(pct_diff):.1f}% above historical training median."
            else:
                alignment = "Below Typical"
                narrative = f"{abs(pct_diff):.1f}% below historical training median."

            feature_item = {
                "feature": feature,
                "label": label,
                "unit": unit,
                "user_value": round(user_val, 2),
                "crop_median": round(median_val, 2),
                "delta": round(diff, 2),
                "percentage_delta": round(pct_diff, 1),
                "alignment": alignment,
                "global_importance_pct": self.global_importance.get(feature, 0.0),
                "narrative": narrative
            }
            feature_breakdown.append(feature_item)

            # High global importance + optimal alignment indicates strong associative support
            if self.global_importance.get(feature, 0.0) >= 15.0 and alignment == "Optimal Match":
                influential_features.append(label)

        # Build disciplined, non-causal summary sentence
        if influential_features:
            influential_summary = ", ".join(influential_features)
            narrative_text = (
                f"The model identified {influential_summary} as key associative factors for "
                f"recommending '{recommended_crop}'. These inputs closely match the historical "
                f"conditions recorded in the training dataset."
            )
        else:
            narrative_text = (
                f"The multivariate combination of soil nutrients and climatic parameters "
                f"aligned with the decision envelope for '{recommended_crop}' in the trained model."
            )

        causal_disclaimer = (
            "Ethical AI Note: The identified feature importances reflect mathematical associations "
            "and decision boundaries learned by the model. They do not constitute scientific proof "
            "of biological causality. Changing individual parameters will not guarantee crop success."
        )

        return {
            "recommended_crop": recommended_crop,
            "influential_factors": influential_features,
            "narrative_summary": narrative_text,
            "feature_breakdown": feature_breakdown,
            "global_feature_importance": self.global_importance,
            "causal_disclaimer": causal_disclaimer
        }


def run_explainability_audit() -> Dict[str, Any]:
    """
    Executes an audit of global feature attribution and permutation importance,
    saving the benchmark report to reports/explainability.json.

    Returns:
        Dict[str, Any]: Full explainability audit metrics.
    """
    logger.info("Initializing Explainability Audit...")
    from src.evaluate import load_test_partition

    X_test_scaled, y_test_encoded, features, _ = load_test_partition()
    model = joblib.load(BEST_MODEL_PATH)

    mdi_importances = compute_global_feature_importance(model, features)
    perm_metrics = compute_permutation_importance_metrics(model, X_test_scaled, y_test_encoded, features)

    audit_data = {
        "model_architecture": "Random Forest Classifier (100 Trees)",
        "mean_decrease_in_impurity_pct": mdi_importances,
        "permutation_importance_on_test_set": perm_metrics,
        "causal_cautionary_guidelines": [
            "Feature importance shows statistical association within the training dataset.",
            "Do not assert that modifying an input parameter guarantees a specific biological yield.",
            "Always state: 'The model identified these features as influential for this prediction.'"
        ]
    }

    save_json(audit_data, EXPLAINABILITY_JSON)
    logger.info(f"Explainability audit successfully saved to: {EXPLAINABILITY_JSON}")
    return audit_data


if __name__ == "__main__":
    audit = run_explainability_audit()
    print("\n" + "=" * 70)
    print("SMARTAGRI-AI MODEL EXPLAINABILITY BENCHMARK")
    print("=" * 70)
    print("GLOBAL FEATURE IMPORTANCE (MDI - Gini):")
    for feat, score in audit["mean_decrease_in_impurity_pct"].items():
        print(f"  - {feat:<12}: {score:>6.2f}%")
    print("\nPERMUTATION IMPORTANCE DROP ON HELD-OUT TEST DATA:")
    for feat, data in audit["permutation_importance_on_test_set"].items():
        print(f"  - {feat:<12}: Drop = {data['mean_importance_drop']:.4f} (+/- {data['std_importance_drop']:.4f})")
    print("=" * 70)

    # Test an instance explanation
    engine = ExplanationEngine()
    test_input = {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 20.88, "humidity": 82.0, "ph": 6.5, "rainfall": 202.94}
    instance_exp = engine.explain_instance(test_input, "rice")
    print("\nINSTANCE EXPLANATION TEST (Rice Query):")
    print(f"Narrative: {instance_exp['narrative_summary']}")
    print(f"Disclaimer: {instance_exp['causal_disclaimer']}")
    print("=" * 70)
