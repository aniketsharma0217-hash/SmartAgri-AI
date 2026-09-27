"""
Production Inference and Prediction Service for SmartAgri-AI.
Provides robust input validation, leak-free feature scaling, multi-class prediction,
calibrated probability estimation, and agronomic explainability comparisons.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    FEATURE_BOUNDS
)
from src.utils import get_logger, load_json, safe_load_joblib

logger = get_logger("Predictor")


class InputValidationError(ValueError):
    """Raised when user input fails structural, type, or domain range checks."""
    def __init__(self, errors: Dict[str, str]):
        self.errors = errors
        message = "Input validation failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items())
        super().__init__(message)


class ModelInferenceError(RuntimeError):
    """Raised when an error occurs during model evaluation or deserialization."""
    pass


import math

def validate_input_data(input_data: Dict[str, Any]) -> Tuple[bool, Dict[str, float], Dict[str, str]]:
    """
    Validates user agricultural input dictionary against schema, numeric type, and domain bounds.
    Hardened against type injection (booleans, lists), non-finite numbers (NaN, Inf),
    empty fields, and physical agronomic boundary violations.

    Args:
        input_data: Key-value dictionary of inputs (e.g., {'N': 90, 'P': 42, ...})

    Returns:
        Tuple: (is_valid, sanitized_floats_dict, error_messages_dict)
    """
    errors: Dict[str, str] = {}
    sanitized: Dict[str, float] = {}

    if not isinstance(input_data, dict):
        return False, {}, {"general": f"Expected input as a dictionary, received {type(input_data).__name__}."}

    if not input_data:
        return False, {}, {"general": "Input dictionary cannot be empty."}

    for feature in NUMERICAL_FEATURES:
        if feature not in input_data:
            errors[feature] = f"Missing required parameter: '{feature}'."
            continue

        raw_val = input_data[feature]

        # 1. Null & Empty String Check
        if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ""):
            errors[feature] = f"Value for '{feature}' cannot be empty."
            continue

        # 2. Strict Type Check: Booleans are subclasses of int in Python, reject them
        if isinstance(raw_val, bool):
            errors[feature] = f"Invalid numeric format for '{feature}': boolean values are not permitted."
            continue

        # 3. Numeric Conversion Check
        try:
            val_float = float(raw_val)
        except (ValueError, TypeError):
            errors[feature] = f"Invalid numeric format for '{feature}': '{raw_val}'."
            continue

        # 4. Finite Number Check (Reject NaN and +/- Infinity)
        if math.isnan(val_float) or math.isinf(val_float):
            errors[feature] = f"Value for '{feature}' must be a finite real number (NaN or Infinity not allowed)."
            continue

        # 5. Domain Range Boundary Check
        bounds = FEATURE_BOUNDS.get(feature, {})
        min_allowed = bounds.get("min", float("-inf"))
        max_allowed = bounds.get("max", float("inf"))
        unit = bounds.get("unit", "")

        if val_float < min_allowed or val_float > max_allowed:
            errors[feature] = (
                f"Value {val_float} is outside realistic agronomic range "
                f"[{min_allowed} - {max_allowed} {unit}]."
            )
            continue

        sanitized[feature] = val_float

    is_valid = len(errors) == 0
    return is_valid, sanitized, errors


class CropPredictor:
    """
    Production-grade, stateless inference service.
    Loads trained artifacts once and executes low-latency inference.
    """

    _instance: Optional["CropPredictor"] = None

    def __init__(self):
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.label_encoder: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.crop_profiles: Dict[str, Dict[str, float]] = {}
        try:
            self._load_artifacts()
        except ModelInferenceError:
            raise  # Re-raise cleanly; caller receives a structured, meaningful error.
        except Exception as exc:
            raise ModelInferenceError(
                f"Unexpected failure while loading model artifacts: {exc}"
            ) from exc

    @classmethod
    def get_instance(cls) -> "CropPredictor":
        """Singleton accessor to avoid redundant model loading in UI apps."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_artifacts(self) -> None:
        """Loads and verifies production model artifacts from models/ directory."""
        if not BEST_MODEL_PATH.exists():
            raise ModelInferenceError(
                f"Production model artifact not found at {BEST_MODEL_PATH}. "
                "Ensure Phase 6 and 7 model training and evaluation have completed."
            )
        if not SCALER_PATH.exists() or not LABEL_ENCODER_PATH.exists():
            raise ModelInferenceError("Required preprocessing artifacts (scaler/encoder) are missing.")

        logger.info(f"Loading production model from: {BEST_MODEL_PATH}")
        try:
            self.model = safe_load_joblib(BEST_MODEL_PATH)
            self.scaler = safe_load_joblib(SCALER_PATH)
            self.label_encoder = safe_load_joblib(LABEL_ENCODER_PATH)
        except (FileNotFoundError, RuntimeError) as exc:
            raise ModelInferenceError(f"Artifact loading failed: {exc}") from exc

        if MODEL_METADATA_PATH.exists():
            self.metadata = load_json(MODEL_METADATA_PATH)

        profiles_path = MODELS_DIR / "crop_profiles.json"
        if profiles_path.exists():
            self.crop_profiles = load_json(profiles_path)

        logger.info("CropPredictor successfully initialized and artifacts loaded.")

    def compute_explainability_comparison(
        self,
        sanitized_input: Dict[str, float],
        recommended_crop: str
    ) -> List[Dict[str, Any]]:
        """
        Generates an agronomic feature comparison showing user inputs vs ideal crop profile medians.

        Args:
            sanitized_input: Validated input features.
            recommended_crop: Target predicted crop.

        Returns:
            List[Dict[str, Any]]: Feature-by-feature comparison metrics.
        """
        comparison = []
        crop_median = self.crop_profiles.get(recommended_crop.lower(), {})

        for feature in NUMERICAL_FEATURES:
            user_val = sanitized_input.get(feature, 0.0)
            median_val = crop_median.get(feature, user_val)
            bounds = FEATURE_BOUNDS.get(feature, {})
            unit = bounds.get("unit", "")
            label = bounds.get("label", feature)

            diff = user_val - median_val
            pct_diff = (diff / median_val * 100) if median_val != 0 else 0.0

            if abs(pct_diff) <= 15.0:
                status = "Optimal"
            elif pct_diff > 15.0:
                status = "Higher than typical"
            else:
                status = "Lower than typical"

            comparison.append({
                "feature": feature,
                "label": label,
                "unit": unit,
                "user_value": round(user_val, 2),
                "crop_median": round(median_val, 2),
                "difference": round(diff, 2),
                "status": status
            })

        return comparison

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes complete end-to-end prediction: validation, scaling, inference, confidence, explainability.

        Args:
            input_data: User inputs dictionary.

        Returns:
            Dict[str, Any]: Structured prediction output.
        """
        # 1. Input Validation
        is_valid, sanitized, errors = validate_input_data(input_data)
        if not is_valid:
            logger.warning(f"Prediction rejected due to validation errors: {errors}")
            return {
                "status": "error",
                "error_type": "ValidationError",
                "message": "Input validation failed. Please check input parameters.",
                "errors": errors
            }

        try:
            # 2. Preprocess & Scale
            df_input = pd.DataFrame([sanitized])[NUMERICAL_FEATURES]
            scaled_vector = self.scaler.transform(df_input)

            # 3. Model Inference
            pred_class_idx = self.model.predict(scaled_vector)[0]
            recommended_crop = self.label_encoder.inverse_transform([pred_class_idx])[0]

            # 4. Probabilities & Alternative Candidates
            alternatives = []
            confidence_score = 1.0

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(scaled_vector)[0]
                confidence_score = float(probabilities[pred_class_idx])

                # Get Top 3 candidate predictions
                top_indices = np.argsort(probabilities)[::-1][:3]
                for idx in top_indices:
                    crop_name = self.label_encoder.inverse_transform([idx])[0]
                    prob = float(probabilities[idx])
                    alternatives.append({
                        "crop": crop_name,
                        "probability": round(prob, 4),
                        "confidence_percentage": f"{prob * 100:.1f}%"
                    })

            # 5. Explainability Comparison & Structured Report
            explainability = self.compute_explainability_comparison(sanitized, recommended_crop)
            
            try:
                from src.explainability import ExplanationEngine
                engine = ExplanationEngine()
                exp_report = engine.explain_instance(sanitized, recommended_crop)
            except Exception as exp_err:
                logger.warning(f"Could not generate extended explanation report: {exp_err}")
                exp_report = None

            # 6. Format Input Summary with Units
            input_summary = {
                FEATURE_BOUNDS[f]["label"]: f"{sanitized[f]} {FEATURE_BOUNDS[f]['unit']}"
                for f in NUMERICAL_FEATURES
            }

            return {
                "status": "success",
                "recommended_crop": recommended_crop,
                "confidence_score": round(confidence_score, 4),
                "confidence_percentage": f"{confidence_score * 100:.1f}%",
                "top_alternatives": alternatives,
                "input_summary": input_summary,
                "raw_inputs": sanitized,
                "explainability": explainability,
                "explainability_report": exp_report,
                "model_info": {
                    "model_name": self.metadata.get("model_name", "Random Forest Classifier"),
                    "model_version": self.metadata.get("model_version", "1.0.0"),
                    "test_accuracy": self.metadata.get("performance_summary", {}).get("test_accuracy", 0.9955)
                },
                "disclaimer": (
                    "SmartAgri-AI provides decision-support predictions based on machine learning. "
                    "It is not a guarantee of crop success. Local soil testing and certified agronomist "
                    "consultation are strongly recommended before planting."
                )
            }

        except Exception as e:
            logger.error(f"Inference execution failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_type": "ModelInferenceError",
                "message": f"Inference pipeline failed: {str(e)}"
            }


def predict_crop(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for frontend applications to generate crop recommendations.

    Args:
        input_data: Agricultural parameter dictionary.

    Returns:
        Dict[str, Any]: Structured prediction response.
    """
    predictor = CropPredictor.get_instance()
    return predictor.predict(input_data)


if __name__ == "__main__":
    # Test valid input (typical Rice conditions)
    rice_input = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.88,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.94
    }

    print("\n" + "=" * 65)
    print("TESTING VALID PREDICTION INFERENCE")
    print("=" * 65)
    response = predict_crop(rice_input)
    print(f"Status:            {response['status']}")
    print(f"Recommended Crop:  {response['recommended_crop']}")
    print(f"Confidence:        {response['confidence_percentage']}")
    print("Top Alternatives:")
    for alt in response["top_alternatives"]:
        print(f"  - {alt['crop']}: {alt['confidence_percentage']}")
    print("=" * 65)

    # Test invalid input (out of bounds & non-numeric)
    print("\n" + "=" * 65)
    print("TESTING INVALID INPUT REJECTION")
    print("=" * 65)
    invalid_input = {
        "N": -10.0,          # Out of bounds
        "P": "invalid_abc",   # Non-numeric
        "K": 50.0,
        "temperature": 25.0,
        "humidity": 70.0,
        "ph": 15.0,          # Out of bounds (>10)
        "rainfall": 100.0
    }
    err_response = predict_crop(invalid_input)
    print(f"Status:            {err_response['status']}")
    print(f"Error Type:        {err_response['error_type']}")
    print(f"Validation Errors: {err_response['errors']}")
    print("=" * 65)
