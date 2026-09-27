"""
Centralized Configuration for SmartAgri-AI
Defines paths, feature definitions, agronomic validation boundaries, and ML parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

# Base Directory Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = Path(os.getenv("SMARTAGRI_DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = Path(os.getenv("SMARTAGRI_MODELS_DIR", str(PROJECT_ROOT / "models")))
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
APP_DIR: Path = PROJECT_ROOT / "app"

# Cache & Temporary Directories (Sandboxing & Permissions Friendly)
CACHE_DIR: Path = PROJECT_ROOT / ".cache"
MATPLOTLIB_CACHE_DIR: Path = CACHE_DIR / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

# Primary Raw Dataset Path
DATASET_FILENAME: str = os.getenv("SMARTAGRI_DATASET_FILENAME", "Crop_recommendation.csv")
RAW_DATASET_PATH: Path = Path(os.getenv("SMARTAGRI_RAW_DATASET_PATH", str(RAW_DATA_DIR / DATASET_FILENAME)))
PROCESSED_DATASET_PATH: Path = Path(os.getenv("SMARTAGRI_PROCESSED_DATASET_PATH", str(PROCESSED_DATA_DIR / "crop_data_processed.csv")))
TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"
PREPROCESSOR_META_PATH: Path = PROCESSED_DATA_DIR / "preprocessor_meta.json"

# Model Artifact Paths
BEST_MODEL_PATH: Path = Path(os.getenv("SMARTAGRI_MODEL_PATH", str(MODELS_DIR / "crop_recommender_model.joblib")))
SCALER_PATH: Path = MODELS_DIR / "scaler.joblib"
LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.joblib"
MODEL_METRICS_PATH: Path = MODELS_DIR / "model_metrics.json"
MODEL_METADATA_PATH: Path = MODELS_DIR / "metadata.json"
CANDIDATE_MODELS_DIR: Path = MODELS_DIR / "candidates"
MODEL_COMPARISON_JSON: Path = REPORTS_DIR / "model_comparison.json"

# Feature Definitions
NUMERICAL_FEATURES: List[str] = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

TARGET_COLUMN: str = "label"

# Realistic Agronomic Domain Ranges for Input Validation
# Based on agricultural reference benchmarks for crop soil & climate requirements
FEATURE_BOUNDS: Dict[str, Dict[str, Any]] = {
    "N": {
        "min": 0.0,
        "max": 140.0,
        "unit": "kg/ha",
        "label": "Nitrogen (N)",
        "description": "Ratio of Nitrogen content in soil",
        "default": 50.0
    },
    "P": {
        "min": 5.0,
        "max": 145.0,
        "unit": "kg/ha",
        "label": "Phosphorus (P)",
        "description": "Ratio of Phosphorus content in soil",
        "default": 50.0
    },
    "K": {
        "min": 5.0,
        "max": 205.0,
        "unit": "kg/ha",
        "label": "Potassium (K)",
        "description": "Ratio of Potassium content in soil",
        "default": 50.0
    },
    "temperature": {
        "min": 8.0,
        "max": 45.0,
        "unit": "°C",
        "label": "Temperature",
        "description": "Ambient temperature in Celsius",
        "default": 25.0
    },
    "humidity": {
        "min": 10.0,
        "max": 100.0,
        "unit": "%",
        "label": "Humidity",
        "description": "Relative atmospheric humidity percentage",
        "default": 70.0
    },
    "ph": {
        "min": 3.5,
        "max": 10.0,
        "unit": "pH",
        "label": "Soil pH",
        "description": "Acidity / alkalinity value of the soil (0-14 scale)",
        "default": 6.5
    },
    "rainfall": {
        "min": 20.0,
        "max": 300.0,
        "unit": "mm",
        "label": "Rainfall",
        "description": "Total seasonal / annual precipitation in millimeters",
        "default": 100.0
    }
}

# Machine Learning Parameters
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# Candidate Models to Evaluate
CANDIDATE_MODELS: List[str] = [
    "Random Forest",
    "Decision Tree",
    "Logistic Regression",
    "Support Vector Machine",
    "K-Nearest Neighbors"
]

# Application Configuration
APP_TITLE: str = "SmartAgri AI — Crop Recommendation System"
APP_TAGLINE: str = "AI-Powered Crop Recommendation & Agricultural Decision Support"
