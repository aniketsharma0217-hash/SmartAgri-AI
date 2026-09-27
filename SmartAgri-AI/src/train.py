"""
Machine Learning Model Training and Multi-Algorithm Benchmarking Module.
Trains, cross-validates, and evaluates 5 classification models on the verified dataset
using strict leak-prevention protocols and reproducible seeds.
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    CANDIDATE_MODELS_DIR,
    MODEL_COMPARISON_JSON,
    NUMERICAL_FEATURES,
    TARGET_COLUMN
)
from src.data_loader import load_raw_data
from src.preprocessing import PreprocessingPipeline
from src.utils import get_logger, ensure_directory, save_json

logger = get_logger("Train")


def get_candidate_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Initializes candidate classification models with reproducible configurations.

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Dict[str, Any]: Dictionary mapping model names to model instances.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            random_state=random_state,
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1
        )
    }


def train_and_benchmark_models() -> Dict[str, Any]:
    """
    Executes the training and benchmarking pipeline across all candidate models:
    1. Loads dataset and applies leak-free preprocessing pipeline.
    2. Performs 5-fold stratified cross-validation on training partition.
    3. Fits each model on full training partition.
    4. Evaluates predictions on the held-out test partition.
    5. Serializes trained candidate models and records factual metrics.

    Returns:
        Dict[str, Any]: Benchmark results and performance metrics across all models.
    """
    ensure_directory(CANDIDATE_MODELS_DIR)
    logger.info("Initializing multi-model training pipeline...")

    # 1. Ingest and Preprocess Data
    raw_df = load_raw_data()
    pipeline = PreprocessingPipeline(random_state=RANDOM_STATE, test_size=0.20)
    clean_df = pipeline.clean_data(raw_df)
    X_train, X_test, y_train, y_test = pipeline.split_data(clean_df)
    X_train_scaled, X_test_scaled, y_train_enc, y_test_enc = pipeline.fit_transform(
        X_train, y_train, X_test, y_test
    )
    pipeline.save_pipeline()

    candidate_models = get_candidate_models(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results: Dict[str, Any] = {}

    for model_name, model in candidate_models.items():
        logger.info(f"--- Training & Benchmarking: {model_name} ---")

        # 2. 5-Fold Stratified Cross-Validation on Training Data Only
        cv_start = time.time()
        cv_scores = cross_val_score(
            model,
            X_train_scaled,
            y_train_enc,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )
        cv_time = time.time() - cv_start

        # 3. Fit on Full Training Partition
        fit_start = time.time()
        model.fit(X_train_scaled, y_train_enc)
        fit_time = time.time() - fit_start

        # 4. Predict on Unseen Held-Out Test Data
        pred_start = time.time()
        y_pred = model.predict(X_test_scaled)
        pred_time = time.time() - pred_start

        # 5. Calculate Metrics on Held-Out Test Data
        acc = float(accuracy_score(y_test_enc, y_pred))
        prec_macro = float(precision_score(y_test_enc, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_test_enc, y_pred, average="weighted", zero_division=0))
        rec_macro = float(recall_score(y_test_enc, y_pred, average="macro", zero_division=0))
        rec_weighted = float(recall_score(y_test_enc, y_pred, average="weighted", zero_division=0))
        f1_mac = float(f1_score(y_test_enc, y_pred, average="macro", zero_division=0))
        f1_wt = float(f1_score(y_test_enc, y_pred, average="weighted", zero_division=0))

        # 6. Serialize Candidate Model
        slug = model_name.lower().replace(" ", "_").replace("-", "_")
        model_save_path = CANDIDATE_MODELS_DIR / f"{slug}.joblib"
        joblib.dump(model, model_save_path)

        results[model_name] = {
            "model_name": model_name,
            "artifact_path": str(model_save_path),
            "hyperparameters": {k: str(v) for k, v in model.get_params().items() if k in [
                "n_estimators", "criterion", "max_depth", "C", "kernel", "solver", "n_neighbors", "weights"
            ]},
            "cv_accuracy_mean": float(np.round(cv_scores.mean(), 4)),
            "cv_accuracy_std": float(np.round(cv_scores.std(), 4)),
            "test_accuracy": float(np.round(acc, 4)),
            "precision_macro": float(np.round(prec_macro, 4)),
            "precision_weighted": float(np.round(prec_weighted, 4)),
            "recall_macro": float(np.round(rec_macro, 4)),
            "recall_weighted": float(np.round(rec_weighted, 4)),
            "f1_macro": float(np.round(f1_mac, 4)),
            "f1_weighted": float(np.round(f1_wt, 4)),
            "fit_time_seconds": float(np.round(fit_time, 4)),
            "inference_time_seconds": float(np.round(pred_time, 4))
        }

        logger.info(
            f"{model_name} -> Test Acc: {acc:.4f} | F1-Macro: {f1_mac:.4f} | "
            f"CV Mean Acc: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )

    # 7. Persist Evaluation Comparison Report
    save_json(results, MODEL_COMPARISON_JSON)
    logger.info(f"Model comparison saved successfully to: {MODEL_COMPARISON_JSON}")

    return results


if __name__ == "__main__":
    benchmark_results = train_and_benchmark_models()
    print("\n" + "=" * 80)
    print(f"{'MODEL':<24} | {'TEST ACC':<10} | {'F1-MACRO':<10} | {'CV MEAN':<10} | {'FIT TIME':<10}")
    print("=" * 80)
    for model_name, metrics in benchmark_results.items():
        print(
            f"{model_name:<24} | "
            f"{metrics['test_accuracy']:<10.4f} | "
            f"{metrics['f1_macro']:<10.4f} | "
            f"{metrics['cv_accuracy_mean']:<10.4f} | "
            f"{metrics['fit_time_seconds']:<10.4f}s"
        )
    print("=" * 80)
