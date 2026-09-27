"""
Data Ingestion and Schema Validation Module for SmartAgri-AI.
Loads, validates, and audits the legitimate Crop Recommendation Dataset.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATASET_PATH,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    FEATURE_BOUNDS
)
from src.utils import get_logger

logger = get_logger("DataLoader")


class DatasetValidationError(ValueError):
    """Raised when the dataset fails schema or integrity validation."""
    pass


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads raw agricultural crop dataset from CSV with schema and integrity validation.

    Args:
        filepath: Optional path to the CSV file. Defaults to config.RAW_DATASET_PATH.

    Returns:
        pd.DataFrame: Validated raw DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetValidationError: If the schema or integrity check fails.
    """
    target_path = Path(filepath) if filepath else RAW_DATASET_PATH

    if not target_path.exists():
        msg = (
            f"Dataset not found at '{target_path}'. "
            f"Please download 'Crop_recommendation.csv' and place it in 'data/raw/'."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    if target_path.stat().st_size == 0:
        msg = f"Dataset file at '{target_path}' is empty (0 bytes) and cannot be loaded."
        logger.error(msg)
        raise DatasetValidationError(msg)

    logger.info(f"Loading raw dataset from: {target_path}")
    df = pd.read_csv(target_path)

    # 1. Validate Schema & Column Names
    expected_columns = set(NUMERICAL_FEATURES + [TARGET_COLUMN])
    actual_columns = set(df.columns)

    if not expected_columns.issubset(actual_columns):
        missing = expected_columns - actual_columns
        raise DatasetValidationError(f"Missing required columns in dataset: {missing}")

    # 2. Check Row and Column Count
    num_rows, num_cols = df.shape
    if num_rows == 0:
        raise DatasetValidationError("Dataset is empty (0 rows).")

    logger.info(f"Dataset successfully loaded: {num_rows} rows, {num_cols} columns.")
    return df


def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive statistical, missing value, and integrity audit on the dataset.

    Args:
        df: Loaded raw DataFrame.

    Returns:
        Dict[str, Any]: Comprehensive audit statistics.
    """
    missing_counts = df.isnull().sum().to_dict()
    duplicate_count = int(df.duplicated().sum())
    data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Target class distribution
    if TARGET_COLUMN in df.columns:
        class_counts = df[TARGET_COLUMN].value_counts().to_dict()
        unique_classes = sorted(df[TARGET_COLUMN].unique().tolist())
    else:
        class_counts = {}
        unique_classes = []

    # Numerical feature boundary audits
    feature_stats = {}
    for feat in NUMERICAL_FEATURES:
        if feat in df.columns:
            feature_stats[feat] = {
                "min": float(df[feat].min()),
                "max": float(df[feat].max()),
                "mean": float(df[feat].mean()),
                "median": float(df[feat].median()),
                "std": float(df[feat].std())
            }

    audit_results: Dict[str, Any] = {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "data_types": data_types,
        "total_missing_values": int(df.isnull().sum().sum()),
        "missing_per_column": missing_counts,
        "duplicate_rows": duplicate_count,
        "num_target_classes": len(unique_classes),
        "target_classes": unique_classes,
        "class_distribution": class_counts,
        "is_balanced": len(set(class_counts.values())) == 1 if class_counts else False,
        "feature_statistics": feature_stats
    }

    return audit_results


def validate_feature_ranges(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates whether the numerical features fall within realistic agricultural bounds.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, report)
    """
    anomalies = {}
    is_valid = True

    for feat, bounds in FEATURE_BOUNDS.items():
        if feat in df.columns:
            out_of_bounds = df[(df[feat] < bounds["min"]) | (df[feat] > bounds["max"])]
            if not out_of_bounds.empty:
                is_valid = False
                anomalies[feat] = {
                    "violations_count": len(out_of_bounds),
                    "min_observed": float(df[feat].min()),
                    "max_observed": float(df[feat].max()),
                    "allowed_min": bounds["min"],
                    "allowed_max": bounds["max"]
                }

    return is_valid, anomalies


if __name__ == "__main__":
    df = load_raw_data()
    audit = audit_dataset(df)
    is_valid, anomalies = validate_feature_ranges(df)

    print("\n" + "=" * 50)
    print("SMARTAGRI-AI DATASET AUDIT REPORT")
    print("=" * 50)
    print(f"Total Rows:            {audit['num_rows']}")
    print(f"Total Columns:         {audit['num_columns']}")
    print(f"Total Missing Values:  {audit['total_missing_values']}")
    print(f"Duplicate Rows:        {audit['duplicate_rows']}")
    print(f"Unique Crop Classes:   {audit['num_target_classes']}")
    print(f"Dataset Balanced:      {audit['is_balanced']}")
    print(f"Agronomic Range Valid: {is_valid}")
    if anomalies:
        print(f"Anomalies Detected:    {anomalies}")
    print("=" * 50)
