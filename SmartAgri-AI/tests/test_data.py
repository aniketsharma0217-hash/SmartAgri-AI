"""
Automated unit tests for data loading, schema validation, and integrity checks.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import load_raw_data, audit_dataset, validate_feature_ranges, DatasetValidationError
from src.config import RAW_DATASET_PATH, NUMERICAL_FEATURES, TARGET_COLUMN


def test_raw_dataset_file_exists():
    """Verify that the raw dataset file exists and is non-empty."""
    assert RAW_DATASET_PATH.exists(), f"Dataset file missing at {RAW_DATASET_PATH}"
    assert RAW_DATASET_PATH.stat().st_size > 0, "Dataset file is 0 bytes"


def test_load_raw_data_shape():
    """Verify that the loaded dataset has the exact expected dimensions (2200 rows, 8 columns)."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2200, 8), f"Expected (2200, 8), got {df.shape}"


def test_dataset_columns_schema():
    """Verify that all required features and the target label are present."""
    df = load_raw_data()
    expected_cols = NUMERICAL_FEATURES + [TARGET_COLUMN]
    for col in expected_cols:
        assert col in df.columns, f"Column {col} missing from dataset"


def test_no_missing_values():
    """Verify that the dataset contains zero null or NaN values."""
    df = load_raw_data()
    total_nulls = df.isnull().sum().sum()
    assert total_nulls == 0, f"Found {total_nulls} missing values in dataset"


def test_no_duplicate_records():
    """Verify that there are zero duplicate rows in the dataset."""
    df = load_raw_data()
    dup_count = df.duplicated().sum()
    assert dup_count == 0, f"Found {dup_count} duplicate rows in dataset"


def test_target_class_distribution():
    """Verify that there are exactly 22 target classes, each with 100 samples (balanced)."""
    df = load_raw_data()
    counts = df[TARGET_COLUMN].value_counts()
    assert len(counts) == 22, f"Expected 22 crop classes, got {len(counts)}"
    assert all(count == 100 for count in counts.values), "Classes are not balanced at 100 samples each"


def test_feature_ranges_valid():
    """Verify that all feature values fall within realistic agronomic boundaries."""
    df = load_raw_data()
    is_valid, anomalies = validate_feature_ranges(df)
    assert is_valid is True, f"Feature range anomalies detected: {anomalies}"


def test_load_nonexistent_dataset_raises_error(tmp_path):
    """Verify that attempting to load a non-existent path raises FileNotFoundError."""
    missing_path = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_data(missing_path)


def test_dataset_audit_metrics():
    """Verify that the audit_dataset function extracts correct summary metadata."""
    df = load_raw_data()
    audit = audit_dataset(df)
    assert audit["num_rows"] == 2200
    assert audit["num_columns"] == 8
    assert audit["is_balanced"] is True
    assert audit["total_missing_values"] == 0
    assert audit["duplicate_rows"] == 0
