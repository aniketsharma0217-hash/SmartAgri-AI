"""
Automated unit tests for data preprocessing, leakage prevention, and scaling.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing import PreprocessingPipeline, run_preprocessing_pipeline
from src.data_loader import load_raw_data
from src.config import (
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PREPROCESSOR_META_PATH
)


def test_stratified_split_shapes():
    """Verify that train/test split adheres to 80/20 proportion and preserves feature count."""
    df = load_raw_data()
    pipeline = PreprocessingPipeline(random_state=42, test_size=0.20)
    clean_df = pipeline.clean_data(df)
    X_train, X_test, y_train, y_test = pipeline.split_data(clean_df)

    assert X_train.shape == (1760, 7)
    assert X_test.shape == (440, 7)
    assert y_train.shape == (1760,)
    assert y_test.shape == (440,)


def test_stratified_split_balance():
    """Verify that stratification maintains exact 80/20 class proportions across all 22 classes."""
    df = load_raw_data()
    pipeline = PreprocessingPipeline(random_state=42, test_size=0.20)
    clean_df = pipeline.clean_data(df)
    _, _, y_train, y_test = pipeline.split_data(clean_df)

    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()

    assert len(train_counts) == 22
    assert len(test_counts) == 22
    assert all(count == 80 for count in train_counts.values)
    assert all(count == 20 for count in test_counts.values)


def test_data_leakage_prevention():
    """
    CRITICAL TEST: Verify that StandardScaler learns its parameters (mean_, var_)
    strictly from X_train, and NOT from the overall dataset or X_test.
    """
    df = load_raw_data()
    pipeline = PreprocessingPipeline(random_state=42, test_size=0.20)
    clean_df = pipeline.clean_data(df)
    X_train, X_test, y_train, y_test = pipeline.split_data(clean_df)
    X_train_scaled, _, _, _ = pipeline.fit_transform(X_train, y_train, X_test, y_test)

    # Scaler mean must match X_train mean exactly
    expected_means = X_train.mean().values
    actual_means = pipeline.scaler.mean_
    np.testing.assert_allclose(actual_means, expected_means, rtol=1e-5)

    # Scaler mean must NOT match full dataset mean (verifying isolation)
    full_dataset_means = df[NUMERICAL_FEATURES].mean().values
    assert not np.allclose(actual_means, full_dataset_means, atol=1e-3)


def test_scaler_standardization_properties():
    """Verify that X_train_scaled has approximately mean=0 and variance=1."""
    df = load_raw_data()
    pipeline = PreprocessingPipeline(random_state=42, test_size=0.20)
    clean_df = pipeline.clean_data(df)
    X_train, X_test, y_train, y_test = pipeline.split_data(clean_df)
    X_train_scaled, _, _, _ = pipeline.fit_transform(X_train, y_train, X_test, y_test)

    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-7)


def test_label_encoder_bidirectional_mapping():
    """Verify that LabelEncoder encodes all 22 classes and inverse transforms cleanly."""
    df = load_raw_data()
    pipeline = PreprocessingPipeline(random_state=42, test_size=0.20)
    clean_df = pipeline.clean_data(df)
    X_train, X_test, y_train, y_test = pipeline.split_data(clean_df)
    _, _, y_train_enc, _ = pipeline.fit_transform(X_train, y_train, X_test, y_test)

    assert len(pipeline.classes_) == 22
    assert set(np.unique(y_train_enc)) == set(range(22))

    # Verify perfect round-trip inverse transform
    recovered = pipeline.label_encoder.inverse_transform(y_train_enc)
    np.testing.assert_array_equal(recovered, y_train.values)


def test_pipeline_saved_artifacts():
    """Verify that all pipeline artifacts are properly serialized and non-empty."""
    summary = run_preprocessing_pipeline()
    assert Path(summary["scaler_path"]).exists()
    assert Path(summary["encoder_path"]).exists()
    assert Path(summary["meta_path"]).exists()
    assert TRAIN_DATA_PATH.exists()
    assert TEST_DATA_PATH.exists()


def test_transform_single_input():
    """Verify that a single input dictionary transforms correctly to (1, 7) shape."""
    pipeline = PreprocessingPipeline.load_pipeline()
    sample_input = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.88,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9
    }
    transformed = pipeline.transform_input(sample_input)
    assert transformed.shape == (1, 7)
    assert isinstance(transformed, np.ndarray)
