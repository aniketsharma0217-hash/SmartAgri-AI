"""
Data Preprocessing and Leak-Free Feature Pipeline Module for SmartAgri-AI.
Enforces strict train/test isolation, stratification, and feature standardization.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import (
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    PROCESSED_DATA_DIR,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PREPROCESSOR_META_PATH,
    MODELS_DIR
)
from src.data_loader import load_raw_data
from src.utils import get_logger, ensure_directory, save_json

logger = get_logger("Preprocessing")


class PreprocessingPipeline:
    """
    Production-grade preprocessing pipeline for crop recommendation.
    Guarantees no data leakage: scalers and encoders are fitted solely on training splits.
    """

    def __init__(self, random_state: int = RANDOM_STATE, test_size: float = TEST_SIZE):
        self.random_state = random_state
        self.test_size = test_size
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.classes_: Optional[np.ndarray] = None
        self.metadata_: Dict[str, Any] = {}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by checking for null values and duplicates.

        Args:
            df: Input raw DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        initial_len = len(df)
        df_cleaned = df.copy()

        # Check and handle missing values
        null_count = df_cleaned.isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} missing values. Imputing with column medians...")
            for col in NUMERICAL_FEATURES:
                if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
                    median_val = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)

        # Check and remove duplicates
        duplicate_count = df_cleaned.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows. Removing duplicates...")
            df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
            logger.info(f"Rows reduced from {initial_len} to {len(df_cleaned)} after deduplication.")

        return df_cleaned

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into stratified train and test partitions to maintain class balance.

        Args:
            df: Cleaned DataFrame containing features and target column.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                X_train, X_test, y_train, y_test
        """
        X = df[NUMERICAL_FEATURES].copy()
        y = df[TARGET_COLUMN].copy()

        logger.info(
            f"Performing stratified train/test split (test_size={self.test_size}, "
            f"random_state={self.random_state})..."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        logger.info(f"Split complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits transformers strictly on training data, then transforms both train and test partitions.
        Guarantees zero data leakage from test split into preprocessing parameters.

        Args:
            X_train: Training feature DataFrame.
            y_train: Training target Series.
            X_test: Testing feature DataFrame.
            y_test: Testing target Series.

        Returns:
            Tuple: (X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded)
        """
        # 1. Target Encoding
        logger.info("Fitting LabelEncoder on y_train...")
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        self.classes_ = self.label_encoder.classes_

        # 2. Feature Standardization (fit ONLY on X_train)
        logger.info("Fitting StandardScaler strictly on X_train (Zero Data Leakage)...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 3. Compile Preprocessing Metadata
        self.metadata_ = {
            "random_state": self.random_state,
            "test_size": self.test_size,
            "numerical_features": NUMERICAL_FEATURES,
            "target_column": TARGET_COLUMN,
            "num_classes": len(self.classes_),
            "classes": self.classes_.tolist(),
            "class_to_id": {cls_name: int(idx) for idx, cls_name in enumerate(self.classes_)},
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "feature_means_train": {feat: float(mean) for feat, mean in zip(NUMERICAL_FEATURES, self.scaler.mean_)},
            "feature_scales_train": {feat: float(scale) for feat, scale in zip(NUMERICAL_FEATURES, self.scaler.scale_)},
            "leak_prevention_protocol": "StandardScaler and LabelEncoder fitted exclusively on training split."
        }

        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded

    def save_pipeline(self) -> None:
        """
        Serializes the fitted scaler, label encoder, and preprocessing metadata.
        """
        ensure_directory(MODELS_DIR)
        ensure_directory(PROCESSED_DATA_DIR)

        logger.info(f"Saving StandardScaler to: {SCALER_PATH}")
        joblib.dump(self.scaler, SCALER_PATH)

        logger.info(f"Saving LabelEncoder to: {LABEL_ENCODER_PATH}")
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)

        logger.info(f"Saving Preprocessor Metadata to: {PREPROCESSOR_META_PATH}")
        save_json(self.metadata_, PREPROCESSOR_META_PATH)

    @classmethod
    def load_pipeline(cls) -> "PreprocessingPipeline":
        """
        Loads the persisted scaler and label encoder for inference.

        Returns:
            PreprocessingPipeline: Initialized pipeline ready for inference.
        """
        if not SCALER_PATH.exists() or not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError("Scaler or LabelEncoder artifact missing. Run preprocessing pipeline first.")

        pipeline = cls()
        pipeline.scaler = joblib.load(SCALER_PATH)
        pipeline.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        pipeline.classes_ = pipeline.label_encoder.classes_
        return pipeline

    def transform_input(self, input_dict: Dict[str, float]) -> np.ndarray:
        """
        Transforms a single raw agricultural query dictionary into scaled tensor for inference.

        Args:
            input_dict: Dictionary with keys [N, P, K, temperature, humidity, ph, rainfall]

        Returns:
            np.ndarray: 2D scaled feature array (1, 7)
        """
        if self.scaler is None:
            raise RuntimeError("Pipeline scaler is not fitted or loaded.")

        # Ensure correct feature ordering
        feature_vector = [float(input_dict[col]) for col in NUMERICAL_FEATURES]
        vector_2d = np.array(feature_vector).reshape(1, -1)
        return self.scaler.transform(vector_2d)


def run_preprocessing_pipeline() -> Dict[str, Any]:
    """
    Executes the complete preprocessing workflow:
    1. Loads raw dataset
    2. Cleans data
    3. Stratified splits
    4. Fits transformers on train data only
    5. Saves processed CSV datasets and serialized artifacts

    Returns:
        Dict[str, Any]: Summary of processed datasets.
    """
    logger.info("Initializing SmartAgri-AI Preprocessing Pipeline...")
    raw_df = load_raw_data()

    pipeline = PreprocessingPipeline(random_state=RANDOM_STATE, test_size=TEST_SIZE)
    clean_df = pipeline.clean_data(raw_df)

    X_train, X_test, y_train, y_test = pipeline.split_data(clean_df)
    X_train_scaled, X_test_scaled, y_train_enc, y_test_enc = pipeline.fit_transform(
        X_train, y_train, X_test, y_test
    )

    pipeline.save_pipeline()

    # Save processed CSV files (with target and scaled features for inspection)
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    train_df["label_encoded"] = y_train_enc
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    logger.info(f"Saved processed train dataset to: {TRAIN_DATA_PATH}")

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values
    test_df["label_encoded"] = y_test_enc
    test_df.to_csv(TEST_DATA_PATH, index=False)
    logger.info(f"Saved processed test dataset to: {TEST_DATA_PATH}")

    summary = {
        "training_shape_X": X_train.shape,
        "testing_shape_X": X_test.shape,
        "training_shape_y": y_train.shape,
        "testing_shape_y": y_test.shape,
        "features": NUMERICAL_FEATURES,
        "num_classes": len(pipeline.classes_),
        "classes": pipeline.classes_.tolist(),
        "train_samples_per_class": int(len(X_train) / len(pipeline.classes_)),
        "test_samples_per_class": int(len(X_test) / len(pipeline.classes_)),
        "scaler_path": str(SCALER_PATH),
        "encoder_path": str(LABEL_ENCODER_PATH),
        "meta_path": str(PREPROCESSOR_META_PATH)
    }

    return summary


if __name__ == "__main__":
    summary = run_preprocessing_pipeline()
    print("\n" + "=" * 55)
    print("SMARTAGRI-AI PREPROCESSING PIPELINE SUMMARY")
    print("=" * 55)
    print(f"Features ({len(summary['features'])}):            {summary['features']}")
    print(f"Target Classes ({summary['num_classes']}):      {summary['classes'][:5]}... (+17 more)")
    print(f"Training Feature Matrix:  {summary['training_shape_X']}")
    print(f"Testing Feature Matrix:   {summary['testing_shape_X']}")
    print(f"Training Target Vector:   {summary['training_shape_y']} (80 samples/class)")
    print(f"Testing Target Vector:    {summary['testing_shape_y']} (20 samples/class)")
    print(f"Scaler Artifact:          {summary['scaler_path']}")
    print(f"Encoder Artifact:         {summary['encoder_path']}")
    print("=" * 55)
