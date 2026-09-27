# SmartAgri AI — Automated Testing & Quality Assurance Documentation

This document outlines the testing strategy, test module taxonomy, execution commands, empirical verification results, and technical limitations of the SmartAgri AI test suite.

---

## 1. Testing Strategy & Philosophy

The testing framework is built with **pytest 8.4.2** following strict production engineering standards:
- **Zero Fabrication**: All assertions execute against real historical data (`data/raw/Crop_recommendation.csv`) and genuine serialized model artifacts (`models/`).
- **Data Leakage Defense**: Split tests mathematically verify that test samples never influence training scalers or label distributions.
- **Fail-Safe Contract**: Every user-facing prediction and validation API guarantees structured dictionary responses, ensuring graceful failure modes without unhandled process crashes.
- **Security & Integrity**: Path traversal, file corruption, and credential leakage checks run as integral components of the standard testing routine.

---

## 2. Test Taxonomy & Coverage Mapping

The test suite spans **10 specialized test modules** in `tests/`:

| Module | Test Categories Covered | Number of Tests | Key Assertions / Verifications |
| :--- | :--- | :---: | :--- |
| [`test_data.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_data.py) | 1. Dataset loading<br>2. Dataset validation | 9 | File presence, $(2200 \times 8)$ shape, schema invariance, zero missing values, zero duplicate rows, 22 balanced classes ($100$ per class), physical range adherence, missing file error. |
| [`test_preprocessing.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_preprocessing.py) | 3. Preprocessing | 7 | Stratified 80/20 train-test split ($N_{\text{train}}=1760, N_{\text{test}}=440$), balance retention ($20$ per class in test), strict no-leakage verification (scaler fitted on $X_{\text{train}}$ only), unit variance/zero mean check, bijective label encoder mapping, single-instance transform contract. |
| [`test_validation.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_validation.py) | 4. Input validation<br>8. Invalid input | 9 | Rejection of non-dict types, rejection of empty dict `{}`, rejection of Python booleans (`True`/`False`), rejection of `NaN`/$\pm\infty$, negative nutrient bounds, extreme temperatures, impossible pH ($<3.0$ or $>10.0$), humidity overflows ($>100\%$), structured error payload integrity. |
| [`test_prediction.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_prediction.py) | 6. Prediction<br>7. Prediction output format<br>8. Invalid input | 11 | End-to-end inference on Rice and Cotton, CropPredictor singleton pattern, full output schema verification (`status`, `recommended_crop`, `confidence_score`, `confidence_percentage`, `top_alternatives`, `input_summary`, `explainability`, `explainability_report`), descending ranking of top 3 alternatives, explainability comparison fields. |
| [`test_model.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_model.py) | 5. Model loading | 5 | All 5 candidate models instantiated, candidate joblib serialization, benchmark metric boundaries $[0.0, 1.0]$, benchmark accuracy $>90\%$, selected production model availability and `predict_proba` support, metadata accuracy. |
| [`test_security_and_reliability.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_security_and_reliability.py) | 5. Model loading<br>9. Error handling | 8 | Path traversal mitigation (`safe_resolve_path`), safe joblib loader against missing files, empty 0-byte corrupt files, corrupted byte streams, missing raw dataset handling, empty dataset rejection, automated secrets regex audit across the entire codebase. |
| [`test_app_logic.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_app_logic.py) | 10. Important application logic | 8 | Parameterized validation and prediction verification for 5 agronomic presets (Rice, Cotton, Coffee, Apple, Chickpea), complete `FEATURE_BOUNDS` coverage across all 7 features, metadata consistency with comparison benchmarks, extreme min/max boundary input handling. |
| [`test_explainability.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_explainability.py) | 10. Important application logic | 3 | Gini MDI importance sum to $1.0$, non-causal narrative explanation generation, seamless integration with `predict_crop()`. |
| [`test_ui.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_ui.py) | 10. Important application logic | 3 | Streamlit entrypoint syntax check, stylesheet presence, all 8 EDA and evaluation figures present. |
| [`test_environment.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_environment.py) | Environment integrity | 11 | Python $\ge 3.10$ check, third-party libraries (numpy, pandas, sklearn, matplotlib, seaborn, joblib, streamlit, pytest), internal package imports (`src.config`, `src.utils`). |

---

## 3. Test Execution Commands

All commands are run from the project root directory (`SmartAgri-AI/`):

### Execute Entire Test Suite
```bash
.venv/bin/pytest tests/ -v
```

### Quick Run (Summary Only)
```bash
.venv/bin/pytest tests/ -q
```

### Module-Specific Invocations
```bash
# Data ingestion & schema checks
.venv/bin/pytest tests/test_data.py -v

# Preprocessing & leakage tests
.venv/bin/pytest tests/test_preprocessing.py -v

# Model & benchmark checks
.venv/bin/pytest tests/test_model.py -v

# Prediction pipeline & schema tests
.venv/bin/pytest tests/test_prediction.py -v

# 5-Layer validation suite
.venv/bin/pytest tests/test_validation.py -v

# Security, path traversal & fault tolerance
.venv/bin/pytest tests/test_security_and_reliability.py -v

# Application business logic & presets
.venv/bin/pytest tests/test_app_logic.py -v

# Model explainability suite
.venv/bin/pytest tests/test_explainability.py -v
```

---

## 4. Actual Measured Test Execution Results

> [!NOTE]
> Executed in environment: Python 3.12.3, macOS (Darwin arm64), pytest 8.4.2.

```
============================= test session starts ==============================
platform darwin -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0
rootdir: /Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI
configfile: pytest.ini
plugins: anyio-4.15.1
collected 74 items

tests/test_app_logic.py::test_application_presets_validity_and_recommendation[Rice] PASSED
tests/test_app_logic.py::test_application_presets_validity_and_recommendation[Cotton] PASSED
tests/test_app_logic.py::test_application_presets_validity_and_recommendation[Coffee] PASSED
tests/test_app_logic.py::test_application_presets_validity_and_recommendation[Apple] PASSED
tests/test_app_logic.py::test_application_presets_validity_and_recommendation[Chickpea] PASSED
tests/test_app_logic.py::test_feature_bounds_coverage PASSED
tests/test_app_logic.py::test_metadata_consistency_with_benchmarks PASSED
tests/test_app_logic.py::test_boundary_inputs_min_and_max PASSED
tests/test_data.py::test_raw_dataset_file_exists PASSED
tests/test_data.py::test_load_raw_data_shape PASSED
tests/test_data.py::test_dataset_columns_schema PASSED
tests/test_data.py::test_no_missing_values PASSED
tests/test_data.py::test_no_duplicate_records PASSED
tests/test_data.py::test_target_class_distribution PASSED
tests/test_data.py::test_feature_ranges_valid PASSED
tests/test_data.py::test_load_nonexistent_dataset_raises_error PASSED
tests/test_data.py::test_dataset_audit_metrics PASSED
tests/test_environment.py::test_python_version PASSED
tests/test_environment.py::test_third_party_imports[numpy] PASSED
tests/test_environment.py::test_third_party_imports[pandas] PASSED
tests/test_environment.py::test_third_party_imports[sklearn] PASSED
tests/test_environment.py::test_third_party_imports[matplotlib] PASSED
tests/test_environment.py::test_third_party_imports[seaborn] PASSED
tests/test_environment.py::test_third_party_imports[joblib] PASSED
tests/test_environment.py::test_third_party_imports[streamlit] PASSED
tests/test_environment.py::test_third_party_imports[pytest] PASSED
tests/test_environment.py::test_internal_config_import PASSED
tests/test_environment.py::test_internal_utils_import PASSED
tests/test_explainability.py::test_global_feature_importance_validity PASSED
tests/test_explainability.py::test_explanation_engine_instance_narrative PASSED
tests/test_explainability.py::test_predict_crop_integration_with_explainability PASSED
tests/test_model.py::test_candidate_models_structure PASSED
tests/test_model.py::test_candidate_model_serialization PASSED
tests/test_model.py::test_model_comparison_metrics_validity PASSED
tests/test_model.py::test_selected_production_model_exists PASSED
tests/test_model.py::test_production_model_metadata PASSED
tests/test_prediction.py::test_validate_input_data_success PASSED
tests/test_prediction.py::test_validate_input_data_missing_field PASSED
tests/test_prediction.py::test_validate_input_data_non_numeric PASSED
tests/test_prediction.py::test_validate_input_data_out_of_bounds PASSED
tests/test_prediction.py::test_predict_crop_success PASSED
tests/test_prediction.py::test_predict_crop_cotton PASSED
tests/test_prediction.py::test_predict_crop_error_handling PASSED
tests/test_prediction.py::test_crop_predictor_singleton PASSED
tests/test_prediction.py::test_prediction_output_format_exact_schema PASSED
tests/test_prediction.py::test_top_alternatives_ordering_and_format PASSED
tests/test_prediction.py::test_explainability_comparison_fields PASSED
tests/test_preprocessing.py::test_stratified_split_shapes PASSED
tests/test_preprocessing.py::test_stratified_split_balance PASSED
tests/test_preprocessing.py::test_data_leakage_prevention PASSED
tests/test_preprocessing.py::test_scaler_standardization_properties PASSED
tests/test_preprocessing.py::test_label_encoder_bidirectional_mapping PASSED
tests/test_preprocessing.py::test_pipeline_saved_artifacts PASSED
tests/test_preprocessing.py::test_transform_single_input PASSED
tests/test_security_and_reliability.py::test_safe_resolve_path_prevents_traversal PASSED
tests/test_security_and_reliability.py::test_safe_resolve_path_allows_legitimate_subpaths PASSED
tests/test_security_and_reliability.py::test_safe_load_joblib_missing_file PASSED
tests/test_security_and_reliability.py::test_safe_load_joblib_corrupt_empty_file PASSED
tests/test_security_and_reliability.py::test_safe_load_joblib_garbage_bytes PASSED
tests/test_security_and_reliability.py::test_data_loader_rejects_missing_dataset PASSED
tests/test_security_and_reliability.py::test_data_loader_rejects_empty_dataset PASSED
tests/test_security_and_reliability.py::test_repository_secrets_audit PASSED
tests/test_ui.py::test_app_file_integrity PASSED
tests/test_ui.py::test_custom_stylesheet_integrity PASSED
tests/test_ui.py::test_ui_figures_exist PASSED
tests/test_validation.py::test_validation_rejects_non_dict PASSED
tests/test_validation.py::test_validation_rejects_empty_dict PASSED
tests/test_validation.py::test_validation_rejects_boolean_values PASSED
tests/test_validation.py::test_validation_rejects_nan_and_infinity PASSED
tests/test_validation.py::test_validation_rejects_negative_nutrients PASSED
tests/test_validation.py::test_validation_rejects_extreme_temperatures PASSED
tests/test_validation.py::test_validation_rejects_impossible_ph PASSED
tests/test_validation.py::test_validation_rejects_impossible_humidity PASSED
tests/test_validation.py::test_predict_crop_error_structure PASSED

============================== 74 passed in 4.42s ==============================
```

- **Total Tests**: 74
- **Passed**: 74 (100.0%)
- **Failed**: 0
- **Skipped**: 0
- **Execution Time**: ~4.42 seconds

---

## 5. Known Limitations & Technical Constraints

1. **Artifact Dependency for Inference Tests**:
   - Integration tests in `test_prediction.py` and `test_app_logic.py` depend on the presence of physical serialized artifacts (`models/crop_recommender_model.joblib`, `models/scaler.joblib`, `models/label_encoder.joblib`).
   - If running in a fresh, untrained environment, `src/train.py` and `src/evaluate.py` must run prior to prediction integration tests.
2. **Deterministic Random States**:
   - Machine learning algorithms and splits use frozen seeds (`RANDOM_STATE = 42`). Assertions on probabilities and classification metrics assume deterministic execution. Floating-point variations across different underlying BLAS/LAPACK implementations may induce minute differences at higher decimal precision ($<10^{-6}$).
3. **Headless Streamlit UI Testing**:
   - Streamlit UI testing in `test_ui.py` verifies file syntactical integrity, layout scripts, and static asset references. Full end-to-end browser automation (e.g., via Playwright) is outside the scope of the standard unit/integration test suite.
