# SmartAgri AI — Security, Validation & Reliability Architecture

This document specifies the security controls, validation guarantees, error mitigation strategies, and environment configurations implemented in the SmartAgri AI crop recommendation system.

---

## 1. Threat Model & Design Philosophy

SmartAgri AI is designed as an agricultural decision-support service. In accordance with project requirements:
- **No Unnecessary Authentication**: The application does not collect user credentials, passwords, or personal identifying information (PII). Unauthenticated agricultural extension workers and farmers can utilize inference without identity overhead.
- **Zero Secrets Architecture**: The system does not require third-party API keys, cloud tokens, database passwords, or persistent session secrets.
- **Fail-Safe & Graceful Degradation**: In the event of invalid inputs, missing datasets, or corrupted artifacts, the system logs technical context internally and surfaces actionable, human-readable notifications without revealing raw stack traces or internal server paths.

---

## 2. Input Validation Architecture

To defend against injection attacks, malformed types, and agronomic anomalies, the inference service implements a rigorous **5-layer validation pipeline** in [`src/predict.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/src/predict.py) (`validate_input_data`):

```
User Input Dict
      │
      ▼
Layer 1: Type & Structure Check (Must be a non-empty dictionary)
      │
      ▼
Layer 2: Completeness Check (All 7 required features N, P, K, temp, humidity, pH, rainfall present)
      │
      ▼
Layer 3: Strict Non-Boolean Numeric Check (Rejects Python `bool` values True/False which are ints in Python)
      │
      ▼
Layer 4: Finite Numeric Check (Rejects NaN, positive/negative infinity, non-convertible strings)
      │
      ▼
Layer 5: Agronomic Domain Range Boundaries (Validates biological feasibility bounds)
      │
      ▼
Validated & Sanitized Feature Payload (Ready for StandardScaler)
```

### Agronomic Range Boundaries

| Feature | Unit | Min Allowed | Max Allowed | Agricultural Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **N** (Nitrogen) | kg/ha | 0.0 | 300.0 | Negative nutrient values are physically impossible; max threshold accommodates intensive fertilisation. |
| **P** (Phosphorus) | kg/ha | 0.0 | 200.0 | Soil Olsen/Bray extraction limits. |
| **K** (Potassium) | kg/ha | 0.0 | 300.0 | High-potash tropical soils upper bound. |
| **Temperature** | °C | 0.0 | 60.0 | Realistic global ambient agricultural growing temperatures. |
| **Relative Humidity** | % | 0.0 | 100.0 | Physical saturation boundary of water vapor in air. |
| **pH** | pH scale | 3.0 | 10.0 | Acid sulfate soils (3.0) to extreme sodic/saline soils (10.0). |
| **Rainfall** | mm | 0.0 | 500.0 | Arid zero-rain conditions to tropical monsoon peaks. |

---

## 3. Safe File Handling & Traversal Defense

### Path Traversal Mitigation (`safe_resolve_path`)
Defined in [`src/utils.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/src/utils.py):
- Verifies that any dynamic file resolution stays strictly bounded within permitted base directories.
- Resolves all symlinks and parent aliases (`..`) before validation.
- Prevents arbitrary file reads/writes outside project roots.

### Corrupted & Deserialization Defense (`safe_load_joblib`)
- Validates file existence prior to deserialization.
- Verifies file size: 0-byte truncated files are rejected immediately with a descriptive `RuntimeError`.
- Unpickling failures (e.g., malformed byte streams) are caught and converted to explicit application errors without crashing process threads.

### Dataset Integrity Guard (`load_raw_data`)
- Checks dataset existence and non-zero byte size before parsing.
- Verifies full feature schema against `NUMERICAL_FEATURES + [TARGET_COLUMN]`.
- Ensures zero rows triggers a structured `DatasetValidationError`.

---

## 4. Error Handling & Stack Trace Masking

1. **Prediction API (`predict_crop`)**:
   - Never raises raw unhandled exceptions to callers.
   - Returns a structured dictionary with `"status": "error"`, `"message": "..."`, and an optional `"errors"` map detailing each offending field.
2. **Streamlit UI Application (`app/app.py`)**:
   - Wraps prediction execution in top-level `try...except` blocks.
   - Logs detailed tracebacks to the application logger for system operators.
   - Displays clean, polite warning and error banners to end users without leaking Python line numbers, internal directory paths, or stack frames.

---

## 5. Configuration & Environment Variables

All path overrides and cache settings are centralized in [`src/config.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/src/config.py). If running in customized containerized or multi-tenant environments, the following environment variables may optionally be configured:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `SMARTAGRI_DATA_DIR` | `<PROJECT_ROOT>/data` | Root directory for raw and processed data partitions. |
| `SMARTAGRI_MODELS_DIR` | `<PROJECT_ROOT>/models` | Directory storing production serialized joblib models and metadata. |
| `SMARTAGRI_RAW_DATASET_PATH` | `<DATA_DIR>/raw/Crop_recommendation.csv` | Explicit absolute path to the raw dataset CSV. |
| `MPLCONFIGDIR` | `<PROJECT_ROOT>/.cache/matplotlib` | Writable directory for matplotlib font/glyph caches (avoids sandboxed home-directory errors). |

---

## 6. Audit & Automated Verification

Automated security and reliability verification is maintained through continuous pytest suites:
- [`tests/test_validation.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_validation.py): 9 tests covering empty payloads, boolean injections, NaN/infinity, negative nutrients, extreme temperatures, impossible pH, and humidity overflows.
- [`tests/test_security_and_reliability.py`](file:///Users/aniketkumar/.gemini/antigravity/scratch/SmartAgri-AI/tests/test_security_and_reliability.py): 8 tests verifying traversal prevention, corrupt model unpickling rejection, missing/empty dataset validation, and repository-wide regex scans for leaked secrets and credentials.
- **Full Suite Status**: 63/63 passing tests.
