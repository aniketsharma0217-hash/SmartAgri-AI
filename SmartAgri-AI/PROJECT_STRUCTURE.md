# SmartAgri-AI: Project Structure

This document details the folder structure and component roles for **SmartAgri-AI: AI-Powered Crop Recommendation and Agricultural Decision Support System**.

```
SmartAgri-AI/
├── .gitignore                      # Git exclusion rules for Python, virtual environments, caches
├── LICENSE                         # Open-source MIT license
├── PROJECT_STRUCTURE.md            # Directory and file organization breakdown (this file)
├── README.md                       # Comprehensive project documentation and user guide
├── requirements.txt                # Exact pinned Python dependencies
│
├── app/                            # Application user interface (Streamlit frontend)
│   ├── app.py                      # Main multi-page application entry point
│   ├── assets/                     # Static styling, CSS, diagrams, and brand icons
│   └── components/                 # Reusable UI widgets, card components, form controls
│
├── data/                           # Data storage layer
│   ├── raw/                        # Untouched raw crop datasets (Crop_recommendation.csv)
│   └── processed/                  # Cleaned, validated, and preprocessed datasets
│
├── docs/                           # In-depth technical documentation
│   ├── architecture.md             # System architecture & component interaction flows
│   ├── dataset.md                  # Provenance, feature dictionaries, and distribution audit
│   ├── deployment.md               # Production deployment guide (Streamlit Cloud, Docker)
│   ├── methodology.md              # End-to-end ML lifecycle and scientific workflow
│   └── testing.md                  # Testing framework, test cases, and coverage guide
│
├── models/                         # Serialized ML artifacts & metadata
│   ├── model.joblib                # Production-selected trained classifier
│   ├── scaler.joblib               # Preprocessing standard scaler fitted on training split
│   ├── label_encoder.joblib        # Target label encoder
│   └── metadata.json               # Model versioning, hyperparameter records, metrics
│
├── notebooks/                      # Interactive research notebooks
│   └── exploratory_analysis.ipynb  # Comprehensive EDA, feature distributions, correlations
│
├── reports/                        # Evaluative reports and visual artifacts
│   ├── figures/                    # Generated charts (confusion matrix, ROC, feature importances)
│   ├── model_comparison.json       # Factual evaluation metrics across evaluated algorithms
│   ├── presentation.md             # 12-slide structured final presentation defense
│   └── project_report.md           # Formal 32-section capstone evaluation report
│
├── src/                            # Modular core production codebase
│   ├── __init__.py                 # Package marker
│   ├── data_loader.py              # Ingestion, schema validation, and missing value checks
│   ├── evaluate.py                 # Multi-metric evaluation and matrix generation
│   ├── predict.py                  # High-performance inference engine and explainability
│   ├── preprocessing.py           # Leak-free feature scaling and stratified train-test splits
│   ├── train.py                    # Multi-model training and cross-validation runner
│   └── utils.py                    # Helper utilities, logger configuration, constants
│
└── tests/                          # Automated test suite (pytest)
    ├── test_data.py                # Dataset schema and integrity tests
    ├── test_model.py               # Model behavior and cross-validation tests
    ├── test_prediction.py          # Inference pipeline and output verification tests
    ├── test_preprocessing.py       # Data split, leakage prevention, and transformation tests
    └── test_validation.py          # Input boundary validation and error rejection tests
```
