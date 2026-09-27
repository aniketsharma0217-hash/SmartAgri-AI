# Environment and Dependency Setup Guide

This document provides instructions for setting up, activating, and verifying the Python environment for **SmartAgri-AI**.

---

## 1. Recommended Python Version

* **Target Runtime**: **Python 3.12** (specifically `Python 3.12.3` or any `3.12.x`)
* **Minimum Supported Version**: Python 3.10
* **Architecture**: Compatible with macOS (Apple Silicon `arm64` and Intel `x86_64`), Linux, and Windows.

> [!NOTE]
> Python 3.12 is chosen for high runtime performance and optimal pre-compiled binary wheel support for `scikit-learn`, `numpy`, `pandas`, and `streamlit` on Apple Silicon.

---

## 2. Environment Creation

It is strongly recommended to use Python's built-in `venv` module to isolate project dependencies.

### macOS / Linux
```bash
# Navigate to the project root
cd SmartAgri-AI

# Create a virtual environment named .venv
python3.12 -m venv .venv
```

### Windows (PowerShell)
```powershell
# Navigate to the project root
cd SmartAgri-AI

# Create a virtual environment named .venv
py -3.12 -m venv .venv
```

---

## 3. How to Activate the Environment

### macOS / Linux
```bash
source .venv/bin/activate
```
*Your terminal prompt will indicate active environment:* `(.venv) user@host:~/SmartAgri-AI$`

### Windows
* **Command Prompt (cmd.exe)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **PowerShell**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

To deactivate when finished:
```bash
deactivate
```

---

## 4. Dependency Installation

With the virtual environment activated, install the exact pinned dependencies:

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### Core Dependency Inventory
| Package | Installed Version | Primary Role |
| :--- | :--- | :--- |
| `numpy` | `1.26.4` | Numerical tensor operations & array transformations |
| `pandas` | `2.3.3` | Tabular data manipulation, ingestion, and validation |
| `scikit-learn` | `1.5.2` | Machine learning classification models, metrics & preprocessing scalers |
| `joblib` | `1.4.2` | High-efficiency model serialization and persistence |
| `matplotlib` | `3.11.2` | Static graph generation and figure rendering |
| `seaborn` | `0.13.2` | Statistical data visualization (heatmaps, distributions) |
| `streamlit` | `1.64.0` | Modern, responsive web application framework |
| `pytest` | `8.4.2` | Automated testing framework |

---

## 5. How to Verify the Installation

Run the automated verification script to confirm all libraries, configurations, and internal modules import without error:

```bash
python -c "
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn as sns
import joblib
import streamlit
import pytest
import src.config as cfg
import src.utils as utils

print('=== DEPENDENCY VERIFICATION SUMMARY ===')
print('Python:       ', sys.version.split()[0])
print('NumPy:        ', np.__version__)
print('Pandas:       ', pd.__version__)
print('Scikit-learn: ', sklearn.__version__)
print('Matplotlib:   ', matplotlib.__version__)
print('Seaborn:      ', sns.__version__)
print('Joblib:       ', joblib.__version__)
print('Streamlit:    ', streamlit.__version__)
print('Pytest:       ', pytest.__version__)
print('Project Root: ', cfg.PROJECT_ROOT)
print('STATUS:       ', 'SUCCESSFUL ENVIRONMENT SETUP')
"
```

---

## 6. Security and API Key Configuration

* **Zero External API Keys**: SmartAgri-AI is completely self-contained. It performs inference locally and does **NOT** require OpenAI, Gemini, Hugging Face, or cloud API keys.
* **No Secrets Committed**: Hardcoded credentials, tokens, or passwords are strictly excluded.
* **Environment Overrides (Optional)**: If you wish to customize storage locations, you may set the following optional environment variables:
  * `SMARTAGRI_DATA_DIR`: Custom path to data folder.
  * `SMARTAGRI_MODELS_DIR`: Custom path to serialized models folder.
