# SmartAgri AI — Screenshot Capture Checklist & Verification Guide

This document provides a complete checklist and step-by-step instructions for capturing the 12 recommended manual screenshots from the running SmartAgri AI application for presentation slide decks, capstone reports, or documentation repositories.

---

## 1. Application Startup Instructions

Before capturing screenshots, launch the application locally:

```bash
# 1. Open terminal and navigate to project root
cd /path/to/SmartAgri-AI

# 2. Activate Python virtual environment
source .venv/bin/activate

# 3. Launch Streamlit application
streamlit run app/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 2. 12-Point Screenshot Capture Checklist

| # | Screenshot Name | File Target | Recommended Capture View / User Action | Key UI Elements to Highlight |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Home Page Hero** | `assets/screenshots/01_home_page.png` | Navigate to **🏠 Home**. Capture top hero banner and problem statement. | SmartAgri AI logo, subtitle, problem statement, key feature badges. |
| **2** | **Recommendation Form** | `assets/screenshots/02_recommendation_form.png` | Navigate to **🌱 Crop Recommendation**. Capture input sliders and presets. | 7 parameter input controls ($N, P, K$, Temp, Humidity, pH, Rainfall), 5 preset buttons. |
| **3** | **Example Input Selection** | `assets/screenshots/03_example_input.png` | Click preset **"Cotton (Warm / High Nitrogen)"** on the recommendation form. | Populated input values ($N=120, P=40, K=20, \text{Temp}=24, \text{Humidity}=80, \text{pH}=6.8, \text{Rainfall}=80$). |
| **4** | **Prediction Result Card** | `assets/screenshots/04_prediction_result.png` | Click **Generate Crop Recommendation**. Capture top prediction output card. | Recommended crop title (**Cotton**), Model Confidence badge ($100.0\%$), input summary table. |
| **5** | **Runner-Up Alternatives** | `assets/screenshots/05_runner_up_alternatives.png` | Scroll to alternative recommendations under prediction result. | Top-3 candidate crops with calibrated probability progress bars. |
| **6** | **Prediction Explanation Table** | `assets/screenshots/06_prediction_explanation.png` | Scroll down to **Model Explainability & Agronomic Alignment**. | Narrative summary banner, feature comparison table (User Input vs. Crop Ideal Median, Alignment status). |
| **7** | **Model Comparison Dashboard** | `assets/screenshots/07_model_comparison.png` | Navigate to **📊 Model Insights**. Capture benchmark table. | Live benchmark comparison table across 5 models (Accuracy, Precision, Recall, F1-Score, Latency). |
| **8** | **Confusion Matrix Chart** | `assets/screenshots/08_confusion_matrix.png` | On **📊 Model Insights**, select **Confusion Matrix** view. | $22 \times 22$ heatmap showing perfect diagonal classification across 20/22 crops. |
| **9** | **Feature Importance Chart** | `assets/screenshots/09_feature_importance.png` | On **📊 Model Insights**, select **Feature Importance** view. | Horizontal bar chart displaying MDI importances (Rainfall 23.02%, Humidity 22.42%, Potassium 17.54%, etc.). |
| **10** | **Data Exploration Dashboard** | `assets/screenshots/10_data_exploration.png` | Navigate to **📁 Data Explorer**. Capture correlation heatmap. | Pearson correlation heatmap ($r=0.74$ between P and K) and univariate feature histograms. |
| **11** | **About & Disclaimers Page** | `assets/screenshots/11_about_disclaimers.png` | Navigate to **ℹ️ About**. Capture open data provenance and disclaimers. | Kaggle CC0 provenance attribution, technical methodology summary, Responsible AI disclaimers. |
| **12** | **Test Execution Output** | `assets/screenshots/12_test_execution.png` | In terminal, run `.venv/bin/pytest tests/ -v`. Capture successful test run. | Terminal output showing `74 passed in 4.42s` (100% pass rate). |

---

## 3. User Journey & UI Verification Log

All 15 verification steps have been tested and verified operational:

- [x] **1. Application Startup**: Starts cleanly with zero syntax or import errors.
- [x] **2. Home Page**: Displays hero banner, capabilities, problem overview, and disclaimers.
- [x] **3. Navigation**: Radio buttons switch seamlessly between all 5 pages.
- [x] **4. Recommendation Page**: Renders all 7 numerical inputs with correct units.
- [x] **5. Valid Input**: Presets auto-populate input fields reliably.
- [x] **6. Prediction Engine**: `CropPredictor` loads artifacts once and returns structured results.
- [x] **7. Result Display**: Displays recommended crop in enterprise green card styling.
- [x] **8. Confidence Display**: Calibrated probabilities displayed as percentages and progress bars.
- [x] **9. Explainability**: Displays narrative factor summary and centroid deviation alignment table.
- [x] **10. Model Insights**: Benchmark table and high-resolution charts render correctly.
- [x] **11. Data Visualization**: Exploratory charts display dataset statistics accurately.
- [x] **12. About Page**: Renders license, provenance, and ethical guidelines cleanly.
- [x] **13. Invalid Input Defense**: Rejects negative values, out-of-bound pH, and non-numeric entries with clear warning banners.
- [x] **14. Error Handling**: Exceptions caught gracefully; no raw Python stack traces leaked.
- [x] **15. Reset Functionality**: Presets and form submit buttons reset state smoothly via `st.rerun()`.
